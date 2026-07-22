from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baselines.data.loader import episode_field, select_episode_splits
from baselines.data.observation import (
    goal_conditioned_observation_schema,
    vectorize_goal_conditioned_observation,
)


@dataclass(frozen=True)
class GCRLEpisode:
    variant: str
    states: np.ndarray
    goals: np.ndarray
    actions: np.ndarray

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class _VariantArrays:
    states: np.ndarray
    goals: np.ndarray
    actions: np.ndarray
    anchor_state_indices: np.ndarray
    final_state_indices: np.ndarray

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class PreparedGCRLDatasets:
    train: "GCRLDataset"
    validation: "GCRLDataset"
    manifest: dict


@dataclass(frozen=True)
class GCRLNormalizer:
    state_mean: np.ndarray
    state_std: np.ndarray
    goal_mean: np.ndarray
    goal_std: np.ndarray

    @classmethod
    def fit(cls, dataset: "GCRLDataset") -> "GCRLNormalizer":
        state_count = 0
        goal_count = 0
        state_sum = None
        state_square_sum = None
        goal_sum = None
        goal_square_sum = None
        for arrays in dataset.variants.values():
            states = np.asarray(arrays.states, dtype=np.float64)
            goals = np.asarray(arrays.goals, dtype=np.float64)
            state_sum = states.sum(axis=0) if state_sum is None else state_sum + states.sum(axis=0)
            state_square_sum = (
                np.square(states).sum(axis=0)
                if state_square_sum is None
                else state_square_sum + np.square(states).sum(axis=0)
            )
            goal_sum = goals.sum(axis=0) if goal_sum is None else goal_sum + goals.sum(axis=0)
            goal_square_sum = (
                np.square(goals).sum(axis=0)
                if goal_square_sum is None
                else goal_square_sum + np.square(goals).sum(axis=0)
            )
            state_count += len(states)
            goal_count += len(goals)
        if state_count < 1 or goal_count < 1:
            raise ValueError("Cannot fit a GCRL normalizer on an empty dataset")
        state_mean = state_sum / state_count
        goal_mean = goal_sum / goal_count
        state_var = np.maximum(state_square_sum / state_count - np.square(state_mean), 0.0)
        goal_var = np.maximum(goal_square_sum / goal_count - np.square(goal_mean), 0.0)
        state_std = np.sqrt(state_var)
        goal_std = np.sqrt(goal_var)
        state_std = np.where(state_std < 1e-6, 1.0, state_std)
        goal_std = np.where(goal_std < 1e-6, 1.0, goal_std)
        return cls(
            state_mean=np.asarray(state_mean, dtype=np.float32),
            state_std=np.asarray(state_std, dtype=np.float32),
            goal_mean=np.asarray(goal_mean, dtype=np.float32),
            goal_std=np.asarray(goal_std, dtype=np.float32),
        )

    @classmethod
    def identity(cls, state_dim: int, goal_dim: int) -> "GCRLNormalizer":
        return cls(
            state_mean=np.zeros(state_dim, dtype=np.float32),
            state_std=np.ones(state_dim, dtype=np.float32),
            goal_mean=np.zeros(goal_dim, dtype=np.float32),
            goal_std=np.ones(goal_dim, dtype=np.float32),
        )

    def normalize_states(self, values) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.state_mean) / self.state_std

    def normalize_goals(self, values) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.goal_mean) / self.goal_std

    def normalize_batch(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        state_keys = {
            "observations",
            "next_observations",
            "high_actor_target_states",
        }
        goal_keys = {
            "value_goals",
            "actor_goals",
            "low_actor_goals",
            "high_actor_goals",
            "high_actor_target_goals",
        }
        normalized = {}
        for key, value in batch.items():
            if key in state_keys:
                normalized[key] = self.normalize_states(value)
            elif key in goal_keys:
                normalized[key] = self.normalize_goals(value)
            else:
                normalized[key] = np.asarray(value)
        return normalized

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "goal_mean": self.goal_mean.tolist(),
            "goal_std": self.goal_std.tolist(),
        }


class GCRLDataset:
    """Compact episodic arrays with within-variant goal relabeling.

    Every sampled batch comes from one variant. This prevents CRL's in-batch
    negatives and HIQL's random goals from crossing incompatible maze maps.
    """

    def __init__(self, episodes: list[GCRLEpisode], *, seed: int):
        if not episodes:
            raise ValueError("GCRL dataset requires at least one episode")
        grouped: dict[str, list[GCRLEpisode]] = {}
        for episode in episodes:
            grouped.setdefault(episode.variant, []).append(episode)
        self.variants = {
            variant: self._pack_variant(items) for variant, items in grouped.items()
        }
        self._variant_names = list(self.variants)
        counts = np.asarray(
            [self.variants[name].transition_count for name in self._variant_names],
            dtype=np.float64,
        )
        self._variant_probabilities = counts / counts.sum()
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _pack_variant(episodes: list[GCRLEpisode]) -> _VariantArrays:
        states = []
        goals = []
        actions = []
        anchor_indices = []
        final_indices = []
        state_offset = 0
        for episode in episodes:
            step_count = episode.transition_count
            states.append(episode.states)
            goals.append(episode.goals)
            actions.append(episode.actions)
            anchor_indices.append(state_offset + np.arange(step_count, dtype=np.int64))
            final_indices.append(
                np.full(step_count, state_offset + step_count, dtype=np.int64)
            )
            state_offset += step_count + 1
        return _VariantArrays(
            states=np.concatenate(states, axis=0),
            goals=np.concatenate(goals, axis=0),
            actions=np.concatenate(actions, axis=0),
            anchor_state_indices=np.concatenate(anchor_indices, axis=0),
            final_state_indices=np.concatenate(final_indices, axis=0),
        )

    @property
    def episode_count(self) -> int:
        return sum(
            int(np.count_nonzero(np.diff(arrays.final_state_indices, prepend=-1)))
            for arrays in self.variants.values()
        )

    @property
    def transition_count(self) -> int:
        return sum(arrays.transition_count for arrays in self.variants.values())

    @property
    def state_dim(self) -> int:
        return int(next(iter(self.variants.values())).states.shape[-1])

    @property
    def goal_dim(self) -> int:
        return int(next(iter(self.variants.values())).goals.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(next(iter(self.variants.values())).actions.shape[-1])

    def _sample_goal_indices(
        self,
        arrays: _VariantArrays,
        anchors: np.ndarray,
        finals: np.ndarray,
        *,
        p_current: float,
        p_future: float,
        p_random: float,
        geometric: bool,
        discount: float,
    ) -> np.ndarray:
        del p_random  # The remainder after current/future is the random branch.
        random_goals = self._rng.integers(0, len(arrays.states), size=len(anchors))
        if geometric:
            offsets = self._rng.geometric(p=1.0 - discount, size=len(anchors))
            future_goals = np.minimum(anchors + offsets, finals)
        else:
            distances = self._rng.random(len(anchors))
            first_future = np.minimum(anchors + 1, finals)
            future_goals = np.rint(
                first_future * distances + finals * (1.0 - distances)
            ).astype(np.int64)
        if p_current == 1.0:
            return anchors.copy()
        choose_future = self._rng.random(len(anchors)) < p_future / (1.0 - p_current)
        result = np.where(choose_future, future_goals, random_goals)
        choose_current = self._rng.random(len(anchors)) < p_current
        return np.where(choose_current, anchors, result)

    def sample(self, batch_size: int, *, algorithm: str, config: dict) -> dict[str, np.ndarray]:
        variant = self._rng.choice(
            self._variant_names, p=self._variant_probabilities
        )
        arrays = self.variants[str(variant)]
        transition_indices = self._rng.integers(
            0, arrays.transition_count, size=batch_size
        )
        anchors = arrays.anchor_state_indices[transition_indices]
        finals = arrays.final_state_indices[transition_indices]
        value_goal_indices = self._sample_goal_indices(
            arrays,
            anchors,
            finals,
            p_current=config["value_p_curgoal"],
            p_future=config["value_p_trajgoal"],
            p_random=config["value_p_randomgoal"],
            geometric=config["value_geom_sample"],
            discount=config["discount"],
        )
        successes = (anchors == value_goal_indices).astype(np.float32)
        batch = {
            "observations": arrays.states[anchors],
            "next_observations": arrays.states[anchors + 1],
            "actions": arrays.actions[transition_indices],
            "value_goals": arrays.goals[value_goal_indices],
            "masks": 1.0 - successes,
            "rewards": successes - (1.0 if algorithm == "hiql" else 0.0),
        }
        if algorithm == "crl":
            actor_goal_indices = self._sample_goal_indices(
                arrays,
                anchors,
                finals,
                p_current=config["actor_p_curgoal"],
                p_future=config["actor_p_trajgoal"],
                p_random=config["actor_p_randomgoal"],
                geometric=config["actor_geom_sample"],
                discount=config["discount"],
            )
            batch["actor_goals"] = arrays.goals[actor_goal_indices]
        elif algorithm == "hiql":
            subgoal_steps = config["subgoal_steps"]
            low_goal_indices = np.minimum(anchors + subgoal_steps, finals)
            if config["actor_geom_sample"]:
                offsets = self._rng.geometric(
                    p=1.0 - config["discount"], size=batch_size
                )
                high_future_goals = np.minimum(anchors + offsets, finals)
            else:
                distances = self._rng.random(batch_size)
                first_future = np.minimum(anchors + 1, finals)
                high_future_goals = np.rint(
                    first_future * distances + finals * (1.0 - distances)
                ).astype(np.int64)
            high_future_targets = np.minimum(
                anchors + subgoal_steps, high_future_goals
            )
            high_random_goals = self._rng.integers(
                0, len(arrays.states), size=batch_size
            )
            high_random_targets = np.minimum(anchors + subgoal_steps, finals)
            choose_random = (
                self._rng.random(batch_size) < config["actor_p_randomgoal"]
            )
            high_goal_indices = np.where(
                choose_random, high_random_goals, high_future_goals
            )
            high_target_indices = np.where(
                choose_random, high_random_targets, high_future_targets
            )
            batch.update(
                {
                    "low_actor_goals": arrays.goals[low_goal_indices],
                    "high_actor_goals": arrays.goals[high_goal_indices],
                    "high_actor_target_states": arrays.states[high_target_indices],
                    "high_actor_target_goals": arrays.goals[high_target_indices],
                }
            )
        else:
            raise ValueError(f"Unsupported GCRL algorithm: {algorithm!r}")
        return {
            key: np.asarray(value, dtype=np.float32) for key, value in batch.items()
        }


def _convert_episode(
    episode: Any,
    *,
    env_family: str,
    variant: str,
    observation_config: dict,
) -> GCRLEpisode:
    observations = episode_field(episode, "observations")
    if not isinstance(observations, dict):
        try:
            observations = dict(observations)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{env_family} GCRL requires dict observations"
            ) from exc
    if env_family == "pointmaze":
        achieved_xy = np.asarray(observations["observation"], dtype=np.float32)[
            ..., :2
        ]
    else:
        achieved_xy = np.asarray(observations["achieved_goal"], dtype=np.float32)[
            ..., :2
        ]
    states, goals = vectorize_goal_conditioned_observation(
        observations,
        env_family,
        observation_config=observation_config,
        variant=variant,
        goal_xy=achieved_xy,
    )
    actions = np.asarray(episode_field(episode, "actions"), dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    expected_action_dim = 2 if env_family == "pointmaze" else 8
    if actions.ndim != 2 or actions.shape[1] != expected_action_dim:
        raise ValueError(
            f"Unexpected {env_family} action shape for variant={variant!r}: {actions.shape}"
        )
    if len(states) != len(actions) + 1 or len(goals) != len(actions) + 1:
        raise ValueError(
            f"GCRL episode for variant={variant!r} must have T+1 observations and T actions"
        )
    if len(actions) < 1:
        raise ValueError(f"Episode for variant={variant!r} has no transitions")
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"Actions for variant={variant!r} contain non-finite values")
    if np.any(actions < -1.0001) or np.any(actions > 1.0001):
        raise ValueError(f"Actions for variant={variant!r} fall outside [-1, 1]")
    return GCRLEpisode(
        variant=variant,
        states=np.asarray(states, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        actions=np.clip(actions, -1.0, 1.0).astype(np.float32),
    )


def prepare_gcrl_datasets(
    config: dict,
    selected_variants: list[str],
    reward_types: dict[str, str],
) -> PreparedGCRLDatasets:
    selected = select_episode_splits(config, selected_variants, reward_types)
    train_episodes = []
    validation_episodes = []
    manifest_variants = {}
    for variant in selected_variants:
        split = selected.variants[variant]
        converted_train = [
            _convert_episode(
                split.loaded.episodes[index],
                env_family=config["env_family"],
                variant=variant,
                observation_config=config["observation"],
            )
            for index in split.train_indices
        ]
        converted_validation = [
            _convert_episode(
                split.loaded.episodes[index],
                env_family=config["env_family"],
                variant=variant,
                observation_config=config["observation"],
            )
            for index in split.validation_indices
        ]
        train_episodes.extend(converted_train)
        validation_episodes.extend(converted_validation)
        manifest_variants[variant] = {
            "source": split.loaded.source,
            "dataset_path": split.loaded.dataset_path,
            "reward_type": split.loaded.reward_type,
            "total_episodes": len(split.loaded.episodes),
            "initial_sampled_episode_target": split.initial_sampled_target,
            "sampled_episode_count": split.sampled_target,
            "train_episode_count": len(converted_train),
            "validation_episode_count": len(converted_validation),
            "train_transition_count": sum(
                episode.transition_count for episode in converted_train
            ),
            "validation_transition_count": sum(
                episode.transition_count for episode in converted_validation
            ),
            "train_episode_indices": split.train_indices,
            "validation_episode_indices": split.validation_indices,
        }
    train = GCRLDataset(train_episodes, seed=config["seed"])
    validation = GCRLDataset(
        validation_episodes, seed=config["seed"] + 1_000_003
    )
    manifest = {
        "env_family": config["env_family"],
        "observation_config": dict(config["observation"]),
        "observation_schema": goal_conditioned_observation_schema(
            config["env_family"], config["observation"]
        ),
        "goal_relabeling_scope": "within_variant",
        "batch_variant_policy": "one_variant_per_batch",
        "sampling_seed": config["sampling_seed"],
        "train_data_ratio": config["train_data_ratio"],
        "balance_variant_episode_count": bool(
            selected.balanced_episode_target is not None
        ),
        "balanced_episode_target": selected.balanced_episode_target,
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "train_transition_count": train.transition_count,
        "validation_transition_count": validation.transition_count,
        "variants": manifest_variants,
        "warnings": selected.warnings,
    }
    return PreparedGCRLDatasets(
        train=train,
        validation=validation,
        manifest=manifest,
    )
