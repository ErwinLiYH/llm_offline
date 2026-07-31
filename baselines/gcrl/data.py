from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from baselines.data.loader import episode_field, select_episode_splits
from baselines.data.observation import (
    GCRL_GOAL_SEMANTICS,
    GOAL_CONDITIONED_STATE_DIMS,
    family_map_shape,
    gcrl_dynamic_map_indices,
    goal_conditioned_observation_schema,
    vectorize_gcrl_state_observation,
)


@dataclass(frozen=True)
class GCRLEpisode:
    variant: str
    states: np.ndarray
    actions: np.ndarray
    # The map is invariant for every time step of one variant.  Keep it once
    # instead of repeating it in every offline state (especially important for
    # the 65M-transition multi-map AntMaze suite), then restore the original
    # vector order only when a minibatch is sampled.
    map_features: np.ndarray | None = None
    map_insert_index: int | None = None
    dynamic_map_insert_index: int | None = None
    dynamic_position_indices: np.ndarray | None = None
    dynamic_goal_indices: np.ndarray | None = None

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class _VariantArrays:
    states: np.ndarray
    actions: np.ndarray
    anchor_state_indices: np.ndarray
    final_state_indices: np.ndarray
    map_features: np.ndarray | None
    map_insert_index: int | None
    dynamic_map_insert_index: int | None
    dynamic_position_indices: np.ndarray | None
    dynamic_goal_indices: np.ndarray | None

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
    observation_mean: np.ndarray
    observation_std: np.ndarray

    @classmethod
    def fit(cls, dataset: "GCRLDataset") -> "GCRLNormalizer":
        state_count = 0
        state_sum = None
        state_square_sum = None
        for arrays in dataset.variants.values():
            states = np.asarray(arrays.states, dtype=np.float64)
            current_state_sum, current_state_square_sum = dataset.state_statistics(
                arrays
            )
            state_sum = (
                current_state_sum
                if state_sum is None
                else state_sum + current_state_sum
            )
            state_square_sum = (
                current_state_square_sum
                if state_square_sum is None
                else state_square_sum + current_state_square_sum
            )
            state_count += len(states)
        if state_count < 1:
            raise ValueError("Cannot fit a GCRL normalizer on an empty dataset")
        state_mean = state_sum / state_count
        state_var = np.maximum(state_square_sum / state_count - np.square(state_mean), 0.0)
        state_std = np.sqrt(state_var)
        state_std = np.where(state_std < 1e-6, 1.0, state_std)
        return cls(
            observation_mean=np.asarray(state_mean, dtype=np.float32),
            observation_std=np.asarray(state_std, dtype=np.float32),
        )

    @classmethod
    def identity(cls, observation_dim: int) -> "GCRLNormalizer":
        return cls(
            observation_mean=np.zeros(observation_dim, dtype=np.float32),
            observation_std=np.ones(observation_dim, dtype=np.float32),
        )

    def normalize_states(self, values) -> np.ndarray:
        return (
            np.asarray(values, dtype=np.float32) - self.observation_mean
        ) / self.observation_std

    def normalize_goals(self, values) -> np.ndarray:
        return self.normalize_states(values)

    def normalize_batch(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        observation_keys = {
            "observations",
            "next_observations",
            "value_goals",
            "actor_goals",
            "low_actor_goals",
            "high_actor_goals",
            "high_actor_targets",
        }
        normalized = {}
        for key, value in batch.items():
            if key in observation_keys:
                normalized[key] = self.normalize_states(value)
            else:
                normalized[key] = np.asarray(value)
        return normalized

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "gcrl_goal_semantics": GCRL_GOAL_SEMANTICS,
            "observation_mean": self.observation_mean.tolist(),
            "observation_std": self.observation_std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GCRLNormalizer":
        validate_gcrl_goal_semantics(payload, source="normalizer")
        if payload.get("version") != 2:
            raise ValueError(f"Unsupported GCRL normalizer version: {payload.get('version')!r}")
        return cls(
            observation_mean=np.asarray(payload["observation_mean"], dtype=np.float32),
            observation_std=np.asarray(payload["observation_std"], dtype=np.float32),
        )


def validate_gcrl_goal_semantics(payload: dict, *, source: str) -> None:
    actual = payload.get("gcrl_goal_semantics")
    if actual != GCRL_GOAL_SEMANTICS:
        description = "missing" if actual is None else repr(actual)
        raise ValueError(
            f"{source} uses unsupported GCRL goal semantics {description}; "
            f"required {GCRL_GOAL_SEMANTICS!r}. Legacy compact_xy artifacts cannot be loaded."
        )


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
        state_dims = {
            arrays.states.shape[-1]
            + (0 if arrays.map_insert_index is None else len(arrays.map_features))
            + (0 if arrays.dynamic_map_insert_index is None else len(arrays.map_features))
            for arrays in self.variants.values()
        }
        action_dims = {arrays.actions.shape[-1] for arrays in self.variants.values()}
        if len(state_dims) != 1 or len(action_dims) != 1:
            raise ValueError("All GCRL variants must share state and action dimensions")
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
        actions = []
        anchor_indices = []
        final_indices = []
        map_features = None
        map_insert_index = None
        dynamic_map_insert_index = None
        dynamic_position_indices = []
        dynamic_goal_indices = []
        uses_dynamic_map = None
        uses_static_map = None
        state_offset = 0
        for episode in episodes:
            episode_uses_static_map = episode.map_insert_index is not None
            episode_uses_dynamic_map = episode.dynamic_map_insert_index is not None
            if (episode.map_features is None) == (
                episode_uses_static_map or episode_uses_dynamic_map
            ):
                raise ValueError("GCRL episode map metadata is incomplete")
            if uses_static_map is None:
                uses_static_map = episode_uses_static_map
            elif uses_static_map != episode_uses_static_map:
                raise ValueError(
                    "All GCRL episodes for one variant must agree on static-map usage"
                )
            if episode.map_features is not None:
                candidate = np.asarray(episode.map_features, dtype=np.float32)
                if candidate.ndim != 1:
                    raise ValueError("GCRL episode map features must be one-dimensional")
                if map_features is None:
                    map_features = candidate
                    map_insert_index = episode.map_insert_index
                elif (
                    map_insert_index != episode.map_insert_index
                    or not np.array_equal(map_features, candidate)
                ):
                    raise ValueError(
                        "All GCRL episodes for one variant must share one map feature vector"
                    )
            elif map_features is not None:
                raise ValueError(
                    "All GCRL episodes for one variant must agree on map feature usage"
                )
            if episode_uses_dynamic_map != (
                episode.dynamic_position_indices is not None
                and episode.dynamic_goal_indices is not None
            ):
                raise ValueError("GCRL episode dynamic-map metadata is incomplete")
            if uses_dynamic_map is None:
                uses_dynamic_map = episode_uses_dynamic_map
                dynamic_map_insert_index = episode.dynamic_map_insert_index
            elif (
                uses_dynamic_map != episode_uses_dynamic_map
                or dynamic_map_insert_index != episode.dynamic_map_insert_index
            ):
                raise ValueError(
                    "All GCRL episodes for one variant must agree on dynamic-map usage"
                )
            if episode_uses_dynamic_map:
                positions = np.asarray(episode.dynamic_position_indices, dtype=np.int64)
                goals = np.asarray(episode.dynamic_goal_indices, dtype=np.int64)
                if positions.shape != (len(episode.states),) or goals.shape != positions.shape:
                    raise ValueError(
                        "GCRL episode dynamic-map indices must match its T+1 states"
                    )
                if episode.map_features is None or (
                    np.any(positions < 0)
                    or np.any(goals < 0)
                    or np.any(positions >= len(episode.map_features))
                    or np.any(goals >= len(episode.map_features))
                ):
                    raise ValueError("GCRL episode dynamic-map indices are out of bounds")
                dynamic_position_indices.append(positions)
                dynamic_goal_indices.append(goals)
            step_count = episode.transition_count
            states.append(episode.states)
            actions.append(episode.actions)
            anchor_indices.append(state_offset + np.arange(step_count, dtype=np.int64))
            final_indices.append(
                np.full(step_count, state_offset + step_count, dtype=np.int64)
            )
            state_offset += step_count + 1
        return _VariantArrays(
            states=np.concatenate(states, axis=0),
            actions=np.concatenate(actions, axis=0),
            anchor_state_indices=np.concatenate(anchor_indices, axis=0),
            final_state_indices=np.concatenate(final_indices, axis=0),
            map_features=map_features,
            map_insert_index=map_insert_index,
            dynamic_map_insert_index=dynamic_map_insert_index,
            dynamic_position_indices=(
                np.concatenate(dynamic_position_indices, axis=0)
                if uses_dynamic_map
                else None
            ),
            dynamic_goal_indices=(
                np.concatenate(dynamic_goal_indices, axis=0)
                if uses_dynamic_map
                else None
            ),
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
        arrays = next(iter(self.variants.values()))
        return int(
            arrays.states.shape[-1]
            + (0 if arrays.map_insert_index is None else arrays.map_features.shape[-1])
            + (
                0
                if arrays.dynamic_map_insert_index is None
                else arrays.map_features.shape[-1]
            )
        )

    @property
    def goal_dim(self) -> int:
        return self.state_dim

    @property
    def action_dim(self) -> int:
        return int(next(iter(self.variants.values())).actions.shape[-1])

    @staticmethod
    def _restore_observation_features(
        arrays: _VariantArrays,
        state_indices: np.ndarray,
    ) -> np.ndarray:
        """Restore static/dynamic map slots only for the requested state rows."""
        state_indices = np.asarray(state_indices, dtype=np.int64)
        values = np.asarray(arrays.states[state_indices], dtype=np.float32)
        if arrays.map_features is None:
            return values
        map_dim = arrays.map_features.shape[-1]
        components: list[tuple[int, np.ndarray]] = []
        if arrays.map_insert_index is not None:
            components.append(
                (
                    arrays.map_insert_index,
                    np.broadcast_to(
                        arrays.map_features,
                        values.shape[:-1] + arrays.map_features.shape,
                    ),
                )
            )
        if arrays.dynamic_map_insert_index is not None:
            if (
                arrays.dynamic_position_indices is None
                or arrays.dynamic_goal_indices is None
            ):
                raise ValueError("GCRL variant dynamic-map metadata is incomplete")
            dynamic = np.array(
                np.broadcast_to(
                    arrays.map_features,
                    values.shape[:-1] + arrays.map_features.shape,
                ),
                copy=True,
            )
            rows = dynamic.reshape((-1, map_dim))
            positions = arrays.dynamic_position_indices[state_indices].reshape(-1)
            goals = arrays.dynamic_goal_indices[state_indices].reshape(-1)
            row_indices = np.arange(len(rows), dtype=np.int64)
            distinct = positions != goals
            rows[row_indices[distinct], positions[distinct]] = 2.0
            rows[row_indices[distinct], goals[distinct]] = 3.0
            rows[row_indices[~distinct], positions[~distinct]] = 4.0
            components.append((arrays.dynamic_map_insert_index, dynamic))
        if not components:
            return values
        full_dim = values.shape[-1] + map_dim * len(components)
        restored = np.empty(values.shape[:-1] + (full_dim,), dtype=np.float32)
        compact_mask = np.ones(full_dim, dtype=bool)
        for insert_index, component in components:
            if insert_index < 0 or insert_index + map_dim > full_dim:
                raise ValueError("GCRL variant map insertion index is invalid")
            compact_mask[insert_index : insert_index + map_dim] = False
            restored[..., insert_index : insert_index + map_dim] = component
        if np.count_nonzero(compact_mask) != values.shape[-1]:
            raise ValueError("GCRL map component slots overlap")
        restored[..., compact_mask] = values
        return restored

    @classmethod
    def state_statistics(cls, arrays: _VariantArrays) -> tuple[np.ndarray, np.ndarray]:
        """Return exact moments without materializing repeated static/dynamic maps."""
        states = np.asarray(arrays.states, dtype=np.float64)
        state_sum = states.sum(axis=0)
        state_square_sum = np.square(states).sum(axis=0)
        if arrays.map_features is None:
            return state_sum, state_square_sum
        count = len(states)
        map_features = np.asarray(arrays.map_features, dtype=np.float64)
        map_dim = len(map_features)
        components: list[tuple[int, np.ndarray, np.ndarray]] = []
        if arrays.map_insert_index is not None:
            components.append(
                (
                    arrays.map_insert_index,
                    map_features * count,
                    np.square(map_features) * count,
                )
            )
        if arrays.dynamic_map_insert_index is not None:
            if (
                arrays.dynamic_position_indices is None
                or arrays.dynamic_goal_indices is None
            ):
                raise ValueError("GCRL variant dynamic-map metadata is incomplete")
            positions = arrays.dynamic_position_indices
            goals = arrays.dynamic_goal_indices
            same = positions == goals
            position_only = np.bincount(
                positions[~same], minlength=map_dim
            ).astype(np.float64)
            goal_only = np.bincount(goals[~same], minlength=map_dim).astype(np.float64)
            success = np.bincount(positions[same], minlength=map_dim).astype(np.float64)
            dynamic_sum = (
                map_features * count
                + position_only * (2.0 - map_features)
                + goal_only * (3.0 - map_features)
                + success * (4.0 - map_features)
            )
            dynamic_square_sum = (
                np.square(map_features) * count
                + position_only * (4.0 - np.square(map_features))
                + goal_only * (9.0 - np.square(map_features))
                + success * (16.0 - np.square(map_features))
            )
            components.append(
                (
                    arrays.dynamic_map_insert_index,
                    dynamic_sum,
                    dynamic_square_sum,
                )
            )
        if not components:
            return state_sum, state_square_sum
        full_dim = len(state_sum) + map_dim * len(components)
        full_sum = np.empty(full_dim, dtype=np.float64)
        full_square_sum = np.empty(full_dim, dtype=np.float64)
        compact_mask = np.ones(full_dim, dtype=bool)
        for insert_index, component_sum, component_square_sum in components:
            compact_mask[insert_index : insert_index + map_dim] = False
            full_sum[insert_index : insert_index + map_dim] = component_sum
            full_square_sum[insert_index : insert_index + map_dim] = component_square_sum
        if np.count_nonzero(compact_mask) != len(state_sum):
            raise ValueError("GCRL map component slots overlap")
        full_sum[compact_mask] = state_sum
        full_square_sum[compact_mask] = state_square_sum
        return full_sum, full_square_sum

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
            "observations": self._restore_observation_features(arrays, anchors),
            "next_observations": self._restore_observation_features(arrays, anchors + 1),
            "actions": arrays.actions[transition_indices],
            "value_goals": self._restore_observation_features(
                arrays, value_goal_indices
            ),
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
            batch["actor_goals"] = self._restore_observation_features(
                arrays, actor_goal_indices
            )
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
                    "low_actor_goals": self._restore_observation_features(
                        arrays, low_goal_indices
                    ),
                    "high_actor_goals": self._restore_observation_features(
                        arrays, high_goal_indices
                    ),
                    "high_actor_targets": self._restore_observation_features(
                        arrays, high_target_indices
                    ),
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
    states = vectorize_gcrl_state_observation(
        observations,
        env_family,
        observation_config=observation_config,
        variant=variant,
    )
    map_features = None
    map_insert_index = None
    dynamic_map_insert_index = None
    dynamic_position_indices = None
    dynamic_goal_indices = None
    include_map = bool(observation_config.get("include_map", False))
    include_dynamic_map = bool(
        observation_config.get("include_dynamic_map", False)
    )
    if include_map or include_dynamic_map:
        base_dim = GOAL_CONDITIONED_STATE_DIMS[env_family]
        rows, cols = family_map_shape(env_family)
        map_dim = rows * cols
        removal_ranges = []
        if include_map:
            map_insert_index = base_dim
            map_features = np.asarray(
                states[0, map_insert_index : map_insert_index + map_dim],
                dtype=np.float32,
            ).copy()
            repeated_map = states[:, map_insert_index : map_insert_index + map_dim]
            if not np.all(repeated_map == map_features[None, :]):
                raise ValueError(
                    f"GCRL map features unexpectedly vary within variant={variant!r}"
                )
            removal_ranges.append((map_insert_index, map_insert_index + map_dim))
        if include_dynamic_map:
            dynamic_map_insert_index = base_dim + (
                map_dim if include_map else 0
            )
            dynamic_maps = states[
                :, dynamic_map_insert_index : dynamic_map_insert_index + map_dim
            ]
            if map_features is None:
                map_features = np.asarray(dynamic_maps[0], dtype=np.float32).copy()
                map_features[np.isin(map_features, [2.0, 3.0, 4.0])] = 0.0
            dynamic_position_indices, dynamic_goal_indices = gcrl_dynamic_map_indices(
                observations,
                env_family,
                observation_config=observation_config,
                variant=variant,
            )
            expected_dynamic = np.broadcast_to(map_features, dynamic_maps.shape).copy()
            row_indices = np.arange(len(expected_dynamic), dtype=np.int64)
            distinct = dynamic_position_indices != dynamic_goal_indices
            expected_dynamic[
                row_indices[distinct], dynamic_position_indices[distinct]
            ] = 2.0
            expected_dynamic[row_indices[distinct], dynamic_goal_indices[distinct]] = 3.0
            expected_dynamic[
                row_indices[~distinct], dynamic_position_indices[~distinct]
            ] = 4.0
            if not np.array_equal(dynamic_maps, expected_dynamic):
                raise ValueError(
                    f"GCRL dynamic-map reconstruction mismatch for variant={variant!r}"
                )
            removal_ranges.append(
                (dynamic_map_insert_index, dynamic_map_insert_index + map_dim)
            )
        keep = np.ones(states.shape[-1], dtype=bool)
        for start, stop in removal_ranges:
            keep[start:stop] = False
        states = states[:, keep]
    actions = np.asarray(episode_field(episode, "actions"), dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    expected_action_dim = 2 if env_family == "pointmaze" else 8
    if actions.ndim != 2 or actions.shape[1] != expected_action_dim:
        raise ValueError(
            f"Unexpected {env_family} action shape for variant={variant!r}: {actions.shape}"
        )
    if len(states) != len(actions) + 1:
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
        actions=np.clip(actions, -1.0, 1.0).astype(np.float32),
        map_features=map_features,
        map_insert_index=map_insert_index,
        dynamic_map_insert_index=dynamic_map_insert_index,
        dynamic_position_indices=(
            np.asarray(dynamic_position_indices, dtype=np.int64)
            if dynamic_position_indices is not None
            else None
        ),
        dynamic_goal_indices=(
            np.asarray(dynamic_goal_indices, dtype=np.int64)
            if dynamic_goal_indices is not None
            else None
        ),
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
        "gcrl_goal_semantics": GCRL_GOAL_SEMANTICS,
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
