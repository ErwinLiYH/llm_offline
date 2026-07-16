from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import minari
import numpy as np
from minari import MinariDataset

from crossmaze import get_env_facts, list_variants
from crossmaze.reward import normalize_reward_type, reward_typed_dataset_path

from baselines.data.observation import vectorize_observation


@dataclass(frozen=True)
class LoadedVariant:
    variant: str
    source: str
    reward_type: str
    episodes: list[Any]
    dataset_path: str
    warnings: list[str]


@dataclass(frozen=True)
class PreparedDatasets:
    train_buffer: Any
    validation_buffer: Any
    train_episodes: list[Any]
    validation_episodes: list[Any]
    manifest: dict


def _read_hdf5_tree(node):
    if isinstance(node, h5py.Dataset):
        return node[()]
    return {name: _read_hdf5_tree(node[name]) for name in node.keys()}


def _load_hdf5_episodes(data_path: Path) -> list[Any]:
    h5_path = data_path / "main_data.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Local baseline dataset file not found: {h5_path}")
    episodes = []
    with h5py.File(h5_path, "r") as file:
        episode_names = sorted(
            (name for name in file.keys() if name.startswith("episode_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        for name in episode_names:
            group = file[name]
            fields = {
                "observations": _read_hdf5_tree(group["observations"]),
                "actions": group["actions"][()],
            }
            for field in ("rewards", "terminations", "truncations", "infos"):
                if field in group:
                    fields[field] = _read_hdf5_tree(group[field])
            episodes.append(SimpleNamespace(**fields))
    if not episodes:
        raise ValueError(f"No episodes found in local baseline dataset: {h5_path}")
    return episodes


def _load_local_episodes(data_path: Path) -> list[Any]:
    try:
        dataset = MinariDataset(data_path)
    except ValueError as exc:
        if "No data found in data path" not in str(exc):
            raise
        return _load_hdf5_episodes(data_path)
    return list(dataset.iterate_episodes())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_local_dataset_path(
    dataset_path: str | Path,
    *,
    local_dataset_root: str | Path | None,
    reward_type: str,
    default_reward_type: str,
) -> Path:
    default_path = reward_typed_dataset_path(
        dataset_path,
        reward_type=reward_type,
        default_reward_type=default_reward_type,
    )
    if local_dataset_root is None:
        resolved = default_path
    else:
        root_path = Path(local_dataset_root).expanduser()
        resolved = (
            root_path
            if root_path.name == default_path.name or (root_path / "data").is_dir()
            else root_path / default_path.name
        )
    if not resolved.is_absolute():
        resolved = _repo_root() / resolved
    return resolved


def _hdf5_reward_type(data_path: Path) -> str | None:
    h5_path = data_path / "main_data.hdf5"
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as file:
        env_spec = file.attrs.get("env_spec")
    if env_spec is None:
        return None
    if isinstance(env_spec, bytes):
        env_spec = env_spec.decode("utf-8")
    try:
        payload = json.loads(str(env_spec))
    except (TypeError, json.JSONDecodeError):
        return None
    reward_type = (payload.get("kwargs") or {}).get("reward_type")
    return normalize_reward_type(reward_type) if reward_type is not None else None


def _local_dataset_reward_type(dataset_root: Path) -> str | None:
    summary_reward_type = None
    summary_path = dataset_root / "generation_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        reward_type = payload.get("reward_type")
        if reward_type is not None:
            summary_reward_type = normalize_reward_type(reward_type)
    hdf5_reward_type = _hdf5_reward_type(dataset_root / "data")
    if (
        summary_reward_type is not None
        and hdf5_reward_type is not None
        and summary_reward_type != hdf5_reward_type
    ):
        raise ValueError(
            f"Local dataset reward metadata conflicts at {dataset_root}: "
            f"generation_summary={summary_reward_type!r}, hdf5={hdf5_reward_type!r}"
        )
    return summary_reward_type or hdf5_reward_type


def load_variant_episodes(
    env_family: str,
    variant: str,
    *,
    reward_type: str,
    local_dataset_root: str | None,
) -> LoadedVariant:
    facts = get_env_facts(env_family, variant)
    default_reward_type = normalize_reward_type(facts["reward_type"])
    effective_reward_type = normalize_reward_type(reward_type)
    warnings = []
    if facts["kind"] == "remote":
        if effective_reward_type != default_reward_type:
            raise ValueError(
                f"Remote dataset {variant!r} has reward_type={default_reward_type!r}, "
                f"not {effective_reward_type!r}"
            )
        dataset_id = facts["dataset_id"]
        dataset = minari.load_dataset(dataset_id, download=True)
        episodes = list(dataset.iterate_episodes())
        source = f"minari:{dataset_id}"
        dataset_path = dataset_id
    else:
        dataset_root = resolve_local_dataset_path(
            facts["dataset_path"],
            local_dataset_root=local_dataset_root,
            reward_type=effective_reward_type,
            default_reward_type=default_reward_type,
        )
        data_path = dataset_root / "data"
        if not data_path.is_dir():
            raise FileNotFoundError(
                f"Local {env_family} dataset for variant={variant!r} was not found at {data_path}"
            )
        detected_reward_type = _local_dataset_reward_type(dataset_root)
        if detected_reward_type is None:
            if effective_reward_type != default_reward_type:
                raise ValueError(
                    f"Cannot verify reward_type={effective_reward_type!r} for legacy local "
                    f"dataset without reward metadata: {dataset_root}"
                )
            warnings.append(
                f"Legacy local dataset has no reward metadata; assuming {default_reward_type}: "
                f"{dataset_root}"
            )
        elif detected_reward_type != effective_reward_type:
            raise ValueError(
                f"Local dataset reward mismatch at {dataset_root}: "
                f"configured={effective_reward_type!r}, stored={detected_reward_type!r}"
            )
        episodes = _load_local_episodes(data_path)
        source = f"local:{dataset_root}"
        dataset_path = str(dataset_root)
    if not episodes:
        raise ValueError(f"Dataset for variant={variant!r} contains zero episodes")
    return LoadedVariant(
        variant=variant,
        source=source,
        reward_type=effective_reward_type,
        episodes=episodes,
        dataset_path=dataset_path,
        warnings=warnings,
    )


def _episode_field(episode, name: str, default=None):
    if isinstance(episode, Mapping):
        return episode.get(name, default)
    return getattr(episode, name, default)


def _convert_episode(episode, *, env_family: str, variant: str):
    from baselines.data.transitions import MinariTransitionEpisode

    observations = _episode_field(episode, "observations")
    if not isinstance(observations, Mapping):
        raise ValueError(
            f"{env_family} baseline requires dict observations, got {type(observations).__name__}"
        )
    vector_observations = vectorize_observation(dict(observations), env_family)
    actions = np.asarray(_episode_field(episode, "actions"), dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    expected_action_dim = 2 if env_family == "pointmaze" else 8
    if actions.ndim != 2 or actions.shape[1] != expected_action_dim:
        raise ValueError(
            f"Unexpected {env_family} action shape for variant={variant!r}: {actions.shape}"
        )
    if np.any(actions < -1.0001) or np.any(actions > 1.0001):
        raise ValueError(f"Actions for variant={variant!r} fall outside [-1, 1]")
    step_count = actions.shape[0]
    if step_count < 1:
        raise ValueError(f"Episode for variant={variant!r} has no transitions")
    actions = np.clip(actions, -1.0, 1.0)
    rewards_value = _episode_field(episode, "rewards")
    if rewards_value is None:
        raise ValueError(f"Episode for variant={variant!r} is missing rewards")
    rewards = np.asarray(rewards_value, dtype=np.float32).reshape(-1)
    if rewards.shape[0] != step_count:
        raise ValueError("Episode rewards and actions must have the same length")

    terminations = np.asarray(
        _episode_field(episode, "terminations", np.zeros(step_count, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    truncations = np.asarray(
        _episode_field(episode, "truncations", np.zeros(step_count, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if terminations.shape[0] != step_count or truncations.shape[0] != step_count:
        raise ValueError("Episode termination flags and actions must have the same length")
    if np.any(terminations & truncations):
        raise ValueError("An episode step cannot be both terminated and truncated")
    if np.any(terminations[:-1]) or np.any(truncations[:-1]):
        raise ValueError("Termination or truncation may only appear on the final episode step")

    return MinariTransitionEpisode(
        observations=vector_observations,
        actions=actions,
        rewards=rewards,
        terminated=bool(terminations[-1]),
        truncated=bool(truncations[-1]),
        source_variant=variant,
    )


def _variant_sampling_seed(variant: str, sampling_seed: int) -> int:
    digest = hashlib.sha256(f"{variant}:{sampling_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _sampled_target(total_episodes: int, keep: int | None) -> int:
    return total_episodes if keep is None else min(total_episodes, keep)


def prepare_datasets(
    config: dict,
    selected_variants: list[str],
    reward_types: dict[str, str],
) -> PreparedDatasets:
    from baselines.data.transitions import build_replay_buffer

    available = set(list_variants(config["env_family"]))
    unknown_keep_variants = sorted(
        set(config["episode_keep_per_variant"]) - available
    )
    if unknown_keep_variants:
        raise ValueError(
            f"episode_keep_per_variant contains unknown variants: {unknown_keep_variants}"
        )

    loaded = {
        variant: load_variant_episodes(
            config["env_family"],
            variant,
            reward_type=reward_types[variant],
            local_dataset_root=config["local_dataset_root"],
        )
        for variant in selected_variants
    }
    initial_targets = {}
    for variant, result in loaded.items():
        keep = config["episode_keep_per_variant"].get(
            variant, config["episode_keep_num"]
        )
        initial_targets[variant] = _sampled_target(len(result.episodes), keep)

    balance_enabled = config["balance_variant_episode_count"]
    per_variant_keep_enabled = bool(config["episode_keep_per_variant"])
    warnings = [warning for result in loaded.values() for warning in result.warnings]
    if balance_enabled and per_variant_keep_enabled:
        warnings.append(
            "episode_keep_per_variant is configured; balance_variant_episode_count was ignored"
        )
        balance_enabled = False
    if balance_enabled and len(selected_variants) > 1:
        balanced_target = min(initial_targets.values())
    else:
        balanced_target = None

    train_episodes = []
    validation_episodes = []
    manifest_variants = {}
    for variant in selected_variants:
        result = loaded[variant]
        sampled_target = (
            balanced_target if balanced_target is not None else initial_targets[variant]
        )
        if sampled_target < 1:
            raise ValueError(f"No episodes selected for variant={variant!r}")
        rng = np.random.default_rng(
            _variant_sampling_seed(variant, config["sampling_seed"])
        )
        sampled_indices = rng.permutation(len(result.episodes)).tolist()[:sampled_target]
        train_target = math.floor(sampled_target * config["train_data_ratio"])
        if train_target < 1:
            raise ValueError(
                f"train_data_ratio selected zero train episodes for variant={variant!r}"
            )
        train_indices = sorted(sampled_indices[:train_target])
        validation_indices = sorted(sampled_indices[train_target:])
        if not validation_indices:
            raise ValueError(
                f"train_data_ratio selected zero validation episodes for variant={variant!r}"
            )

        converted_train = [
            _convert_episode(result.episodes[index], env_family=config["env_family"], variant=variant)
            for index in train_indices
        ]
        converted_validation = [
            _convert_episode(result.episodes[index], env_family=config["env_family"], variant=variant)
            for index in validation_indices
        ]
        train_episodes.extend(converted_train)
        validation_episodes.extend(converted_validation)
        manifest_variants[variant] = {
            "source": result.source,
            "dataset_path": result.dataset_path,
            "reward_type": result.reward_type,
            "total_episodes": len(result.episodes),
            "initial_sampled_episode_target": initial_targets[variant],
            "sampled_episode_count": sampled_target,
            "train_episode_count": len(converted_train),
            "validation_episode_count": len(converted_validation),
            "train_transition_count": sum(ep.transition_count for ep in converted_train),
            "validation_transition_count": sum(
                ep.transition_count for ep in converted_validation
            ),
            "train_episode_indices": train_indices,
            "validation_episode_indices": validation_indices,
        }

    train_buffer = build_replay_buffer(train_episodes)
    validation_buffer = build_replay_buffer(validation_episodes)
    manifest = {
        "env_family": config["env_family"],
        "observation_schema": (
            "observation+desired_goal"
            if config["env_family"] == "pointmaze"
            else "achieved_goal+observation+desired_goal"
        ),
        "sampling_seed": config["sampling_seed"],
        "train_data_ratio": config["train_data_ratio"],
        "balance_variant_episode_count": bool(balanced_target is not None),
        "balanced_episode_target": balanced_target,
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "train_transition_count": train_buffer.transition_count,
        "validation_transition_count": validation_buffer.transition_count,
        "variants": manifest_variants,
        "warnings": warnings,
    }
    return PreparedDatasets(
        train_buffer=train_buffer,
        validation_buffer=validation_buffer,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        manifest=manifest,
    )
