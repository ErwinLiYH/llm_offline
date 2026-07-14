"""Reward-type contracts shared by CrossMaze environments and data tools."""

from __future__ import annotations

import re
from pathlib import Path


REWARD_TYPES = ("sparse", "dense")


def normalize_reward_type(
    reward_type: str | None,
    *,
    default: str | None = None,
) -> str:
    """Return a validated canonical CrossMaze reward type."""
    value = default if reward_type is None else reward_type
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "reward_type must be one of "
            f"{list(REWARD_TYPES)}, got {value!r}"
        )
    normalized = value.strip().lower()
    if normalized not in REWARD_TYPES:
        raise ValueError(
            "reward_type must be one of "
            f"{list(REWARD_TYPES)}, got {value!r}"
        )
    return normalized


def resolve_reward_type(config: dict | None, *, default: str) -> str:
    """Resolve the single top-level reward setting used by CrossMaze."""
    config = dict(config or {})
    env_kwargs = config.get("env_kwargs") or {}
    if not isinstance(env_kwargs, dict):
        raise ValueError("env_kwargs must be a mapping when provided")
    if "reward_type" in env_kwargs:
        raise ValueError(
            "env_kwargs.reward_type is not supported; configure top-level "
            "reward_type instead"
        )
    return normalize_reward_type(config.get("reward_type"), default=default)


def reward_typed_dataset_path(
    dataset_path: str | Path,
    *,
    reward_type: str,
    default_reward_type: str,
) -> Path:
    """Keep the default path stable and suffix alternate reward datasets."""
    path = Path(dataset_path).expanduser()
    resolved = normalize_reward_type(reward_type)
    default = normalize_reward_type(default_reward_type)
    if resolved == default:
        return path

    match = re.fullmatch(r"(.+)-v(\d+)", path.name)
    if match is None:
        return path.with_name(f"{path.name}-{resolved}")
    base_name, version = match.groups()
    return path.with_name(f"{base_name}-{resolved}-v{version}")
