from __future__ import annotations

import math
from collections.abc import Mapping


WALL_SENSING_VERSION_KEY = "wall_sensing_version"
MAP_SENSING_BOUNDARY_RISK_THRESHOLD_KEY = "map_sensing_boundary_risk_threshold"
SENSING_CONFIG_KEYS = (
    WALL_SENSING_VERSION_KEY,
    MAP_SENSING_BOUNDARY_RISK_THRESHOLD_KEY,
)

DEFAULT_WALL_SENSING_VERSION = "v3"
DEFAULT_MAP_SENSING_BOUNDARY_RISK_THRESHOLD = 0.10
VALID_WALL_SENSING_VERSIONS = frozenset({"v1", "v2", "v3", "v4", "v5"})


def resolve_wall_sensing_version(value=None) -> str:
    if value is None:
        return DEFAULT_WALL_SENSING_VERSION
    version = str(value).strip().lower()
    if version not in VALID_WALL_SENSING_VERSIONS:
        allowed = ", ".join(sorted(VALID_WALL_SENSING_VERSIONS))
        raise ValueError(
            f"{WALL_SENSING_VERSION_KEY} must be one of {allowed}, got {value!r}"
        )
    return version


def resolve_map_sensing_boundary_risk_threshold(value=None) -> float:
    if value is None:
        return float(DEFAULT_MAP_SENSING_BOUNDARY_RISK_THRESHOLD)
    if isinstance(value, bool):
        raise ValueError(
            f"{MAP_SENSING_BOUNDARY_RISK_THRESHOLD_KEY} must be a non-negative float, "
            f"got {value!r}"
        )
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(
            f"{MAP_SENSING_BOUNDARY_RISK_THRESHOLD_KEY} must be a finite value >= 0, "
            f"got {threshold}"
        )
    return threshold


def resolve_sensing_config(config: Mapping | None) -> dict:
    config = config or {}
    return {
        WALL_SENSING_VERSION_KEY: resolve_wall_sensing_version(
            config.get(WALL_SENSING_VERSION_KEY)
        ),
        MAP_SENSING_BOUNDARY_RISK_THRESHOLD_KEY: (
            resolve_map_sensing_boundary_risk_threshold(
                config.get(MAP_SENSING_BOUNDARY_RISK_THRESHOLD_KEY)
            )
        ),
    }


def normalize_sensing_config(config: dict) -> dict:
    resolved = resolve_sensing_config(config)
    config.update(resolved)
    return resolved


def has_explicit_sensing_config(config: Mapping | None) -> bool:
    if not config:
        return False
    return any(key in config for key in SENSING_CONFIG_KEYS)


def apply_sensing_config_to_prompt_vars(prompt_vars: Mapping, config: Mapping | None) -> dict:
    resolved = dict(prompt_vars)
    resolved.update(resolve_sensing_config(config))
    return resolved


def _resolve_explicit_sensing_key(key: str, value):
    if key == WALL_SENSING_VERSION_KEY:
        return resolve_wall_sensing_version(value)
    if key == MAP_SENSING_BOUNDARY_RISK_THRESHOLD_KEY:
        return resolve_map_sensing_boundary_risk_threshold(value)
    raise KeyError(key)


def apply_checkpoint_sensing_config(config: dict, checkpoint_config: Mapping | None) -> dict:
    merged = dict(config)
    if has_explicit_sensing_config(checkpoint_config):
        checkpoint_sensing = resolve_sensing_config(checkpoint_config)
        conflicts = []
        for key in SENSING_CONFIG_KEYS:
            if key not in config:
                continue
            requested_value = _resolve_explicit_sensing_key(key, config.get(key))
            checkpoint_value = checkpoint_sensing[key]
            if requested_value != checkpoint_value:
                conflicts.append((key, requested_value, checkpoint_value))
        if conflicts:
            lines = [
                "Eval/score sensing config must match checkpoint config.yaml when the checkpoint records it."
            ]
            for key, requested_value, checkpoint_value in conflicts:
                lines.append(
                    f"  {key}: requested={requested_value!r}, checkpoint={checkpoint_value!r}"
                )
            raise ValueError("\n".join(lines))
        merged.update(checkpoint_sensing)
        return merged

    normalize_sensing_config(merged)
    return merged
