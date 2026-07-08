"""Repo-side sensing config: canonical helpers re-exported from `crossmaze`.

Only checkpoint-specific merging stays here; everything else lives in
`crossmaze.sensing_config` so the CrossMaze package is self-contained.
"""

from __future__ import annotations

from collections.abc import Mapping

from crossmaze.sensing_config import (  # noqa: F401
    DEFAULT_MAP_SENSING_BOUNDARY_RISK_THRESHOLD,
    DEFAULT_WALL_SENSING_VERSION,
    MAP_SENSING_BOUNDARY_RISK_THRESHOLD_KEY,
    SENSING_CONFIG_KEYS,
    VALID_WALL_SENSING_VERSIONS,
    WALL_SENSING_VERSION_KEY,
    apply_sensing_config_to_prompt_vars,
    has_explicit_sensing_config,
    normalize_sensing_config,
    resolve_map_sensing_boundary_risk_threshold,
    resolve_sensing_config,
    resolve_wall_sensing_version,
)


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
