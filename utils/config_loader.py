from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import yaml


def deep_merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two config dictionaries without mutating either input."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge_configs(merged[key], dict(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_merged_config(config_paths: str | list[str] | tuple[str, ...]) -> dict[str, Any]:
    if isinstance(config_paths, str):
        paths = [config_paths]
    else:
        paths = [str(path) for path in config_paths]
    if not paths:
        raise ValueError("At least one config path is required")

    merged: dict[str, Any] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must contain a mapping at top level: {path}")
        merged = deep_merge_configs(merged, payload)
    return merged
