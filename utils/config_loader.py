from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import yaml

DELETE_KEYS_FIELD = "config_delete_keys"


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


def _delete_config_key(config: dict[str, Any], key_path: str) -> None:
    current: Any = config
    parts = key_path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


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
        delete_keys = payload.pop(DELETE_KEYS_FIELD, [])
        if delete_keys is None:
            delete_keys = []
        if not isinstance(delete_keys, list) or any(
            not isinstance(key, str) for key in delete_keys
        ):
            raise ValueError(f"{DELETE_KEYS_FIELD} must be a list of strings: {path}")
        for key_path in delete_keys:
            _delete_config_key(merged, key_path)
        merged = deep_merge_configs(merged, payload)
    return merged
