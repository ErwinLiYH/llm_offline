from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import Any

import yaml

DELETE_KEYS_FIELD = "config_delete_keys"


def resolve_dataset_cache_dir(config: Mapping[str, Any]) -> str | None:
    """Return the effective tokenized-dataset cache directory.

    ``dataset_cache_v2_root`` and ``dataset_cache_v2_dir`` are an optional pair.
    When both are non-empty, they take precedence over the legacy
    ``dataset_cache_dir`` setting.  A partial V2 pair deliberately falls back to
    the legacy setting so layered configs can opt into V2 incrementally.
    """
    v2_root = config.get("dataset_cache_v2_root")
    v2_dir = config.get("dataset_cache_v2_dir")
    if not v2_root or not v2_dir:
        return config.get("dataset_cache_dir")

    if not isinstance(v2_root, str):
        raise ValueError(
            "dataset_cache_v2_root must be a non-empty path string when "
            "dataset_cache_v2_dir is configured"
        )
    if not isinstance(v2_dir, str):
        raise ValueError(
            "dataset_cache_v2_dir must be a non-empty relative path string when "
            "dataset_cache_v2_root is configured"
        )
    if os.path.isabs(v2_dir):
        raise ValueError(
            "dataset_cache_v2_dir must be a relative path so it can be joined "
            "under dataset_cache_v2_root"
        )
    return os.path.join(v2_root, v2_dir)


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
