"""Helpers for saving per-experiment runtime configuration snapshots."""

from __future__ import annotations

import os

import yaml


def save_experiment_config_snapshot(
    config: dict,
    *,
    root: str = "exp_configs",
    filename: str = "config.yaml",
) -> str:
    experiment_id = str(config.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("Cannot save experiment config snapshot without experiment_id.")
    if not filename or os.path.basename(filename) != filename:
        raise ValueError(f"filename must be a plain file name, got {filename!r}")

    config_dir = os.path.join(root, experiment_id)
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, filename)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return config_path
