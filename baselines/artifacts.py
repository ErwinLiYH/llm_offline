from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _safe_experiment_id(value: str) -> str:
    normalized = value.strip()
    if not normalized or not re.fullmatch(r"[A-Za-z0-9._+-]+", normalized):
        raise ValueError(
            "experiment_id may only contain letters, numbers, '.', '_', '+', and '-'"
        )
    return normalized


def create_run_dir(config: dict, selection_tag: str) -> tuple[str, Path]:
    experiment_id = config["experiment_id"]
    if experiment_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_id = (
            f"{timestamp}-{config['algorithm']}-{config['env_family']}-{selection_tag}"
        )
    experiment_id = _safe_experiment_id(experiment_id)
    output_root = Path(config["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    run_dir = output_root / experiment_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Baseline run directory already exists: {run_dir}. "
            "Choose a different experiment_id."
        ) from exc
    (run_dir / "checkpoints").mkdir()
    return experiment_id, run_dir
