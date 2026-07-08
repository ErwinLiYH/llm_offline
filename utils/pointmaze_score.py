"""PointMaze official-style scoring: reference scores and reference-file IO.

Score environment construction (official eval maps, env specs, fingerprints)
lives in `crossmaze.score` and is re-exported here for compatibility.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from crossmaze.score import (  # noqa: F401 re-exported for compatibility
    OFFICIAL_POINTMAZE_EVAL_MAPS,
    REMOTE_POINTMAZE_HORIZONS,
    PointMazeScoreEnvSpec,
    build_local_pointmaze_score_env_spec,
    build_local_score_maze_map,
    build_pointmaze_score_env_spec,
    build_remote_pointmaze_score_env_spec,
    fingerprint_score_env_spec,
    make_pointmaze_score_env,
)


REMOTE_POINTMAZE_REFERENCE_SCORES = {
    "open": {
        "ref_min_score": 7.199999809265137,
        "ref_max_score": 229.86000061035156,
    },
    "open-dense": {
        "ref_min_score": 70.7329330444336,
        "ref_max_score": 229.4267120361328,
    },
    "umaze": {
        "ref_min_score": 13.489999771118164,
        "ref_max_score": 218.6999969482422,
    },
    "umaze-dense": {
        "ref_min_score": 59.25226974487305,
        "ref_max_score": 223.9688720703125,
    },
    "medium": {
        "ref_min_score": 17.65999984741211,
        "ref_max_score": 361.04998779296875,
    },
    "medium-dense": {
        "ref_min_score": 49.2408447265625,
        "ref_max_score": 368.8089599609375,
    },
    "large": {
        "ref_min_score": 3.549999952316284,
        "ref_max_score": 462.260009765625,
    },
    "large-dense": {
        "ref_min_score": 27.165931701660156,
        "ref_max_score": 481.5344543457031,
    },
}


def normalize_score(mean_return: float, ref_min_score: float, ref_max_score: float) -> float:
    denom = float(ref_max_score) - float(ref_min_score)
    if denom == 0:
        raise ValueError("Cannot normalize score when ref_min_score == ref_max_score")
    return 100.0 * (float(mean_return) - float(ref_min_score)) / denom


def normalize_score_std(std_return: float, ref_min_score: float, ref_max_score: float) -> float:
    denom = float(ref_max_score) - float(ref_min_score)
    if denom == 0:
        raise ValueError("Cannot normalize score std when ref_min_score == ref_max_score")
    return 100.0 * float(std_return) / abs(denom)


def get_remote_pointmaze_reference(variant: str) -> dict:
    if variant not in REMOTE_POINTMAZE_REFERENCE_SCORES:
        raise ValueError(f"No Minari/D4RL PointMaze reference scores for variant {variant!r}")
    ref = dict(REMOTE_POINTMAZE_REFERENCE_SCORES[variant])
    ref["reference_source"] = "minari_d4rl_metadata"
    ref["num_episodes_average_score"] = 100
    return ref


def local_reference_path(config: dict, variant: str) -> Path:
    root = Path(config.get("local_reference_root", "local_references/pointmaze")).expanduser()
    if not root.is_absolute():
        root = Path(os.getcwd()) / root
    return root / f"{variant}.json"


def load_and_validate_local_reference(
    *,
    config: dict,
    variant: str,
    score_env_spec: PointMazeScoreEnvSpec,
) -> tuple[dict, Path]:
    path = local_reference_path(config, variant)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing local reference for {variant!r}: {path}. "
            "Run score.py --mode reference for this variant first."
        )
    with open(path, "r", encoding="utf-8") as f:
        reference = json.load(f)
    actual_fingerprint = reference.get("env_fingerprint")
    expected_fingerprint = score_env_spec.env_fingerprint
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Local reference fingerprint mismatch for {variant!r}: "
            f"reference={actual_fingerprint}, current={expected_fingerprint}. "
            "Regenerate the reference with the current score.yaml local_eval_maps settings."
        )
    for key in ("ref_min_score", "ref_max_score"):
        if key not in reference:
            raise ValueError(f"Local reference {path} is missing {key}")
    return reference, path
