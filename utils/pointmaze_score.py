"""PointMaze official-style score environment and reference helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym

from data.pointmaze.variants import POINTMAZE_VARIANTS, get_pointmaze_variant_type


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


OFFICIAL_POINTMAZE_EVAL_MAPS = {
    "open": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, "g", 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    "umaze": [
        [1, 1, 1, 1, 1],
        [1, "g", 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ],
    "medium": [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, "g", 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "large": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, "g", 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}


REMOTE_POINTMAZE_HORIZONS = {
    "open": 300,
    "open-dense": 300,
    "umaze": 300,
    "umaze-dense": 300,
    "medium": 600,
    "medium-dense": 600,
    "large": 800,
    "large-dense": 800,
}


@dataclass(frozen=True)
class PointMazeScoreEnvSpec:
    variant: str
    variant_type: str
    env_id: str
    env_kwargs: dict
    max_episode_steps: int
    reward_type: str
    goal_cell: list[int]
    env_fingerprint: str

    def to_result_dict(self) -> dict:
        return asdict(self)


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


def _remote_eval_map_key(variant: str) -> str:
    if variant.endswith("-dense"):
        return variant[: -len("-dense")]
    return variant


def _find_goal_cell(maze_map: list[list[object]]) -> list[int]:
    goal_cells = []
    for row_idx, row in enumerate(maze_map):
        for col_idx, value in enumerate(row):
            if value == "g":
                goal_cells.append([row_idx, col_idx])
    if len(goal_cells) != 1:
        raise ValueError(f"Score maze must contain exactly one goal cell, found {goal_cells}")
    return goal_cells[0]


def _fingerprint_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def fingerprint_score_env_spec(
    *,
    env_id: str,
    maze_map: list[list[object]],
    reward_type: str,
    continuing_task: bool,
    reset_target: bool,
    max_episode_steps: int,
    goal_cell: list[int],
) -> str:
    return _fingerprint_payload(
        {
            "env_id": env_id,
            "maze_map": maze_map,
            "reward_type": reward_type,
            "continuing_task": bool(continuing_task),
            "reset_target": bool(reset_target),
            "max_episode_steps": int(max_episode_steps),
            "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
        }
    )


def build_remote_pointmaze_score_env_spec(variant: str) -> PointMazeScoreEnvSpec:
    if variant not in POINTMAZE_VARIANTS:
        raise ValueError(f"Unknown PointMaze variant: {variant!r}")
    meta = POINTMAZE_VARIANTS[variant]
    if get_pointmaze_variant_type(meta) != "remote":
        raise ValueError(f"Variant {variant!r} is not a remote PointMaze variant")

    map_key = _remote_eval_map_key(variant)
    maze_map = copy.deepcopy(OFFICIAL_POINTMAZE_EVAL_MAPS[map_key])
    goal_cell = _find_goal_cell(maze_map)
    reward_type = str(meta["prompt_vars"].get("reward_type", "sparse"))
    env_kwargs = {
        "maze_map": maze_map,
        "reward_type": reward_type,
        "continuing_task": True,
        "reset_target": False,
    }
    max_episode_steps = int(REMOTE_POINTMAZE_HORIZONS[variant])
    return PointMazeScoreEnvSpec(
        variant=variant,
        variant_type="remote",
        env_id=meta["env_id"],
        env_kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
        reward_type=reward_type,
        goal_cell=goal_cell,
        env_fingerprint=fingerprint_score_env_spec(
            env_id=meta["env_id"],
            maze_map=maze_map,
            reward_type=reward_type,
            continuing_task=True,
            reset_target=False,
            max_episode_steps=max_episode_steps,
            goal_cell=goal_cell,
        ),
    )


def _normalize_goal_cell(raw_goal_cell, *, variant: str) -> list[int]:
    if (
        not isinstance(raw_goal_cell, (list, tuple))
        or len(raw_goal_cell) != 2
        or not all(isinstance(value, int) for value in raw_goal_cell)
    ):
        raise ValueError(
            f"local_eval_maps.{variant}.goal_cell must be a two-int [row, col] list"
        )
    return [int(raw_goal_cell[0]), int(raw_goal_cell[1])]


def build_local_score_maze_map(
    *,
    variant: str,
    base_maze_map: list[list[int]],
    goal_cell: list[int],
) -> list[list[object]]:
    if not base_maze_map or not all(isinstance(row, list) and row for row in base_maze_map):
        raise ValueError(f"Variant {variant!r} has an invalid maze_map")
    row_count = len(base_maze_map)
    col_count = len(base_maze_map[0])
    if any(len(row) != col_count for row in base_maze_map):
        raise ValueError(f"Variant {variant!r} has a non-rectangular maze_map")

    row_idx, col_idx = goal_cell
    if row_idx < 0 or row_idx >= row_count or col_idx < 0 or col_idx >= col_count:
        raise ValueError(
            f"local_eval_maps.{variant}.goal_cell={goal_cell} is outside maze shape "
            f"{row_count}x{col_count}"
        )
    if base_maze_map[row_idx][col_idx] != 0:
        raise ValueError(
            f"local_eval_maps.{variant}.goal_cell={goal_cell} must point to a free cell"
        )

    maze_map = copy.deepcopy(base_maze_map)
    maze_map[row_idx][col_idx] = "g"
    return maze_map


def build_local_pointmaze_score_env_spec(variant: str, config: dict) -> PointMazeScoreEnvSpec:
    if variant not in POINTMAZE_VARIANTS:
        raise ValueError(f"Unknown PointMaze variant: {variant!r}")
    meta = POINTMAZE_VARIANTS[variant]
    if get_pointmaze_variant_type(meta) != "local":
        raise ValueError(f"Variant {variant!r} is not a local PointMaze variant")

    local_eval_maps = config.get("local_eval_maps") or {}
    variant_config = local_eval_maps.get(variant)
    if not isinstance(variant_config, dict):
        raise ValueError(
            f"Local scoring for {variant!r} requires local_eval_maps.{variant}.goal_cell"
        )
    goal_cell = _normalize_goal_cell(variant_config.get("goal_cell"), variant=variant)
    base_maze_map = meta["prompt_vars"]["maze_map"]
    maze_map = build_local_score_maze_map(
        variant=variant,
        base_maze_map=base_maze_map,
        goal_cell=goal_cell,
    )

    env_paras = dict(meta["env_paras"])
    env_id = env_paras.get("id", "PointMaze_UMaze-v3")
    reward_type = str(env_paras.get("reward_type", meta["prompt_vars"].get("reward_type", "sparse")))
    max_episode_steps = int(
        variant_config.get(
            "max_episode_steps",
            env_paras.get("max_episode_steps", meta["prompt_vars"].get("max_episode_steps", 300)),
        )
    )
    env_kwargs = {
        "maze_map": maze_map,
        "reward_type": reward_type,
        "continuing_task": True,
        "reset_target": False,
    }

    return PointMazeScoreEnvSpec(
        variant=variant,
        variant_type="local",
        env_id=env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
        reward_type=reward_type,
        goal_cell=goal_cell,
        env_fingerprint=fingerprint_score_env_spec(
            env_id=env_id,
            maze_map=maze_map,
            reward_type=reward_type,
            continuing_task=True,
            reset_target=False,
            max_episode_steps=max_episode_steps,
            goal_cell=goal_cell,
        ),
    )


def build_pointmaze_score_env_spec(variant: str, config: dict) -> PointMazeScoreEnvSpec:
    meta = POINTMAZE_VARIANTS[variant]
    if get_pointmaze_variant_type(meta) == "remote":
        return build_remote_pointmaze_score_env_spec(variant)
    return build_local_pointmaze_score_env_spec(variant, config)


def make_pointmaze_score_env(
    score_env_spec: PointMazeScoreEnvSpec,
    *,
    render_mode: str | None = None,
):
    kwargs = copy.deepcopy(score_env_spec.env_kwargs)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if score_env_spec.variant_type == "local":
        return gym.make(
            score_env_spec.env_id,
            max_episode_steps=score_env_spec.max_episode_steps,
            **kwargs,
        )
    return gym.make(score_env_spec.env_id, **kwargs)


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
