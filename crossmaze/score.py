"""Official-style PointMaze score environment construction.

Score environments keep the official D4RL evaluation semantics: the official
Farama single-goal eval maps (goal marked as "g" in the map), forced
`continuing_task: true` / `reset_target: false`, and official horizons. This
deliberately differs from CrossMaze eval envs, which use plain collection maps
plus coordinate-based reset options. Reference-score data and reference-file
validation stay on the repo side (`utils.pointmaze_score`).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass

import gymnasium as gym

from crossmaze.variants import POINTMAZE_ENV_FACTS


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
    if variant not in POINTMAZE_ENV_FACTS:
        raise ValueError(f"Unknown PointMaze variant: {variant!r}")
    facts = POINTMAZE_ENV_FACTS[variant]
    if facts["kind"] != "remote":
        raise ValueError(f"Variant {variant!r} is not a remote PointMaze variant")

    map_key = _remote_eval_map_key(variant)
    maze_map = copy.deepcopy(OFFICIAL_POINTMAZE_EVAL_MAPS[map_key])
    goal_cell = _find_goal_cell(maze_map)
    reward_type = str(facts["reward_type"])
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
        env_id=facts["env_id"],
        env_kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
        reward_type=reward_type,
        goal_cell=goal_cell,
        env_fingerprint=fingerprint_score_env_spec(
            env_id=facts["env_id"],
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
    if variant not in POINTMAZE_ENV_FACTS:
        raise ValueError(f"Unknown PointMaze variant: {variant!r}")
    facts = POINTMAZE_ENV_FACTS[variant]
    if facts["kind"] != "local":
        raise ValueError(f"Variant {variant!r} is not a local PointMaze variant")

    local_eval_maps = config.get("local_eval_maps") or {}
    variant_config = local_eval_maps.get(variant)
    if not isinstance(variant_config, dict):
        raise ValueError(
            f"Local scoring for {variant!r} requires local_eval_maps.{variant}.goal_cell"
        )
    goal_cell = _normalize_goal_cell(variant_config.get("goal_cell"), variant=variant)
    base_maze_map = facts["maze_map"]
    maze_map = build_local_score_maze_map(
        variant=variant,
        base_maze_map=base_maze_map,
        goal_cell=goal_cell,
    )

    env_paras = dict(facts["env_paras"])
    env_id = env_paras.get("id", "PointMaze_UMaze-v3")
    reward_type = str(env_paras.get("reward_type", facts.get("reward_type", "sparse")))
    max_episode_steps = int(
        variant_config.get(
            "max_episode_steps",
            env_paras.get("max_episode_steps", 300),
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
    if variant not in POINTMAZE_ENV_FACTS:
        raise ValueError(f"Unknown PointMaze variant: {variant!r}")
    if POINTMAZE_ENV_FACTS[variant]["kind"] == "remote":
        return build_remote_pointmaze_score_env_spec(variant)
    return build_local_pointmaze_score_env_spec(variant, config)


def make_pointmaze_score_env(
    score_env_spec: PointMazeScoreEnvSpec,
    *,
    render_mode: str | None = None,
):
    import gymnasium_robotics  # noqa: F401 registers PointMaze envs

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
