import re

import numpy as np

from crossmaze.layout import format_visual_map as _format_visual_map  # noqa: F401 back-compat name
from crossmaze.layout import live_env_layout_overrides
from utils.maze_sensing import build_sensing, obs_xy_to_row_col, sensing_text_from_obs


ACTION_DIM = 8

_ACTION_PATTERN = re.compile(
    r"(?<![\d,])[-+]?\d+(?:\s*,\s*[-+]?\d+){7}(?!\s*,\s*[-+]?\d)"
)


def prepare_eval_prompt_vars(prompt_vars: dict, env) -> dict:
    """Use the instantiated eval env map for rollout sensing and rendering."""
    resolved = dict(prompt_vars)
    resolved.update(live_env_layout_overrides(env))
    return resolved


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.2f}" for value in values) + "]"


def format_history_observation(obs) -> np.ndarray:
    """Keep the ant torso xy position for compact trajectory history."""
    return np.asarray(obs["achieved_goal"], dtype=np.float32)


def format_history(history_entries: list[dict], meta: dict) -> dict:
    if not history_entries:
        return {"history_block_en": ""}

    lines = ["## History"]
    for entry in history_entries:
        xy = np.asarray(entry["observation"], dtype=np.float32)
        row, col = obs_xy_to_row_col(
            float(xy[0]),
            float(xy[1]),
            meta["maze_map"],
            maze_size_scaling=float(meta.get("maze_size_scaling", 4.0)),
        )
        lines.append(
            f"{int(entry['steps_ago'])} step(s) ago: "
            f"torso_xy=({float(xy[0]):.4f}, {float(xy[1]):.4f}), "
            f"cell=row {row + 1}, column {col + 1}, "
            f"action={entry['action_text']}."
        )
    return {"history_block_en": "\n" + "\n".join(lines)}


def format_obs(obs, meta: dict) -> dict:
    state = np.asarray(obs["observation"], dtype=np.float32)
    achieved_goal = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired_goal = np.asarray(obs["desired_goal"], dtype=np.float32)
    if state.shape != (27,):
        raise ValueError(
            "AntMaze expects the D4RL v4 27-dimensional observation vector, "
            f"got shape={state.shape}. Use the variant env_id/env_kwargs from the registry."
        )

    joint_angles = state[5:13]
    joint_velocities = state[19:27]
    return {
        "obs_text": (
            f"  Position: (x={float(achieved_goal[0]):.2f}, y={float(achieved_goal[1]):.2f})\n"
            f"  Goal:     (gx={float(desired_goal[0]):.2f}, gy={float(desired_goal[1]):.2f})\n"
            f"  Torso:   z={float(state[0]):.2f}, "
            f"quat=[{float(state[1]):.2f}, {float(state[2]):.2f}, "
            f"{float(state[3]):.2f}, {float(state[4]):.2f}]\n"
            "  Velocity: "
            f"linear=[{float(state[13]):.2f}, {float(state[14]):.2f}, {float(state[15]):.2f}], "
            f"angular=[{float(state[16]):.2f}, {float(state[17]):.2f}, {float(state[18]):.2f}]\n"
            f"  Joints:   q={_format_vector(joint_angles)}\n"
            f"  JointVel: dq={_format_vector(joint_velocities)}"
        ),
        **sensing_text_from_obs(obs, achieved_goal, desired_goal, meta),
    }


def format_action(action: np.ndarray) -> str:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"AntMaze action must have shape ({ACTION_DIM},), got {action.shape}")
    values = np.clip(np.round(action * 100), -100, 100).astype(np.int32)
    return ",".join(str(int(value)) for value in values)


def parse_action(text: str) -> tuple[np.ndarray, bool]:
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return np.zeros(ACTION_DIM, dtype=np.float32), False
    try:
        values = [
            float(int(part.strip())) / 100.0
            for part in match.group(0).split(",")
        ]
    except ValueError:
        return np.zeros(ACTION_DIM, dtype=np.float32), False
    if len(values) != ACTION_DIM:
        return np.zeros(ACTION_DIM, dtype=np.float32), False
    return np.asarray(values, dtype=np.float32), True


def validate_action(action: np.ndarray) -> bool:
    action = np.asarray(action)
    return bool(
        action.shape == (ACTION_DIM,)
        and np.all(np.isfinite(action))
        and np.all(action >= -1.0)
        and np.all(action <= 1.0)
    )
