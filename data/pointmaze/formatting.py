import re

import numpy as np

from utils.maze_sensing import (
    build_sensing as _build_sensing,
    obs_xy_to_row_col as _obs_xy_to_row_col,
)


def _format_cell_and_xy(vec: np.ndarray, meta: dict, *, zh: bool = False) -> str:
    row, col = _obs_xy_to_row_col(
        float(vec[0]),
        float(vec[1]),
        meta["maze_map"],
        maze_size_scaling=float(meta.get("maze_size_scaling", 1.0)),
    )
    row_1 = row + 1
    col_1 = col + 1
    if zh:
        return (
            f"起始格子：第 {row_1} 行，第 {col_1} 列；"
            f"起始坐标：(x={float(vec[0]):.4f}, y={float(vec[1]):.4f})。"
        )
    return (
        f"Start cell: row {row_1}, column {col_1}; "
        f"start xy: (x={float(vec[0]):.4f}, y={float(vec[1]):.4f})."
    )


def _format_history_entry(entry: dict, meta: dict, *, zh: bool = False) -> str:
    obs_vec = entry["observation"].astype(np.float32)
    action_text = str(entry["action_text"])
    steps_ago = int(entry["steps_ago"])
    row, col = _obs_xy_to_row_col(
        float(obs_vec[0]),
        float(obs_vec[1]),
        meta["maze_map"],
        maze_size_scaling=float(meta.get("maze_size_scaling", 1.0)),
    )
    row_1 = row + 1
    col_1 = col + 1
    if zh:
        return (
            f"前 {steps_ago} 步的起始点为 "
            f"(x={float(obs_vec[0]):.4f}, y={float(obs_vec[1]):.4f})，"
            f"所在格为第 {row_1} 行，第 {col_1} 列，"
            f"动作为 {action_text}。"
        )
    return (
        f"The start point {steps_ago} steps before the current step was "
        f"(x={float(obs_vec[0]):.4f}, y={float(obs_vec[1]):.4f}), "
        f"the cell was row {row_1}, column {col_1}, "
        f"and the action was {action_text}."
    )


def format_history_observation(obs) -> np.ndarray:
    """Keep the point-mass state vector used by PointMaze history prompts."""
    return np.asarray(obs["observation"], dtype=np.float32)


def format_history(history_entries: list[dict], meta: dict) -> dict:
    """Serialize sampled trajectory history for prompt insertion.

    Each history entry must contain:
    - observation: np.ndarray with at least x/y in the first two slots
    - action_text: compact action string such as "35,-72"
    - steps_ago: how many executed steps before the current step this entry came from
    """
    if not history_entries:
        return {
            "history_block_en": "",
            "history_block_zh": "",
        }

    en_lines = ["## History"]
    zh_lines = ["## History"]
    for entry in history_entries:
        en_lines.append(_format_history_entry(entry, meta, zh=False))
        zh_lines.append(_format_history_entry(entry, meta, zh=True))

    return {
        "history_block_en": "\n" + "\n".join(en_lines),
        "history_block_zh": "\n" + "\n".join(zh_lines),
    }


def format_obs(obs, meta: dict) -> dict:
    """Serialize PointMaze observation plus derived context for prompt insertion.

    Returns a dict that must contain obs_text and may contain extra prompt vars.
    """
    obs_vec = obs["observation"].astype(np.float32)
    goal = obs["desired_goal"].astype(np.float32)
    x, y, vx, vy = obs_vec
    gx, gy = goal
    return {
        "obs_text": (
            f"  Position: (x={x:.4f}, y={y:.4f})\n"
            f"  Velocity: (vx={vx:.4f}, vy={vy:.4f})\n"
            f"  Goal:     (gx={gx:.4f}, gy={gy:.4f})"
        ),
        **_build_sensing(obs_vec, goal, meta),
    }


def format_action(action: np.ndarray) -> str:
    """Serialize a 2D action vector into the training target text."""
    ax, ay = action
    return f"{int(np.clip(np.round(ax * 100), -100, 100))},{int(np.clip(np.round(ay * 100), -100, 100))}"


_ACTION_PATTERN = re.compile(
    r"[-+]?\d+\s*,\s*[-+]?\d+"
)


def parse_action(text: str) -> tuple[np.ndarray, bool]:
    """Parse model output text into a 2D action vector.

    Returns (action, success). On failure returns (zeros, False).
    """
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return np.zeros(2, dtype=np.float32), False
    try:
        parts = match.group(0).split(",")
        ax = float(int(parts[0].strip())) / 100.0
        ay = float(int(parts[1].strip())) / 100.0
        action = np.array([ax, ay], dtype=np.float32)
        return action, True
    except (ValueError, IndexError):
        return np.zeros(2, dtype=np.float32), False


def validate_action(action: np.ndarray) -> bool:
    """Return True if all action components are within [-1, 1]."""
    return bool(np.all(action >= -1.0) and np.all(action <= 1.0))
