import math
import re

import numpy as np


def _obs_xy_to_row_col(
    x: float,
    y: float,
    maze_map: list[list[int]],
    maze_size_scaling: float = 1.0,
) -> tuple[int, int]:
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    x_map_center = cols / 2 * maze_size_scaling
    y_map_center = rows / 2 * maze_size_scaling
    row = math.floor((y_map_center - y) / maze_size_scaling)
    col = math.floor((x + x_map_center) / maze_size_scaling)
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    return row, col


def _neighbor_status(maze_map: list[list[int]], row: int, col: int, d_row: int, d_col: int) -> str:
    n_row = row + d_row
    n_col = col + d_col
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    if n_row < 0 or n_row >= rows or n_col < 0 or n_col >= cols:
        return "wall"
    return "wall" if maze_map[n_row][n_col] == 1 else "free"


def _build_map_sensing(obs: np.ndarray, goal: np.ndarray, meta: dict) -> dict:
    """Build dynamic map sensing context from the current observation and maze layout."""
    x, y = float(obs[0]), float(obs[1])
    gx, gy = float(goal[0]), float(goal[1])
    maze_map = meta["maze_map"]
    maze_size_scaling = float(meta.get("maze_size_scaling", 1.0))
    row, col = _obs_xy_to_row_col(x, y, maze_map, maze_size_scaling=maze_size_scaling)
    goal_row, goal_col = _obs_xy_to_row_col(gx, gy, maze_map, maze_size_scaling=maze_size_scaling)
    up = _neighbor_status(maze_map, row, col, -1, 0)
    down = _neighbor_status(maze_map, row, col, 1, 0)
    left = _neighbor_status(maze_map, row, col, 0, -1)
    right = _neighbor_status(maze_map, row, col, 0, 1)

    row_1 = row + 1
    col_1 = col + 1
    goal_row_1 = goal_row + 1
    goal_col_1 = goal_col + 1
    return {
        "map_sensing_en": (
            f"Current cell: row {row_1}, column {col_1}. "
            f"Goal cell: row {goal_row_1}, column {goal_col_1}. "
            "Rows and columns are counted from the top-left corner starting at 1. "
            f"Neighboring cells: up={up}, down={down}, left={left}, right={right}."
        ),
        "map_sensing_zh": (
            f"当前位置格子：第 {row_1} 行，第 {col_1} 列。"
            f"目标格子：第 {goal_row_1} 行，第 {goal_col_1} 列。"
            "行列都从左上角开始计数，起始为 1。"
            f"相邻格子：上={up}，下={down}，左={left}，右={right}。"
        ),
    }


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


def format_history(history_entries: list[dict], meta: dict) -> dict:
    """Serialize sampled trajectory history for prompt insertion.

    Each history entry must contain:
    - observation: np.ndarray with at least x/y in the first two slots
    - action_text: compact action string such as "35,-72"
    """
    if not history_entries:
        return {
            "history_block_en": "",
            "history_block_zh": "",
        }

    en_lines = ["Step history:"]
    zh_lines = ["历史轨迹："]
    for idx, entry in enumerate(history_entries, start=1):
        obs_vec = entry["observation"].astype(np.float32)
        action_text = str(entry["action_text"])
        en_lines.append(f"  {idx}. {_format_cell_and_xy(obs_vec, meta, zh=False)} Action: {action_text}.")
        zh_lines.append(f"  {idx}. {_format_cell_and_xy(obs_vec, meta, zh=True)}动作：{action_text}。")

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
        **_build_map_sensing(obs_vec, goal, meta),
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
