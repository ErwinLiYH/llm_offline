import math

import numpy as np


DEFAULT_BOUNDARY_RISK_FRACTION = 0.10


def obs_xy_to_row_col(
    x: float,
    y: float,
    maze_map: list[list[object]],
    maze_size_scaling: float = 1.0,
) -> tuple[int, int]:
    """Map continuous maze xy coordinates to a free cell when possible."""
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    if rows == 0 or cols == 0:
        raise ValueError("maze_map must be non-empty")

    x_map_center = cols / 2 * maze_size_scaling
    y_map_center = rows / 2 * maze_size_scaling
    row = math.floor((y_map_center - y) / maze_size_scaling)
    col = math.floor((x + x_map_center) / maze_size_scaling)
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    if _is_free_cell(maze_map, row, col):
        return row, col

    snapped = _nearest_free_row_col(
        x,
        y,
        maze_map,
        maze_size_scaling=maze_size_scaling,
    )
    if snapped is not None:
        return snapped
    return row, col


def _is_free_cell(maze_map: list[list[object]], row: int, col: int) -> bool:
    return maze_map[row][col] != 1


def _cell_center_xy(
    row: int,
    col: int,
    rows: int,
    cols: int,
    maze_size_scaling: float,
) -> tuple[float, float]:
    x_map_center = cols / 2 * maze_size_scaling
    y_map_center = rows / 2 * maze_size_scaling
    x = (col + 0.5) * maze_size_scaling - x_map_center
    y = y_map_center - (row + 0.5) * maze_size_scaling
    return x, y


def _cell_bounds(
    row: int,
    col: int,
    rows: int,
    cols: int,
    maze_size_scaling: float,
) -> tuple[float, float, float, float]:
    x_map_center = cols / 2 * maze_size_scaling
    y_map_center = rows / 2 * maze_size_scaling
    left_x = col * maze_size_scaling - x_map_center
    right_x = (col + 1) * maze_size_scaling - x_map_center
    top_y = y_map_center - row * maze_size_scaling
    bottom_y = y_map_center - (row + 1) * maze_size_scaling
    return left_x, right_x, bottom_y, top_y


def _nearest_free_row_col(
    x: float,
    y: float,
    maze_map: list[list[object]],
    maze_size_scaling: float = 1.0,
) -> tuple[int, int] | None:
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    best_cell = None
    best_dist = float("inf")
    for row, row_values in enumerate(maze_map):
        for col, _value in enumerate(row_values):
            if not _is_free_cell(maze_map, row, col):
                continue
            center_x, center_y = _cell_center_xy(
                row,
                col,
                rows,
                cols,
                maze_size_scaling,
            )
            dist = (center_x - x) ** 2 + (center_y - y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_cell = (row, col)
    return best_cell


def _cell_status(maze_map: list[list[object]], row: int, col: int) -> str:
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return "wall"
    return "free" if _is_free_cell(maze_map, row, col) else "wall"


def _has_new_corner(
    maze_map: list[list[object]],
    row: int,
    col: int,
    side_d_row: int,
    side_d_col: int,
    diagonal_d_row: int,
    diagonal_d_col: int,
) -> bool:
    """Return whether a free side closes into a wall at the next cell."""
    return (
        _cell_status(maze_map, row + side_d_row, col + side_d_col) == "free"
        and _cell_status(
            maze_map,
            row + diagonal_d_row,
            col + diagonal_d_col,
        )
        == "wall"
    )


def _neighbor_status(
    maze_map: list[list[object]],
    row: int,
    col: int,
    d_row: int,
    d_col: int,
    *,
    x: float | None = None,
    y: float | None = None,
    maze_size_scaling: float = 1.0,
    boundary_risk_threshold: float | None = None,
) -> str:
    n_row = row + d_row
    n_col = col + d_col
    if _cell_status(maze_map, n_row, n_col) == "wall":
        return "wall"
    if x is None or y is None:
        return "free"

    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    threshold = (
        float(boundary_risk_threshold)
        if boundary_risk_threshold is not None
        else DEFAULT_BOUNDARY_RISK_FRACTION * maze_size_scaling
    )
    left_x, right_x, bottom_y, top_y = _cell_bounds(
        row,
        col,
        rows,
        cols,
        maze_size_scaling,
    )
    near_left = x - left_x <= threshold
    near_right = right_x - x <= threshold
    near_bottom = y - bottom_y <= threshold
    near_top = top_y - y <= threshold

    if d_col:
        side_checks = ((-1, 0, near_top), (1, 0, near_bottom))
    else:
        side_checks = ((0, -1, near_left), (0, 1, near_right))

    for side_d_row, side_d_col, near_side in side_checks:
        if near_side and _has_new_corner(
            maze_map,
            row,
            col,
            side_d_row,
            side_d_col,
            d_row + side_d_row,
            d_col + side_d_col,
        ):
            return "wall"
    return "free"


def build_sensing(position: np.ndarray, goal: np.ndarray, meta: dict) -> dict:
    """Build shared location and conservative four-neighbor wall sensing."""
    x, y = float(position[0]), float(position[1])
    gx, gy = float(goal[0]), float(goal[1])
    maze_map = meta["maze_map"]
    maze_size_scaling = float(meta.get("maze_size_scaling", 1.0))
    boundary_risk_threshold = meta.get("map_sensing_boundary_risk_threshold")
    if boundary_risk_threshold is not None:
        boundary_risk_threshold = float(boundary_risk_threshold)
    row, col = obs_xy_to_row_col(
        x,
        y,
        maze_map,
        maze_size_scaling=maze_size_scaling,
    )
    goal_row, goal_col = obs_xy_to_row_col(
        gx,
        gy,
        maze_map,
        maze_size_scaling=maze_size_scaling,
    )
    neighbor_kwargs = {
        "x": x,
        "y": y,
        "maze_size_scaling": maze_size_scaling,
        "boundary_risk_threshold": boundary_risk_threshold,
    }
    up = _neighbor_status(maze_map, row, col, -1, 0, **neighbor_kwargs)
    down = _neighbor_status(maze_map, row, col, 1, 0, **neighbor_kwargs)
    left = _neighbor_status(maze_map, row, col, 0, -1, **neighbor_kwargs)
    right = _neighbor_status(maze_map, row, col, 0, 1, **neighbor_kwargs)

    row_1 = row + 1
    col_1 = col + 1
    goal_row_1 = goal_row + 1
    goal_col_1 = goal_col + 1
    return {
        "location_sensing_en": (
            f"Current cell: row {row_1}, column {col_1}. "
            f"Goal cell: row {goal_row_1}, column {goal_col_1}. "
            "Rows and columns are counted from the top-left corner starting at 1."
        ),
        "wall_sensing_en": (
            f"Neighboring cells: up={up}, down={down}, left={left}, right={right}."
        ),
        "location_sensing_zh": (
            f"当前位置格子：第 {row_1} 行，第 {col_1} 列。"
            f"目标格子：第 {goal_row_1} 行，第 {goal_col_1} 列。"
            "行列都从左上角开始计数，起始为 1。"
        ),
        "wall_sensing_zh": (
            f"相邻格子：上={up}，下={down}，左={left}，右={right}。"
        ),
    }
