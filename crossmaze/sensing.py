"""Canonical maze location + wall sensing for CrossMaze.

This is the single implementation shared by the CrossMaze env wrapper, offline
tokenization, and eval rollout. Repo-side `utils.maze_sensing` re-exports these
names, so dataset cache signatures and prompt text stay unchanged.
"""

import math

import numpy as np

from crossmaze.sensing_config import (  # noqa: F401 re-exported for wrapper/formatters
    DEFAULT_MAP_SENSING_BOUNDARY_RISK_THRESHOLD,
    DEFAULT_WALL_SENSING_VERSION,
    resolve_map_sensing_boundary_risk_threshold,
    resolve_sensing_config,
    resolve_wall_sensing_version,
)

DEFAULT_BOUNDARY_RISK_FRACTION = DEFAULT_MAP_SENSING_BOUNDARY_RISK_THRESHOLD

# Observation key under which CrossMazeEnv attaches structured sensing state.
CROSSMAZE_OBS_KEY = "crossmaze"

# Fixed observation contract for neighbor_status: [up, down, left, right].
NEIGHBOR_DIRECTIONS = ("up", "down", "left", "right")
NEIGHBOR_STATUS_FREE = 0
NEIGHBOR_STATUS_WALL = 1
NEIGHBOR_STATUS_RISK = 2

_NEIGHBOR_STATUS_TEXT = {
    NEIGHBOR_STATUS_FREE: "free",
    NEIGHBOR_STATUS_WALL: "wall",
    NEIGHBOR_STATUS_RISK: "risk",
}


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


def _cell_status(maze_map: list[list[object]], row: int, col: int) -> int:
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return NEIGHBOR_STATUS_WALL
    if _is_free_cell(maze_map, row, col):
        return NEIGHBOR_STATUS_FREE
    return NEIGHBOR_STATUS_WALL


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
        _cell_status(maze_map, row + side_d_row, col + side_d_col)
        == NEIGHBOR_STATUS_FREE
        and _cell_status(
            maze_map,
            row + diagonal_d_row,
            col + diagonal_d_col,
        )
        == NEIGHBOR_STATUS_WALL
    )


def _near_cell_boundaries(
    *,
    x: float,
    y: float,
    row: int,
    col: int,
    rows: int,
    cols: int,
    maze_size_scaling: float,
    boundary_risk_threshold: float | None,
) -> dict[str, bool]:
    threshold_fraction = resolve_map_sensing_boundary_risk_threshold(
        boundary_risk_threshold
    )
    threshold = threshold_fraction * maze_size_scaling
    left_x, right_x, bottom_y, top_y = _cell_bounds(
        row,
        col,
        rows,
        cols,
        maze_size_scaling,
    )
    return {
        "left": x - left_x <= threshold,
        "right": right_x - x <= threshold,
        "bottom": y - bottom_y <= threshold,
        "top": top_y - y <= threshold,
    }


def _near_opposite_boundary(
    d_row: int,
    d_col: int,
    near: dict[str, bool],
) -> bool:
    if d_row < 0:
        return near["bottom"]
    if d_row > 0:
        return near["top"]
    if d_col < 0:
        return near["right"]
    if d_col > 0:
        return near["left"]
    return False


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
    wall_sensing_version: str | None = None,
) -> int:
    version = resolve_wall_sensing_version(wall_sensing_version)
    n_row = row + d_row
    n_col = col + d_col
    direct_status = _cell_status(maze_map, n_row, n_col)
    if version == "v1":
        return direct_status
    if x is None or y is None:
        return direct_status

    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    near = _near_cell_boundaries(
        x=x,
        y=y,
        row=row,
        col=col,
        rows=rows,
        cols=cols,
        maze_size_scaling=maze_size_scaling,
        boundary_risk_threshold=boundary_risk_threshold,
    )
    if direct_status == NEIGHBOR_STATUS_WALL:
        if version == "v4" and _near_opposite_boundary(d_row, d_col, near):
            return NEIGHBOR_STATUS_FREE
        return NEIGHBOR_STATUS_WALL

    if d_col:
        side_checks = ((-1, 0, near["top"]), (1, 0, near["bottom"]))
    else:
        side_checks = ((0, -1, near["left"]), (0, 1, near["right"]))

    for side_d_row, side_d_col, near_side in side_checks:
        if not near_side:
            continue
        diagonal_is_wall = (
            _cell_status(maze_map, row + d_row + side_d_row, col + d_col + side_d_col)
            == NEIGHBOR_STATUS_WALL
        )
        if version == "v2" and diagonal_is_wall:
            return NEIGHBOR_STATUS_WALL
        if version in {"v3", "v4", "v5"} and _has_new_corner(
            maze_map,
            row,
            col,
            side_d_row,
            side_d_col,
            d_row + side_d_row,
            d_col + side_d_col,
        ):
            return NEIGHBOR_STATUS_RISK if version == "v5" else NEIGHBOR_STATUS_WALL
    return NEIGHBOR_STATUS_FREE


def compute_sensing_state(
    position: np.ndarray,
    goal: np.ndarray,
    meta: dict,
) -> dict:
    """Compute structured location and versioned four-neighbor sensing state.

    Returns a JSON-friendly dict with position/goal xy, 0-based position/goal
    cells, neighbor status codes ordered as [up, down, left, right], and the
    resolved sensing config. Status codes are 0=free, 1=wall, and 2=risk.
    """
    x, y = float(position[0]), float(position[1])
    gx, gy = float(goal[0]), float(goal[1])
    maze_map = meta["maze_map"]
    maze_size_scaling = float(meta.get("maze_size_scaling", 1.0))
    sensing_config = resolve_sensing_config(meta)
    wall_sensing_version = sensing_config.get(
        "wall_sensing_version",
        DEFAULT_WALL_SENSING_VERSION,
    )
    boundary_risk_threshold = sensing_config["map_sensing_boundary_risk_threshold"]
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
        "wall_sensing_version": wall_sensing_version,
    }
    up = _neighbor_status(maze_map, row, col, -1, 0, **neighbor_kwargs)
    down = _neighbor_status(maze_map, row, col, 1, 0, **neighbor_kwargs)
    left = _neighbor_status(maze_map, row, col, 0, -1, **neighbor_kwargs)
    right = _neighbor_status(maze_map, row, col, 0, 1, **neighbor_kwargs)

    return {
        "position_xy": [x, y],
        "goal_xy": [gx, gy],
        "position_cell": [row, col],
        "goal_cell": [goal_row, goal_col],
        "neighbor_status": [up, down, left, right],
        "wall_sensing_version": wall_sensing_version,
        "map_sensing_boundary_risk_threshold": boundary_risk_threshold,
    }


def _neighbor_status_text(neighbor_status) -> tuple[str, str, str, str]:
    try:
        values = list(neighbor_status)
    except TypeError as exc:
        raise ValueError(
            "neighbor_status must be [up, down, left, right] status codes"
        ) from exc
    if len(values) != len(NEIGHBOR_DIRECTIONS):
        raise ValueError(
            "neighbor_status must contain four status codes ordered as "
            "[up, down, left, right]"
        )

    rendered = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Invalid neighbor status code: {value!r}")
        try:
            code = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid neighbor status code: {value!r}") from exc
        if code != value or code not in _NEIGHBOR_STATUS_TEXT:
            raise ValueError(f"Invalid neighbor status code: {value!r}")
        rendered.append(_NEIGHBOR_STATUS_TEXT[code])
    return tuple(rendered)


def render_sensing_text(state: dict) -> dict:
    """Render prompt text from the numeric structured sensing state."""
    row, col = state["position_cell"]
    goal_row, goal_col = state["goal_cell"]
    up, down, left, right = _neighbor_status_text(state["neighbor_status"])

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


def build_sensing(position: np.ndarray, goal: np.ndarray, meta: dict) -> dict:
    """Build shared location and versioned four-neighbor wall/risk sensing."""
    return render_sensing_text(compute_sensing_state(position, goal, meta))


def _state_matches_meta(state: dict, meta: dict) -> bool:
    """Return whether a structured sensing state was computed under `meta`.

    Compares the resolved sensing config plus maze layout so any drift makes
    callers fall back to recomputing sensing from `meta` directly.
    """
    if not isinstance(state, dict):
        return False
    try:
        _neighbor_status_text(state.get("neighbor_status"))
    except ValueError:
        return False
    try:
        sensing_config = resolve_sensing_config(meta)
    except (TypeError, ValueError):
        return False
    if state.get("wall_sensing_version") != sensing_config["wall_sensing_version"]:
        return False
    if (
        state.get("map_sensing_boundary_risk_threshold")
        != sensing_config["map_sensing_boundary_risk_threshold"]
    ):
        return False
    state_scaling = state.get("maze_size_scaling")
    if state_scaling is None:
        return False
    try:
        if float(state_scaling) != float(meta.get("maze_size_scaling", 1.0)):
            return False
    except (TypeError, ValueError):
        return False
    state_map = state.get("maze_map")
    meta_map = meta.get("maze_map")
    if state_map is None or meta_map is None:
        return False
    return [list(row) for row in state_map] == [list(row) for row in meta_map]


def sensing_text_from_obs(
    obs,
    position: np.ndarray,
    goal: np.ndarray,
    meta: dict,
) -> dict:
    """Render sensing text, reusing CrossMaze wrapper state when it matches.

    Any mismatch between the attached state and `meta` (sensing version,
    threshold, scaling, maze map) silently falls back to recomputing from
    `meta`, so prompt text can never drift from the legacy path.
    """
    state = obs.get(CROSSMAZE_OBS_KEY) if hasattr(obs, "get") else None
    if state is not None and _state_matches_meta(state, meta):
        return render_sensing_text(state)
    return build_sensing(position, goal, meta)
