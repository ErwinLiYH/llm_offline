"""Maze layout helpers shared by env wrappers and prompt-side formatters.

This module must stay dependency-light (stdlib only): it is imported by
`data/*/formatting.py`, which tokenization worker subprocesses re-import.
"""

DYNAMIC_MAP_CURRENT = 2
DYNAMIC_MAP_GOAL = 3
DYNAMIC_MAP_SUCCESS = 4


def format_visual_map(maze_map: list[list[object]]) -> str:
    """Render a maze map as the two-space-indented `#`/`.` visual block."""
    return "\n".join(
        "  " + " ".join("#" if cell == 1 else "." for cell in row)
        for row in maze_map
    )


def _validate_dynamic_map_cell(
    cell,
    *,
    name: str,
    rows: int,
    cols: int,
) -> tuple[int, int]:
    try:
        row, col = cell
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain exactly [row, col]") from exc
    if isinstance(row, bool) or isinstance(col, bool):
        raise ValueError(f"{name} must contain integer row/column indices")
    try:
        row_int = int(row)
        col_int = int(col)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain integer row/column indices") from exc
    if row_int != row or col_int != col:
        raise ValueError(f"{name} must contain integer row/column indices")
    if not (0 <= row_int < rows and 0 <= col_int < cols):
        raise ValueError(
            f"{name} is outside the {rows}x{cols} maze: [{row_int}, {col_int}]"
        )
    return row_int, col_int


def build_dynamic_map(
    maze_map: list[list[object]],
    position_cell,
    goal_cell,
) -> list[list[object]]:
    """Copy a static maze and mark current/goal cells with numeric codes.

    The marker contract is 2=current, 3=goal, and 4=current+goal. The input
    map is never modified.
    """
    if not maze_map or not maze_map[0]:
        raise ValueError("maze_map must be non-empty")
    rows = len(maze_map)
    cols = len(maze_map[0])
    if any(len(row) != cols for row in maze_map):
        raise ValueError("maze_map must be rectangular")

    position_row, position_col = _validate_dynamic_map_cell(
        position_cell,
        name="position_cell",
        rows=rows,
        cols=cols,
    )
    goal_row, goal_col = _validate_dynamic_map_cell(
        goal_cell,
        name="goal_cell",
        rows=rows,
        cols=cols,
    )

    dynamic_map = [list(row) for row in maze_map]
    if (position_row, position_col) == (goal_row, goal_col):
        dynamic_map[position_row][position_col] = DYNAMIC_MAP_SUCCESS
    else:
        dynamic_map[position_row][position_col] = DYNAMIC_MAP_CURRENT
        dynamic_map[goal_row][goal_col] = DYNAMIC_MAP_GOAL
    return dynamic_map


def format_dynamic_visual_map(dynamic_map: list[list[object]]) -> str:
    """Render a numeric dynamic map as an indented `#`/`.`/`C`/`G`/`S` block."""
    cell_text = {
        1: "#",
        DYNAMIC_MAP_CURRENT: "C",
        DYNAMIC_MAP_GOAL: "G",
        DYNAMIC_MAP_SUCCESS: "S",
    }
    return "\n".join(
        "  " + " ".join(cell_text.get(cell, ".") for cell in row)
        for row in dynamic_map
    )


def format_raw_matrix(maze_map: list[list[int]]) -> str:
    """Render a maze map as the two-space-indented Python row-list block."""
    return "\n".join(f"  {row}" for row in maze_map)


def maze_shape_text(maze_map: list[list[object]]) -> str:
    """Render the `<rows>x<cols>` shape string used in prompt vars."""
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    return f"{rows}x{cols}"


def live_env_layout_overrides(env) -> dict:
    """Read layout fields from an instantiated Gymnasium Robotics maze env.

    Returns the exact prompt-var overrides historically produced by the
    AntMaze formatter's `prepare_eval_prompt_vars`: the live map can differ
    from the offline collection map (for example UMaze wall orientation).
    """
    maze = env.unwrapped.maze
    maze_map = [list(row) for row in maze.maze_map]
    return {
        "maze_map": maze_map,
        "maze_size_scaling": float(maze.maze_size_scaling),
        "maze_shape": f"{len(maze_map)}x{len(maze_map[0])}",
        "maze_visual": format_visual_map(maze_map),
    }


def static_layout_from_prompt_vars(prompt_vars: dict) -> dict:
    """Extract the static variant layout used for sensing and rendering.

    PointMaze eval and score-mode rollouts sense against the static variant
    map (score envs carry a goal-marked map that is intentionally not used
    for prompts).
    """
    maze_map = [list(row) for row in prompt_vars["maze_map"]]
    return {
        "maze_map": maze_map,
        "maze_size_scaling": float(prompt_vars.get("maze_size_scaling", 1.0)),
    }
