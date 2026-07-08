"""Maze layout helpers shared by env wrappers and prompt-side formatters.

This module must stay dependency-light (stdlib only): it is imported by
`data/*/formatting.py`, which tokenization worker subprocesses re-import.
"""


def format_visual_map(maze_map: list[list[object]]) -> str:
    """Render a maze map as the two-space-indented `#`/`.` visual block."""
    return "\n".join(
        "  " + " ".join("#" if cell == 1 else "." for cell in row)
        for row in maze_map
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
