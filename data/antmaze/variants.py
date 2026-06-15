def _format_visual_map(maze_map: list[list[int]]) -> str:
    return "\n".join(
        "  " + " ".join("#" if cell == 1 else "." for cell in row)
        for row in maze_map
    )


def _build_prompt_vars(
    *,
    env_name: str,
    dataset_style: str,
    maze_map: list[list[int]],
    structure_desc_en: str,
) -> dict:
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    return {
        "env_name": env_name,
        "dataset_style": dataset_style,
        "maze_map": maze_map,
        "maze_size_scaling": 4.0,
        "maze_shape": f"{rows}x{cols}",
        "maze_visual": _format_visual_map(maze_map),
        "structure_desc_en": structure_desc_en,
    }


_UMAZE = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
]

_MEDIUM = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

_LARGE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

_UMAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, "r", 1],
    [1, 0, 1, 1, 1],
    [1, 0, 0, "g", 1],
    [1, 1, 1, 1, 1],
]

_MEDIUM_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, "r", 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, "g", 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

_LARGE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, "r", 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, "g", 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def _variant(
    *,
    dataset_id: str,
    env_id: str,
    env_name: str,
    dataset_style: str,
    maze_map: list[list[int]],
    eval_maze_map: list[list[int | str]],
    structure_desc_en: str,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "env_id": env_id,
        "env_kwargs": {
            "maze_map": eval_maze_map,
            "reward_type": "sparse",
            "continuing_task": True,
            "reset_target": False,
        },
        "prompt_vars": _build_prompt_vars(
            env_name=env_name,
            dataset_style=dataset_style,
            maze_map=maze_map,
            structure_desc_en=structure_desc_en,
        ),
    }


ANTMAZE_VARIANTS = {
    "umaze": _variant(
        dataset_id="D4RL/antmaze/umaze-v1",
        env_id="AntMaze_UMaze-v4",
        env_name="AntMaze UMaze",
        dataset_style="fixed reset and goal locations",
        maze_map=_UMAZE,
        eval_maze_map=_UMAZE_EVAL,
        structure_desc_en="A compact U-shaped maze with one long route around a central wall.",
    ),
    "umaze-diverse": _variant(
        dataset_id="D4RL/antmaze/umaze-diverse-v1",
        env_id="AntMaze_UMaze-v4",
        env_name="AntMaze UMaze Diverse",
        dataset_style="diverse offline trajectories in the U-shaped maze",
        maze_map=_UMAZE,
        eval_maze_map=_UMAZE_EVAL,
        structure_desc_en="A compact U-shaped maze with one long route around a central wall.",
    ),
    "medium-play": _variant(
        dataset_id="D4RL/antmaze/medium-play-v1",
        env_id="AntMaze_Medium-v4",
        env_name="AntMaze Medium Play",
        dataset_style="play-style trajectories with varied starts and goals",
        maze_map=_MEDIUM,
        eval_maze_map=_MEDIUM_EVAL,
        structure_desc_en="A medium maze with several corridors, turns, and dead ends.",
    ),
    "medium-diverse": _variant(
        dataset_id="D4RL/antmaze/medium-diverse-v1",
        env_id="AntMaze_Medium_Diverse_GR-v4",
        env_name="AntMaze Medium Diverse",
        dataset_style="diverse-goal-and-reset trajectories",
        maze_map=_MEDIUM,
        eval_maze_map=_MEDIUM_EVAL,
        structure_desc_en="A medium maze with several corridors, turns, and dead ends.",
    ),
    "large-play": _variant(
        dataset_id="D4RL/antmaze/large-play-v1",
        env_id="AntMaze_Large-v4",
        env_name="AntMaze Large Play",
        dataset_style="play-style trajectories with varied starts and goals",
        maze_map=_LARGE,
        eval_maze_map=_LARGE_EVAL,
        structure_desc_en="A large maze with long corridors, branches, and narrow bottlenecks.",
    ),
    "large-diverse": _variant(
        dataset_id="D4RL/antmaze/large-diverse-v1",
        env_id="AntMaze_Large_Diverse_GR-v4",
        env_name="AntMaze Large Diverse",
        dataset_style="diverse-goal-and-reset trajectories",
        maze_map=_LARGE,
        eval_maze_map=_LARGE_EVAL,
        structure_desc_en="A large maze with long corridors, branches, and narrow bottlenecks.",
    ),
}
