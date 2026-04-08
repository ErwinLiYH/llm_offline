def _format_raw_matrix(maze_map: list[list[int]]) -> str:
    return "\n".join(f"  {row}" for row in maze_map)


def _format_visual_map(maze_map: list[list[int]]) -> str:
    return "\n".join(
        "  " + " ".join("#" if cell == 1 else "." for cell in row)
        for row in maze_map
    )


def _build_prompt_vars(
    *,
    env_name: str,
    reward_type: str,
    maze_map: list[list[int]],
    structure_desc_en: str,
    structure_desc_zh: str,
) -> dict:
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    return {
        "reward_type": reward_type,
        "maze_map": maze_map,
        "env_name": env_name,
        "reward_desc_en": f"{reward_type} reward",
        "reward_desc_zh": "稀疏奖励" if reward_type == "sparse" else "稠密奖励",
        "maze_shape": f"{rows}x{cols}",
        "maze_raw_matrix": _format_raw_matrix(maze_map),
        "maze_visual": _format_visual_map(maze_map),
        "structure_desc_en": structure_desc_en,
        "structure_desc_zh": structure_desc_zh,
    }


_OPEN_MAZE = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

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
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


POINTMAZE_VARIANTS = {
    "open": {
        "dataset_id": "D4RL/pointmaze/open-v2",
        "env_id": "PointMaze_Open-v3",
        "prompt_vars": _build_prompt_vars(
            env_name="PointMaze Open",
            reward_type="sparse",
            maze_map=_OPEN_MAZE,
            structure_desc_en="A 5x7 open arena with walls on the outer boundary and no internal obstacles.",
            structure_desc_zh="一个 5x7 的开放迷宫，只有外边界墙壁，内部没有障碍。",
        ),
    },
    "open-dense": {
        "dataset_id": "D4RL/pointmaze/open-dense-v2",
        "env_id": "PointMaze_OpenDense-v3",
        "prompt_vars": _build_prompt_vars(
            env_name="PointMaze OpenDense",
            reward_type="dense",
            maze_map=_OPEN_MAZE,
            structure_desc_en="A 5x7 open arena with walls on the outer boundary and no internal obstacles.",
            structure_desc_zh="一个 5x7 的开放迷宫，只有外边界墙壁，内部没有障碍。",
        ),
    },
    "umaze": {
        "dataset_id": "D4RL/pointmaze/umaze-v2",
        "env_id": "PointMaze_UMaze-v3",
        "prompt_vars": _build_prompt_vars(
            env_name="PointMaze UMaze",
            reward_type="sparse",
            maze_map=_UMAZE,
            structure_desc_en="A compact 5x5 U-shaped maze with a single narrow corridor wrapping around the central wall block.",
            structure_desc_zh="一个紧凑的 5x5 U 形迷宫，需要沿着中央墙块外侧的狭窄通道绕行。",
        ),
    },
    "umaze-dense": {
        "dataset_id": "D4RL/pointmaze/umaze-dense-v2",
        "env_id": "PointMaze_UMazeDense-v3",
        "prompt_vars": _build_prompt_vars(
            env_name="PointMaze UMazeDense",
            reward_type="dense",
            maze_map=_UMAZE,
            structure_desc_en="A compact 5x5 U-shaped maze with a single narrow corridor wrapping around the central wall block.",
            structure_desc_zh="一个紧凑的 5x5 U 形迷宫，需要沿着中央墙块外侧的狭窄通道绕行。",
        ),
    },
    "medium": {
        "dataset_id": "D4RL/pointmaze/medium-v2",
        "env_id": "PointMaze_Medium-v3",
        "prompt_vars": _build_prompt_vars(
            env_name="PointMaze Medium",
            reward_type="sparse",
            maze_map=_MEDIUM,
            structure_desc_en="An 8x8 maze with multiple corridors, dead ends, and several turns between start regions and goals.",
            structure_desc_zh="一个 8x8 的中型迷宫，包含多条走廊、死路和若干转弯，需要绕开内部墙体前往目标。",
        ),
    },
    "medium-dense": {
        "dataset_id": "D4RL/pointmaze/medium-dense-v2",
        "env_id": "PointMaze_MediumDense-v3",
        "prompt_vars": _build_prompt_vars(
            env_name="PointMaze MediumDense",
            reward_type="dense",
            maze_map=_MEDIUM,
            structure_desc_en="An 8x8 maze with multiple corridors, dead ends, and several turns between start regions and goals.",
            structure_desc_zh="一个 8x8 的中型迷宫，包含多条走廊、死路和若干转弯，需要绕开内部墙体前往目标。",
        ),
    },
    "large": {
        "dataset_id": "D4RL/pointmaze/large-v2",
        "env_id": "PointMaze_Large-v3",
        "prompt_vars": _build_prompt_vars(
            env_name="PointMaze Large",
            reward_type="sparse",
            maze_map=_LARGE,
            structure_desc_en="A large maze with long corridors, many branches, and several bottlenecks created by dense internal walls.",
            structure_desc_zh="一个大型迷宫，包含长走廊、多个分支以及由密集内墙形成的若干瓶颈通道。",
        ),
    },
    "large-dense": {
        "dataset_id": "D4RL/pointmaze/large-dense-v2",
        "env_id": "PointMaze_LargeDense-v3",
        "prompt_vars": _build_prompt_vars(
            env_name="PointMaze LargeDense",
            reward_type="dense",
            maze_map=_LARGE,
            structure_desc_en="A large maze with long corridors, many branches, and several bottlenecks created by dense internal walls.",
            structure_desc_zh="一个大型迷宫，包含长走廊、多个分支以及由密集内墙形成的若干瓶颈通道。",
        ),
    },
}
