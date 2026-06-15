from pathlib import Path


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
        "maze_size_scaling": 1.0,
        "env_name": env_name,
        "reward_desc_en": f"{reward_type} reward",
        "reward_desc_zh": "稀疏奖励" if reward_type == "sparse" else "稠密奖励",
        "maze_shape": f"{rows}x{cols}",
        "maze_raw_matrix": _format_raw_matrix(maze_map),
        "maze_visual": _format_visual_map(maze_map),
        "structure_desc_en": structure_desc_en,
        "structure_desc_zh": structure_desc_zh,
    }


def _default_local_max_episode_steps(maze_map: list[list[int]]) -> int:
    """Use a finite eval horizon for local layouts.

    Official PointMaze horizons are roughly 300 for small maps and 800 for
    large maps. Scaling by map area keeps custom layouts bounded without making
    small layouts too short.
    """
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    return max(300, rows * cols * 6)


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
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

_LOCAL_LAYOUT_01 = [
    [1, 1, 1, 1, 1, 1, 1, 1],  ########
    [1, 0, 0, 0, 1, 0, 0, 1],  #...#..#
    [1, 0, 1, 0, 1, 0, 1, 1],  #.#.#.##
    [1, 0, 1, 0, 0, 0, 0, 1],  #.#....#
    [1, 0, 1, 1, 1, 1, 0, 1],  #.####.#
    [1, 0, 0, 0, 0, 1, 0, 1],  #....#.#
    [1, 1, 1, 1, 0, 0, 0, 1],  ####...#
    [1, 1, 1, 1, 1, 1, 1, 1],  ########
]

_LOCAL_LAYOUT_02 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ##########
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #....#...#
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  #.##.#.#.#
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #..#...#.#
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  ##.###.#.#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #....#...#
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],  #.##...#.#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ##########
]

_LOCAL_LAYOUT_03 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],  #########
    [1, 0, 0, 0, 1, 0, 0, 0, 1],  #...#...#
    [1, 0, 1, 0, 1, 0, 1, 0, 1],  #.#.#.#.#
    [1, 0, 1, 0, 0, 0, 1, 0, 1],  #.#...#.#
    [1, 0, 1, 1, 1, 0, 1, 0, 1],  #.###.#.#
    [1, 0, 0, 0, 1, 0, 0, 0, 1],  #...#...#
    [1, 1, 1, 0, 1, 1, 1, 0, 1],  ###.###.#
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  #.......#
    [1, 1, 1, 1, 1, 1, 1, 1, 1],  #########
]

_LOCAL_LAYOUT_04 = [
    [1, 1, 1, 1, 1, 1, 1, 1],  ########
    [1, 0, 0, 0, 0, 0, 1, 1],  #.....##
    [1, 0, 1, 1, 1, 0, 0, 1],  #.###..#
    [1, 0, 1, 0, 0, 0, 1, 1],  #.#...##
    [1, 0, 1, 0, 1, 1, 1, 1],  #.#.####
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 1, 1, 1, 1, 1, 0, 1],  ######.#
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 0, 1, 1, 1, 1, 0, 1],  #.####.#
    [1, 1, 1, 1, 1, 1, 1, 1],  ########
]

_LOCAL_LAYOUT_05 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ############
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  #...#...#..#
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],  #.#.#.#.#.##
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  #.#...#....#
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],  #.###.####.#
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],  #...#....#.#
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],  ###.####.#.#
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #......#...#
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],  #.####...#.#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ############
]

_LOCAL_LAYOUT_06 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ###########
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  #..#...#..#
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1],  #..#.#.#.##
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  ##...#....#
    [1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],  #..#####.##
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],  #.##...#..#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],  #....#...##
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],  ####.###..#
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  #......##.#
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],  #.####....#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ###########
]

_LOCAL_LAYOUT_07 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ##########
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #....#...#
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  #.##.#.#.#
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #..#...#.#
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  ##.###.#.#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #....#...#
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],  #.##.###.#
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #..#...#.#
    [1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  ##.###.#.#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #....#...#
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],  #.##...#.#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #....#...#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ##########
]

_LOCAL_LAYOUT_08 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #############
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  #...#...#...#
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  #.#.#.#.#.#.#
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #.#...#...#.#
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  #.###.###.#.#
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  #...#...#...#
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  ###.###.###.#
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #.....#...#.#
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  #.###.###.#.#
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #.#.....#...#
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  #.#.###.###.#
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  #...#.......#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #############
]

_LOCAL_LAYOUT_09 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ############
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #....#...#.#
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],  #.##.#.#.#.#
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  #..#...#...#
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  ##.###.###.#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #....#...#.#
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  #.##.###.#.#
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  #..#...#...#
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  ##.###.###.#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #....#...#.#
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],  #.##...#...#
    [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  #..###.###.#
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  ##.........#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ############
]

_LOCAL_LAYOUT_10 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ############
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #..........#
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  #..#...#...#
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],  #..#...#.#.#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #....#...#.#
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #.#..#...#.#
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #.#....#...#
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #..........#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ############
]

_LOCAL_LAYOUT_11 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ##########
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #........#
    [1, 0, 1, 1, 1, 0, 0, 1, 0, 1],  #.###..#.#
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1],  #...#..#.#
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],  #.#.#.##.#
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],  #.#...#..#
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1],  #.###.#.##
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 1],  #...#...##
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #........#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ##########
]

_LOCAL_LAYOUT_12 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ###########
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #.........#
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  #.##.#.##.#
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],  #..#.#..#.#
    [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],  ##.#...#..#
    [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],  #...##.#.##
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  #.#.......#
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  #.###.###.#
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #...#...#.#
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  ###.###.#.#
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #.........#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  ###########
]

_LOCAL_LAYOUT_13 = [
    [1, 1, 1, 1, 1, 1, 1, 1],  ########
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 0, 1, 1, 1, 1, 1, 1],  #.######
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 1, 1, 1, 1, 1, 0, 1],  ######.#
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 0, 1, 1, 1, 1, 1, 1],  #.######
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 1, 1, 1, 1, 1, 0, 1],  ######.#
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 0, 1, 1, 1, 1, 1, 1],  #.######
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 1, 1, 1, 1, 1, 1, 1],  ########
]

_TEST_LAYOUT_01 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #############
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  #....#...#..#
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],  #.##.#.#.#.##
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],  #..#...#....#
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],  ##.###.####.#
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],  #....#...#..#
    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],  #.##...#...##
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #############
]

_TEST_LAYOUT_02 = [
    [1, 1, 1, 1, 1, 1, 1, 1],  ########
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 0, 1, 1, 1, 1, 0, 1],  #.####.#
    [1, 0, 0, 0, 0, 1, 0, 1],  #....#.#
    [1, 1, 1, 1, 0, 1, 0, 1],  ####.#.#
    [1, 0, 0, 0, 0, 1, 0, 1],  #....#.#
    [1, 0, 1, 1, 1, 1, 0, 1],  #.####.#
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 1, 1, 1, 1, 0, 1, 1],  #####.##
    [1, 0, 0, 0, 0, 0, 0, 1],  #......#
    [1, 0, 1, 1, 1, 1, 0, 1],  #.####.#
    [1, 1, 1, 1, 1, 1, 1, 1],  ########
]

_TEST_LAYOUT_03 = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #############
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  #...#...#...#
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  #.#.#.#.#.#.#
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #.#...#...#.#
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  #.###.###.#.#
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  #...#...#...#
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  ###.###.###.#
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],  #.....#...#.#
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],  #.###.###.#.#
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  #.#.....#...#
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],  #.#.###.###.#
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],  #...###.....#
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #############
]


def _build_local_variant(
    *,
    index: int | None = None,
    variant_name: str | None = None,
    env_name: str | None = None,
    maze_map: list[list[int]],
    structure_desc_en: str,
    structure_desc_zh: str,
    max_episode_steps: int | None = None,
) -> dict:
    if variant_name is None:
        if index is None:
            raise ValueError("Local variants require index or variant_name")
        variant_name = f"local-layout-{index:02d}"
    if env_name is None:
        env_name = (
            f"PointMaze Local Layout {index:02d}"
            if index is not None
            else f"PointMaze {variant_name}"
        )
    return {
        "varient_type": "local",
        "dataset_path": f"local_datasets/pointmaze-{variant_name}-v0",
        "env_paras": {
            "id": "PointMaze_UMaze-v3",
            "maze_map": maze_map,
            "reward_type": "sparse",
            "continuing_task": True,
            "reset_target": True,
            "max_episode_steps": (
                max_episode_steps
                if max_episode_steps is not None
                else _default_local_max_episode_steps(maze_map)
            ),
        },
        "prompt_vars": _build_prompt_vars(
            env_name=env_name,
            reward_type="sparse",
            maze_map=maze_map,
            structure_desc_en=structure_desc_en,
            structure_desc_zh=structure_desc_zh,
        ),
    }


POINTMAZE_VARIANTS = {
    "open": {
        "varient_type": "remote",
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
        "varient_type": "remote",
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
        "varient_type": "remote",
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
        "varient_type": "remote",
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
        "varient_type": "remote",
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
        "varient_type": "remote",
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
        "varient_type": "remote",
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
        "varient_type": "remote",
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
    "local-layout-01": _build_local_variant(
        index=1,
        maze_map=_LOCAL_LAYOUT_01,
        structure_desc_en="An 8x8 compact layout with a bent main corridor, short branches, and a lower-right passage.",
        structure_desc_zh="一个 8x8 的紧凑布局，包含弯折主通道、短分支和右下方通路。",
    ),
    "local-layout-02": _build_local_variant(
        index=2,
        maze_map=_LOCAL_LAYOUT_02,
        structure_desc_en="An 8x10 layout with two side corridors connected through staggered central gaps.",
        structure_desc_zh="一个 8x10 的布局，两侧走廊通过错开的中央缺口连接。",
    ),
    "local-layout-03": _build_local_variant(
        index=3,
        maze_map=_LOCAL_LAYOUT_03,
        structure_desc_en="A 9x9 layout with vertical barriers, a lower bypass, and several right-side turns.",
        structure_desc_zh="一个 9x9 的布局，包含纵向隔墙、底部绕行通道和右侧多次转弯。",
    ),
    "local-layout-04": _build_local_variant(
        index=4,
        maze_map=_LOCAL_LAYOUT_04,
        structure_desc_en="A 10x8 tall layout with a long descending route and a bottom horizontal corridor.",
        structure_desc_zh="一个 10x8 的纵向布局，包含向下延伸的长路线和底部横向走廊。",
    ),
    "local-layout-05": _build_local_variant(
        index=5,
        maze_map=_LOCAL_LAYOUT_05,
        structure_desc_en="A 10x12 layout with three corridor bands and multiple narrow connections between them.",
        structure_desc_zh="一个 10x12 的布局，包含三条走廊带和多个狭窄连接口。",
    ),
    "local-layout-06": _build_local_variant(
        index=6,
        maze_map=_LOCAL_LAYOUT_06,
        structure_desc_en="An 11x11 layout with room-like pockets, alternating bottlenecks, and diagonal progress.",
        structure_desc_zh="一个 11x11 的布局，包含房间状区域、交替瓶颈和斜向推进路线。",
    ),
    "local-layout-07": _build_local_variant(
        index=7,
        maze_map=_LOCAL_LAYOUT_07,
        structure_desc_en="A 13x10 layout with repeated vertical gates and a long multi-turn traversal.",
        structure_desc_zh="一个 13x10 的布局，包含重复的纵向门洞和长距离多转弯路径。",
    ),
    "local-layout-08": _build_local_variant(
        index=8,
        maze_map=_LOCAL_LAYOUT_08,
        structure_desc_en="A 13x13 layout with mirrored corridor sections, side channels, and bottom bypasses.",
        structure_desc_zh="一个 13x13 的布局，包含近似镜像的走廊段、侧向通道和底部绕行路线。",
    ),
    "local-layout-09": _build_local_variant(
        index=9,
        maze_map=_LOCAL_LAYOUT_09,
        structure_desc_en="A 14x12 layout with stacked gates, narrow vertical shafts, and a lower connecting corridor.",
        structure_desc_zh="一个 14x12 的布局，包含层叠门洞、狭窄纵向通道和底部连接走廊。",
    ),
    "local-layout-10": _build_local_variant(
        index=10,
        maze_map=_LOCAL_LAYOUT_10,
        structure_desc_en="A 9x12 wide layout with staggered vertical barriers and open upper and lower crossings.",
        structure_desc_zh="一个 9x12 的横向布局，包含错开的纵向隔墙以及上下两条开放连接通道。",
    ),
    "local-layout-11": _build_local_variant(
        index=11,
        maze_map=_LOCAL_LAYOUT_11,
        structure_desc_en="A 10x10 square layout with offset wall clusters, short branches, and perimeter bypasses.",
        structure_desc_zh="一个 10x10 的方形布局，包含错位墙簇、短分支和沿外围绕行的通道。",
    ),
    "local-layout-12": _build_local_variant(
        index=12,
        maze_map=_LOCAL_LAYOUT_12,
        structure_desc_en="A 12x11 layout with interlocking corridor sections, alternating gates, and a broad lower passage.",
        structure_desc_zh="一个 12x11 的布局，包含交错走廊段、交替门洞和较宽的底部通路。",
    ),
    "local-layout-13": _build_local_variant(
        index=13,
        maze_map=_LOCAL_LAYOUT_13,
        structure_desc_en="A 13x8 tall serpentine layout with alternating side openings and long horizontal corridors.",
        structure_desc_zh="一个 13x8 的纵向蛇形布局，包含交替侧向开口和多条长横向走廊。",
        max_episode_steps=900,
    ),
    "test-layout-01": _build_local_variant(
        variant_name="test-layout-01",
        env_name="PointMaze Test Layout 01",
        maze_map=_TEST_LAYOUT_01,
        structure_desc_en="An 8x13 wide test layout with staggered gates, branching corridors, and multiple horizontal crossings.",
        structure_desc_zh="一个 8x13 的横向测试布局，包含错位门洞、分支走廊和多条横向连接通道。",
    ),
    "test-layout-02": _build_local_variant(
        variant_name="test-layout-02",
        env_name="PointMaze Test Layout 02",
        maze_map=_TEST_LAYOUT_02,
        structure_desc_en="A 12x8 tall test layout with alternating horizontal barriers and two long side passages.",
        structure_desc_zh="一个 12x8 的纵向测试布局，包含交替横向隔墙和两条较长的侧边通道。",
    ),
    "test-layout-03": _build_local_variant(
        variant_name="test-layout-03",
        env_name="PointMaze Test Layout 03",
        maze_map=_TEST_LAYOUT_03,
        structure_desc_en="A 13x13 large test layout with repeated corridor modules, narrow gates, and lower cross-connections.",
        structure_desc_zh="一个 13x13 的大型测试布局，包含重复走廊模块、狭窄门洞和底部交叉连接。",
    ),
}


def get_pointmaze_variant_type(meta: dict) -> str:
    variant_type = meta.get("varient_type", meta.get("variant_type", "remote"))
    if variant_type not in {"remote", "local"}:
        raise ValueError(f"Unsupported PointMaze variant type: {variant_type!r}")
    return variant_type


def resolve_local_dataset_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path).expanduser()
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path
