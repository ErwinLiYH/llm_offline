from pathlib import Path


R = "r"
G = "g"


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


def _maze_from_strings(rows: list[str]) -> list[list[int]]:
    if not rows:
        raise ValueError("maze rows must be non-empty")
    width = len(rows[0])
    if width == 0:
        raise ValueError("maze rows must be non-empty")
    if any(len(row) != width for row in rows):
        raise ValueError(f"maze rows must have equal width: {rows}")
    return [[1 if cell == "#" else 0 for cell in row] for row in rows]


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

_LOCAL_LAYOUT_01 = _maze_from_strings([
    "#############",
    "#.....#...#.#",
    "#.#.#.###.#.#",
    "#.#.....#...#",
    "#.#.#.#.###.#",
    "#...........#",
    "#.#######.###",
    "#.#.....#...#",
    "#.#.###.###.#",
    "#.....#.....#",
    "#############",
])

_LOCAL_LAYOUT_02 = _maze_from_strings([
    "##############",
    "#.#..........#",
    "#.#.#####.#..#",
    "#...#.#......#",
    "#.#.#.#.#.#..#",
    "#...#........#",
    "#.#.#####.#..#",
    "#.......#....#",
    "#.#.###.####.#",
    "#...#........#",
    "##############",
])

_LOCAL_LAYOUT_03 = _maze_from_strings([
    "#############",
    "#...........#",
    "#.#.###.#.#.#",
    "#.#.#.....#.#",
    "###.#.###.#.#",
    "#...#...#...#",
    "#.###.#.###.#",
    "#...#...#...#",
    "#.#.#.###.###",
    "#.#.........#",
    "#############",
])

_LOCAL_LAYOUT_04 = _maze_from_strings([
    "#############",
    "#.........#.#",
    "#.#.#####.#.#",
    "#.....#.....#",
    "#.###.#.#.#.#",
    "#...#...#...#",
    "###.###.#.#.#",
    "#...........#",
    "#########.#.#",
    "#...........#",
    "#############",
])

_LOCAL_LAYOUT_05 = _maze_from_strings([
    "##############",
    "#.........#..#",
    "#.#######.##.#",
    "#.#.....#....#",
    "#.#.###.###..#",
    "#...#.#...#..#",
    "#.###.#.###..#",
    "#.....#......#",
    "##############",
])

_LOCAL_LAYOUT_06 = _maze_from_strings([
    "################",
    "#.........#.#..#",
    "#########.#.##.#",
    "#.......#.#....#",
    "#.#####.#.####.#",
    "#...#.#.#...#..#",
    "#.#.#.#.#.###.##",
    "#.#.#.....#....#",
    "#.#.#######.#..#",
    "#.#.........#..#",
    "#.#######.#.##.#",
    "#.#.........#..#",
    "#.#.#.#.#.#.#..#",
    "#.......#......#",
    "################",
])

_LOCAL_LAYOUT_07 = _maze_from_strings([
    "################",
    "#.......#...#..#",
    "#.#.#####.#.##.#",
    "#.#.#...#.#....#",
    "#.###.#.#.###..#",
    "#.#...#...#....#",
    "#.#.#######.##.#",
    "#.#.....#...#..#",
    "#.#####.#####..#",
    "#.....#...#....#",
    "#.#.#.###.#.##.#",
    "#.......#...#..#",
    "#.#.###.#####.##",
    "#.....#........#",
    "################",
])

_LOCAL_LAYOUT_08 = _maze_from_strings([
    "################",
    "#.....#........#",
    "###.#.#######.##",
    "#...#.......#..#",
    "#.#########.#..#",
    "#...#...#.#.#..#",
    "###.#.#.#.#.##.#",
    "#...#.#.#.#.#..#",
    "#.###.#.#.#.#..#",
    "#.#...#.#......#",
    "#.#.###.######.#",
    "#.#...#...#....#",
    "#.###.###.#.#..#",
    "#.......#...#..#",
    "################",
])

_LOCAL_LAYOUT_09 = _maze_from_strings([
    "################",
    "#.#............#",
    "#.#.#########..#",
    "#.........#....#",
    "#########.#.##.#",
    "#.......#.#....#",
    "#.#####.#.###..#",
    "#.#...#.#.#....#",
    "#.###.#.###.##.#",
    "#...#.#.....#..#",
    "#.#.#.#.#####..#",
    "#.#.#.....#....#",
    "###.#.###.#.####",
    "#...#...#......#",
    "################",
])

_TEST_LAYOUT_01 = _maze_from_strings([
    "##############",
    "#.........#..#",
    "#.#.#.###.#..#",
    "#.#.#........#",
    "#.#.###.#.#..#",
    "#...#.#......#",
    "#.###.###.##.#",
    "#.#..........#",
    "#.#.#.#.#.####",
    "#.#..........#",
    "##############",
])

_TEST_LAYOUT_02 = _maze_from_strings([
    "##############",
    "#.....#......#",
    "#.###.#.####.#",
    "#.#.#...#....#",
    "#.#.#.#####..#",
    "#.#...#......#",
    "#.#.#.#.####.#",
    "#...#...#....#",
    "##############",
])

_TEST_LAYOUT_03 = _maze_from_strings([
    "###############",
    "#.......#.#.#.#",
    "#.#.###.#.#.#.#",
    "#...#.#...#...#",
    "#.#.#.#.#####.#",
    "#.#.....#...#.#",
    "#.#.#.#.#.#.#.#",
    "#.#.#...#.#...#",
    "#.#.###.#.#####",
    "#.#...#.#.....#",
    "#.###.#######.#",
    "#.#...........#",
    "#.#.###.#.#.###",
    "#.#.....#.....#",
    "###############",
])

_TEST_LAYOUT_04 = _maze_from_strings([
    "###############",
    "#.#.........#.#",
    "#.#####.###.#.#",
    "#.........#...#",
    "#.#.#.###.#.#.#",
    "#.....#.......#",
    "#.#.#.#.###.###",
    "#.#...#.#...#.#",
    "#.###.#.#.#.#.#",
    "#.#...#...#...#",
    "#.#.#########.#",
    "#...#.........#",
    "###############",
])

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


def _default_local_max_episode_steps(maze_map: list[list[int]]) -> int:
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    return max(1000, rows * cols * 8)


def _clean_maze_map(maze_map: list[list[object]]) -> list[list[int]]:
    return [[1 if cell == 1 else 0 for cell in row] for row in maze_map]


def _mark_cells(
    maze_map: list[list[object]],
    markers: list[tuple[int, int, str]],
) -> list[list[object]]:
    marked = [list(row) for row in maze_map]
    for row, col, marker in markers:
        if row < 0 or row >= len(marked) or col < 0 or col >= len(marked[row]):
            raise ValueError(f"Marker cell {(row, col)} is outside the maze")
        if marked[row][col] == 1:
            raise ValueError(f"Marker cell {(row, col)} is a wall")
        marked[row][col] = marker
    return marked


def _build_local_variant(
    *,
    index: int | None = None,
    variant_name: str | None = None,
    maze_map: list[list[int]],
    eval_reset_cell: tuple[int, int],
    eval_goal_cell: tuple[int, int],
    structure_desc_en: str,
    dataset_style: str = "local reset and goal trajectories",
    max_episode_steps: int | None = None,
    env_name: str | None = None,
) -> dict:
    if variant_name is None:
        if index is None:
            raise ValueError("Local AntMaze variants require index or variant_name")
        variant_name = f"local-layout-{index:02d}"
    if env_name is None:
        if index is not None and variant_name.startswith("local-layout-"):
            env_name = f"AntMaze Local Layout {index:02d}"
        else:
            env_name = f"AntMaze {variant_name}"
    clean_map = _clean_maze_map(maze_map)
    episode_steps = (
        int(max_episode_steps)
        if max_episode_steps is not None
        else _default_local_max_episode_steps(clean_map)
    )
    eval_maze_map = _mark_cells(
        clean_map,
        [
            (int(eval_reset_cell[0]), int(eval_reset_cell[1]), R),
            (int(eval_goal_cell[0]), int(eval_goal_cell[1]), G),
        ],
    )
    return {
        "varient_type": "local",
        "dataset_path": f"local_datasets/antmaze-{variant_name}-v0",
        "collection_env_paras": {
            "id": "AntMaze_UMaze-v4",
            "maze_map": clean_map,
            "reward_type": "sparse",
            "continuing_task": True,
            "reset_target": False,
            "max_episode_steps": episode_steps,
        },
        "env_paras": {
            "id": "AntMaze_UMaze-v4",
            "maze_map": eval_maze_map,
            "reward_type": "sparse",
            "continuing_task": True,
            "reset_target": False,
            "max_episode_steps": episode_steps,
        },
        "prompt_vars": _build_prompt_vars(
            env_name=env_name,
            dataset_style=dataset_style,
            maze_map=clean_map,
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
    "local-layout-01": _build_local_variant(
        index=1,
        maze_map=_LOCAL_LAYOUT_01,
        eval_reset_cell=(9, 5),
        eval_goal_cell=(1, 7),
        structure_desc_en="A design-generated large-like maze with distributed corridors, moderate loops, and several short branches.",
    ),
    "local-layout-02": _build_local_variant(
        index=2,
        maze_map=_LOCAL_LAYOUT_02,
        eval_reset_cell=(9, 1),
        eval_goal_cell=(3, 5),
        structure_desc_en="A design-generated large-like maze with broad corridor coverage, staggered gates, and alternate side routes.",
    ),
    "local-layout-03": _build_local_variant(
        index=3,
        maze_map=_LOCAL_LAYOUT_03,
        eval_reset_cell=(3, 1),
        eval_goal_cell=(9, 11),
        structure_desc_en="A design-generated large-like maze with offset barriers, looped bypasses, and compact dead-end pockets.",
    ),
    "local-layout-04": _build_local_variant(
        index=4,
        maze_map=_LOCAL_LAYOUT_04,
        eval_reset_cell=(9, 1),
        eval_goal_cell=(1, 1),
        structure_desc_en="A design-generated large-like maze with a long vertical transfer, distributed openings, and a lower bypass.",
    ),
    "local-layout-05": _build_local_variant(
        index=5,
        maze_map=_LOCAL_LAYOUT_05,
        eval_reset_cell=(1, 11),
        eval_goal_cell=(5, 5),
        structure_desc_en="A design-generated large-like maze with a wide footprint, gated center, and multiple turn-heavy routes.",
    ),
    "local-layout-06": _build_local_variant(
        index=6,
        maze_map=_LOCAL_LAYOUT_06,
        eval_reset_cell=(1, 11),
        eval_goal_cell=(1, 1),
        structure_desc_en="A design-generated harder maze with a larger footprint, narrow bottlenecks, and several dead-end branches.",
        max_episode_steps=1500,
    ),
    "local-layout-07": _build_local_variant(
        index=7,
        maze_map=_LOCAL_LAYOUT_07,
        eval_reset_cell=(3, 5),
        eval_goal_cell=(1, 7),
        structure_desc_en="A design-generated harder maze with stacked corridor bands, long detours, and repeated bottleneck openings.",
        max_episode_steps=1600,
    ),
    "local-layout-08": _build_local_variant(
        index=8,
        maze_map=_LOCAL_LAYOUT_08,
        eval_reset_cell=(13, 1),
        eval_goal_cell=(1, 7),
        structure_desc_en="A design-generated harder maze with repeated vertical gates, deep routing, and a long final approach.",
        max_episode_steps=1800,
    ),
    "local-layout-09": _build_local_variant(
        index=9,
        maze_map=_LOCAL_LAYOUT_09,
        eval_reset_cell=(13, 1),
        eval_goal_cell=(7, 9),
        structure_desc_en="A design-generated harder maze with multi-stage corridors, tight central turns, and extended lower routing.",
        max_episode_steps=1900,
    ),
    "test-layout-01": _build_local_variant(
        variant_name="test-layout-01",
        env_name="AntMaze Test Layout 01",
        maze_map=_TEST_LAYOUT_01,
        eval_reset_cell=(9, 1),
        eval_goal_cell=(9, 3),
        structure_desc_en="A held-out design-generated large-like maze with distributed corridors, central barriers, and looped bypasses.",
        dataset_style="held-out local reset and goal trajectories",
    ),
    "test-layout-02": _build_local_variant(
        variant_name="test-layout-02",
        env_name="AntMaze Test Layout 02",
        maze_map=_TEST_LAYOUT_02,
        eval_reset_cell=(3, 1),
        eval_goal_cell=(7, 9),
        structure_desc_en="A held-out design-generated large-like maze with staggered gates, compact pockets, and a long lower route.",
        dataset_style="held-out local reset and goal trajectories",
    ),
    "test-layout-03": _build_local_variant(
        variant_name="test-layout-03",
        env_name="AntMaze Test Layout 03",
        maze_map=_TEST_LAYOUT_03,
        eval_reset_cell=(13, 1),
        eval_goal_cell=(1, 11),
        structure_desc_en="A held-out design-generated harder maze with one-cell-wide corridors, repeated bottlenecks, and long detours.",
        dataset_style="held-out local reset and goal trajectories",
        max_episode_steps=1700,
    ),
    "test-layout-04": _build_local_variant(
        variant_name="test-layout-04",
        env_name="AntMaze Test Layout 04",
        maze_map=_TEST_LAYOUT_04,
        eval_reset_cell=(11, 5),
        eval_goal_cell=(11, 1),
        structure_desc_en="A held-out design-generated harder maze with narrow corridors, long routing, and dense internal bottlenecks.",
        dataset_style="held-out local reset and goal trajectories",
        max_episode_steps=2100,
    ),
}


def get_antmaze_variant_type(meta: dict) -> str:
    variant_type = meta.get("varient_type", meta.get("variant_type", "remote"))
    if variant_type not in {"remote", "local"}:
        raise ValueError(f"Unsupported AntMaze variant type: {variant_type!r}")
    return variant_type


def resolve_local_dataset_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path).expanduser()
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path
