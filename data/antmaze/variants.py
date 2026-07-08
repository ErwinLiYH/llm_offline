"""AntMaze variant registry: prompt copywriting over CrossMaze env facts.

Maze layouts, env ids/kwargs, dataset ids/paths, horizons, and the fixed
evaluation start/goal cells live in `crossmaze.variants`. Evaluation uses the
same maps the offline data was collected on; start/goal are recorded as
`eval_reset_cell` / `eval_goal_cell` coordinates (passed to the env through
`reset(options=...)`) instead of "r"/"g" map markers.
"""

from pathlib import Path

from crossmaze.layout import format_visual_map, maze_shape_text
from crossmaze.variants import ANTMAZE_ENV_FACTS


def _build_prompt_vars(
    *,
    env_name: str,
    dataset_style: str,
    maze_map: list[list[int]],
    structure_desc_en: str,
) -> dict:
    return {
        "env_name": env_name,
        "dataset_style": dataset_style,
        "maze_map": maze_map,
        "maze_size_scaling": 4.0,
        "maze_shape": maze_shape_text(maze_map),
        "maze_visual": format_visual_map(maze_map),
        "structure_desc_en": structure_desc_en,
    }


def _build_remote_variant(
    variant_name: str,
    *,
    env_name: str,
    dataset_style: str,
    structure_desc_en: str,
) -> dict:
    facts = ANTMAZE_ENV_FACTS[variant_name]
    return {
        "dataset_id": facts["dataset_id"],
        "env_id": facts["env_id"],
        "env_kwargs": dict(facts["env_kwargs"]),
        "eval_reset_cell": list(facts["eval_reset_cell"]),
        "eval_goal_cell": list(facts["eval_goal_cell"]),
        "prompt_vars": _build_prompt_vars(
            env_name=env_name,
            dataset_style=dataset_style,
            maze_map=facts["maze_map"],
            structure_desc_en=structure_desc_en,
        ),
    }


def _build_local_variant(
    *,
    index: int | None = None,
    variant_name: str | None = None,
    env_name: str | None = None,
    structure_desc_en: str,
    dataset_style: str = "local reset and goal trajectories",
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
    facts = ANTMAZE_ENV_FACTS[variant_name]
    return {
        "varient_type": "local",
        "dataset_path": facts["dataset_path"],
        "env_paras": dict(facts["env_paras"]),
        "eval_reset_cell": list(facts["eval_reset_cell"]),
        "eval_goal_cell": list(facts["eval_goal_cell"]),
        "prompt_vars": _build_prompt_vars(
            env_name=env_name,
            dataset_style=dataset_style,
            maze_map=facts["maze_map"],
            structure_desc_en=structure_desc_en,
        ),
    }


ANTMAZE_VARIANTS = {
    "umaze": _build_remote_variant(
        "umaze",
        env_name="AntMaze UMaze",
        dataset_style="fixed reset and goal locations",
        structure_desc_en="A compact U-shaped maze with one long route around a central wall.",
    ),
    "umaze-diverse": _build_remote_variant(
        "umaze-diverse",
        env_name="AntMaze UMaze Diverse",
        dataset_style="diverse offline trajectories in the U-shaped maze",
        structure_desc_en="A compact U-shaped maze with one long route around a central wall.",
    ),
    "medium-play": _build_remote_variant(
        "medium-play",
        env_name="AntMaze Medium Play",
        dataset_style="play-style trajectories with varied starts and goals",
        structure_desc_en="A medium maze with several corridors, turns, and dead ends.",
    ),
    "medium-diverse": _build_remote_variant(
        "medium-diverse",
        env_name="AntMaze Medium Diverse",
        dataset_style="diverse-goal-and-reset trajectories",
        structure_desc_en="A medium maze with several corridors, turns, and dead ends.",
    ),
    "large-play": _build_remote_variant(
        "large-play",
        env_name="AntMaze Large Play",
        dataset_style="play-style trajectories with varied starts and goals",
        structure_desc_en="A large maze with long corridors, branches, and narrow bottlenecks.",
    ),
    "large-diverse": _build_remote_variant(
        "large-diverse",
        env_name="AntMaze Large Diverse",
        dataset_style="diverse-goal-and-reset trajectories",
        structure_desc_en="A large maze with long corridors, branches, and narrow bottlenecks.",
    ),
    "local-layout-01": _build_local_variant(
        index=1,
        structure_desc_en="A compact generated AntMaze layout in the 45 static-difficulty band, with open corridor choices and light bottlenecks.",
    ),
    "local-layout-02": _build_local_variant(
        index=2,
        structure_desc_en="A compact generated AntMaze layout in the 45 static-difficulty band, with open corridor choices and light bottlenecks.",
    ),
    "local-layout-03": _build_local_variant(
        index=3,
        structure_desc_en="A compact generated AntMaze layout in the 45 static-difficulty band, with open corridor choices and light bottlenecks.",
    ),
    "local-layout-04": _build_local_variant(
        index=4,
        structure_desc_en="A compact generated AntMaze layout in the 45 static-difficulty band, with open corridor choices and light bottlenecks.",
    ),
    "local-layout-05": _build_local_variant(
        index=5,
        structure_desc_en="A compact generated AntMaze layout in the 45 static-difficulty band, with open corridor choices and light bottlenecks.",
    ),
    "local-layout-06": _build_local_variant(
        index=6,
        structure_desc_en="A compact generated AntMaze layout in the 50 static-difficulty band, with longer turns and moderate bottlenecks.",
    ),
    "local-layout-07": _build_local_variant(
        index=7,
        structure_desc_en="A compact generated AntMaze layout in the 50 static-difficulty band, with longer turns and moderate bottlenecks.",
    ),
    "local-layout-08": _build_local_variant(
        index=8,
        structure_desc_en="A compact generated AntMaze layout in the 50 static-difficulty band, with longer turns and moderate bottlenecks.",
    ),
    "local-layout-09": _build_local_variant(
        index=9,
        structure_desc_en="A compact generated AntMaze layout in the 45 static-difficulty band, with open corridor choices and light bottlenecks.",
    ),
    "test-layout-01": _build_local_variant(
        variant_name="test-layout-01",
        env_name="AntMaze Test Layout 01",
        structure_desc_en="A held-out compact generated AntMaze layout in the 45 static-difficulty band.",
        dataset_style="held-out local reset and goal trajectories",
    ),
    "test-layout-02": _build_local_variant(
        variant_name="test-layout-02",
        env_name="AntMaze Test Layout 02",
        structure_desc_en="A held-out compact generated AntMaze layout in the 45 static-difficulty band.",
        dataset_style="held-out local reset and goal trajectories",
    ),
    "test-layout-03": _build_local_variant(
        variant_name="test-layout-03",
        env_name="AntMaze Test Layout 03",
        structure_desc_en="A held-out compact generated AntMaze layout in the 50 static-difficulty band.",
        dataset_style="held-out local reset and goal trajectories",
    ),
    "test-layout-04": _build_local_variant(
        variant_name="test-layout-04",
        env_name="AntMaze Test Layout 04",
        structure_desc_en="A held-out compact generated AntMaze layout in the 50 static-difficulty band.",
        dataset_style="held-out local reset and goal trajectories",
    ),
    "ultra": _build_local_variant(
        variant_name="ultra",
        env_name="AntMaze Ultra",
        structure_desc_en="An experimental AntMaze-Ultra layout from Farama D4RL PR #220, larger than the official large map with long corridors and sparse bottlenecks.",
        dataset_style="experimental local AntMaze-Ultra trajectories",
    ),
}


def get_antmaze_variant_type(meta: dict) -> str:
    variant_type = meta.get("varient_type", meta.get("variant_type", "remote"))
    if variant_type not in {"remote", "local"}:
        raise ValueError(f"Unsupported AntMaze variant type: {variant_type!r}")
    return variant_type


def resolve_local_dataset_path(
    dataset_path: str | Path,
    local_dataset_root: str | Path | None = None,
) -> Path:
    default_path = Path(dataset_path).expanduser()
    if local_dataset_root is None:
        path = default_path
    else:
        root_path = Path(local_dataset_root).expanduser()
        path = (
            root_path
            if root_path.name == default_path.name
            else root_path / default_path.name
        )
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path
