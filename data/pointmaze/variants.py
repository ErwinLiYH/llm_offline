"""PointMaze variant registry: prompt copywriting over CrossMaze env facts.

Maze layouts, env ids/kwargs, dataset ids/paths, and episode horizons live in
`crossmaze.variants`; this module only adds the prompt-side copywriting
(env names, reward/structure descriptions) and assembles the variant dicts
consumed by the dataset/eval pipeline. The assembled dict contents must stay
byte-identical to the historical literals so cache signatures do not shift.
"""

from pathlib import Path

from crossmaze.layout import format_raw_matrix, format_visual_map, maze_shape_text
from crossmaze.reward import reward_typed_dataset_path
from crossmaze.variants import POINTMAZE_ENV_FACTS


def _build_prompt_vars(
    *,
    env_name: str,
    reward_type: str,
    maze_map: list[list[int]],
    structure_desc_en: str,
    structure_desc_zh: str,
) -> dict:
    return {
        "reward_type": reward_type,
        "maze_map": maze_map,
        "maze_size_scaling": 1.0,
        "env_name": env_name,
        "reward_desc_en": f"{reward_type} reward",
        "reward_desc_zh": "稀疏奖励" if reward_type == "sparse" else "稠密奖励",
        "maze_shape": maze_shape_text(maze_map),
        "maze_raw_matrix": format_raw_matrix(maze_map),
        "maze_visual": format_visual_map(maze_map),
        "structure_desc_en": structure_desc_en,
        "structure_desc_zh": structure_desc_zh,
    }


def _build_remote_variant(
    variant_name: str,
    *,
    env_name: str,
    structure_desc_en: str,
    structure_desc_zh: str,
) -> dict:
    facts = POINTMAZE_ENV_FACTS[variant_name]
    return {
        "varient_type": "remote",
        "dataset_id": facts["dataset_id"],
        "env_id": facts["env_id"],
        "prompt_vars": _build_prompt_vars(
            env_name=env_name,
            reward_type=facts["reward_type"],
            maze_map=facts["maze_map"],
            structure_desc_en=structure_desc_en,
            structure_desc_zh=structure_desc_zh,
        ),
    }


def _build_local_variant(
    *,
    index: int | None = None,
    variant_name: str | None = None,
    env_name: str | None = None,
    structure_desc_en: str,
    structure_desc_zh: str,
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
    facts = POINTMAZE_ENV_FACTS[variant_name]
    return {
        "varient_type": "local",
        "dataset_path": facts["dataset_path"],
        "env_paras": dict(facts["env_paras"]),
        "prompt_vars": _build_prompt_vars(
            env_name=env_name,
            reward_type=facts["reward_type"],
            maze_map=facts["maze_map"],
            structure_desc_en=structure_desc_en,
            structure_desc_zh=structure_desc_zh,
        ),
    }


_POINTMAZE_V2_DESCS = {
    "55-60": (
        "A generated V2 PointMaze layout in the 55-60 static-difficulty band, with moderate branches and bottlenecks.",
        "一个生成的 V2 PointMaze 布局，静态难度位于 55-60 档，包含中等数量的分支和瓶颈。",
    ),
    "60-65": (
        "A generated V2 PointMaze layout in the 60-65 static-difficulty band, with longer routes and tighter corridor choices.",
        "一个生成的 V2 PointMaze 布局，静态难度位于 60-65 档，包含更长路线和更紧凑的走廊选择。",
    ),
    "65-70": (
        "A generated V2 PointMaze layout in the 65-70 static-difficulty band, with long routes, dead ends, and repeated bottlenecks.",
        "一个生成的 V2 PointMaze 布局，静态难度位于 65-70 档，包含长路线、死路和重复瓶颈。",
    ),
}

_POINTMAZE_EXTURN_DESC = (
    "An extreme-turn PointMaze layout made of diagonal staircase corridors, "
    "where nearly every step along a shortest path requires a 90-degree turn.",
    "一个极高转弯率的 PointMaze 布局，由对角阶梯走廊构成，最短路径上几乎每一步都需要 90 度转弯。",
)

_POINTMAZE_EXTURN_SIMPLE_DESC = (
    "A minimal umaze-scale PointMaze layout containing a single staircase "
    "corridor with consecutive 90-degree turns and a short straight tail.",
    "一个 umaze 级别的极简 PointMaze 布局，只包含一段需要连续 90 度转弯的阶梯走廊和一小段直走廊。",
)


POINTMAZE_VARIANTS = {
    "open": _build_remote_variant(
        "open",
        env_name="PointMaze Open",
        structure_desc_en="A 5x7 open arena with walls on the outer boundary and no internal obstacles.",
        structure_desc_zh="一个 5x7 的开放迷宫，只有外边界墙壁，内部没有障碍。",
    ),
    "open-dense": _build_remote_variant(
        "open-dense",
        env_name="PointMaze OpenDense",
        structure_desc_en="A 5x7 open arena with walls on the outer boundary and no internal obstacles.",
        structure_desc_zh="一个 5x7 的开放迷宫，只有外边界墙壁，内部没有障碍。",
    ),
    "umaze": _build_remote_variant(
        "umaze",
        env_name="PointMaze UMaze",
        structure_desc_en="A compact 5x5 U-shaped maze with a single narrow corridor wrapping around the central wall block.",
        structure_desc_zh="一个紧凑的 5x5 U 形迷宫，需要沿着中央墙块外侧的狭窄通道绕行。",
    ),
    "umaze-dense": _build_remote_variant(
        "umaze-dense",
        env_name="PointMaze UMazeDense",
        structure_desc_en="A compact 5x5 U-shaped maze with a single narrow corridor wrapping around the central wall block.",
        structure_desc_zh="一个紧凑的 5x5 U 形迷宫，需要沿着中央墙块外侧的狭窄通道绕行。",
    ),
    "medium": _build_remote_variant(
        "medium",
        env_name="PointMaze Medium",
        structure_desc_en="An 8x8 maze with multiple corridors, dead ends, and several turns between start regions and goals.",
        structure_desc_zh="一个 8x8 的中型迷宫，包含多条走廊、死路和若干转弯，需要绕开内部墙体前往目标。",
    ),
    "medium-dense": _build_remote_variant(
        "medium-dense",
        env_name="PointMaze MediumDense",
        structure_desc_en="An 8x8 maze with multiple corridors, dead ends, and several turns between start regions and goals.",
        structure_desc_zh="一个 8x8 的中型迷宫，包含多条走廊、死路和若干转弯，需要绕开内部墙体前往目标。",
    ),
    "large": _build_remote_variant(
        "large",
        env_name="PointMaze Large",
        structure_desc_en="A large maze with long corridors, many branches, and several bottlenecks created by dense internal walls.",
        structure_desc_zh="一个大型迷宫，包含长走廊、多个分支以及由密集内墙形成的若干瓶颈通道。",
    ),
    "large-dense": _build_remote_variant(
        "large-dense",
        env_name="PointMaze LargeDense",
        structure_desc_en="A large maze with long corridors, many branches, and several bottlenecks created by dense internal walls.",
        structure_desc_zh="一个大型迷宫，包含长走廊、多个分支以及由密集内墙形成的若干瓶颈通道。",
    ),
    "local-medium": _build_local_variant(
        variant_name="local-medium",
        env_name="PointMaze Local Medium",
        structure_desc_en="A locally generated dataset on the official 8x8 PointMaze medium map.",
        structure_desc_zh="一个基于官方 8x8 PointMaze medium 地图本地生成的数据集。",
    ),
    "local-layout-01": _build_local_variant(
        index=1,
        structure_desc_en="An 8x8 compact layout with a bent main corridor, short branches, and a lower-right passage.",
        structure_desc_zh="一个 8x8 的紧凑布局，包含弯折主通道、短分支和右下方通路。",
    ),
    "local-layout-02": _build_local_variant(
        index=2,
        structure_desc_en="An 8x10 layout with two side corridors connected through staggered central gaps.",
        structure_desc_zh="一个 8x10 的布局，两侧走廊通过错开的中央缺口连接。",
    ),
    "local-layout-03": _build_local_variant(
        index=3,
        structure_desc_en="A 9x9 layout with vertical barriers, a lower bypass, and several right-side turns.",
        structure_desc_zh="一个 9x9 的布局，包含纵向隔墙、底部绕行通道和右侧多次转弯。",
    ),
    "local-layout-04": _build_local_variant(
        index=4,
        structure_desc_en="A 10x8 tall layout with a long descending route and a bottom horizontal corridor.",
        structure_desc_zh="一个 10x8 的纵向布局，包含向下延伸的长路线和底部横向走廊。",
    ),
    "local-layout-05": _build_local_variant(
        index=5,
        structure_desc_en="A 10x12 layout with three corridor bands and multiple narrow connections between them.",
        structure_desc_zh="一个 10x12 的布局，包含三条走廊带和多个狭窄连接口。",
    ),
    "local-layout-06": _build_local_variant(
        index=6,
        structure_desc_en="An 11x11 layout with room-like pockets, alternating bottlenecks, and diagonal progress.",
        structure_desc_zh="一个 11x11 的布局，包含房间状区域、交替瓶颈和斜向推进路线。",
    ),
    "local-layout-07": _build_local_variant(
        index=7,
        structure_desc_en="A 13x10 layout with repeated vertical gates and a long multi-turn traversal.",
        structure_desc_zh="一个 13x10 的布局，包含重复的纵向门洞和长距离多转弯路径。",
    ),
    "local-layout-08": _build_local_variant(
        index=8,
        structure_desc_en="A 13x13 layout with mirrored corridor sections, side channels, and bottom bypasses.",
        structure_desc_zh="一个 13x13 的布局，包含近似镜像的走廊段、侧向通道和底部绕行路线。",
    ),
    "local-layout-09": _build_local_variant(
        index=9,
        structure_desc_en="A 14x12 layout with stacked gates, narrow vertical shafts, and a lower connecting corridor.",
        structure_desc_zh="一个 14x12 的布局，包含层叠门洞、狭窄纵向通道和底部连接走廊。",
    ),
    "local-layout-10": _build_local_variant(
        index=10,
        structure_desc_en="A 9x12 wide layout with staggered vertical barriers and open upper and lower crossings.",
        structure_desc_zh="一个 9x12 的横向布局，包含错开的纵向隔墙以及上下两条开放连接通道。",
    ),
    "local-layout-11": _build_local_variant(
        index=11,
        structure_desc_en="A 10x10 square layout with offset wall clusters, short branches, and perimeter bypasses.",
        structure_desc_zh="一个 10x10 的方形布局，包含错位墙簇、短分支和沿外围绕行的通道。",
    ),
    "local-layout-12": _build_local_variant(
        index=12,
        structure_desc_en="A 12x11 layout with interlocking corridor sections, alternating gates, and a broad lower passage.",
        structure_desc_zh="一个 12x11 的布局，包含交错走廊段、交替门洞和较宽的底部通路。",
    ),
    "local-layout-13": _build_local_variant(
        index=13,
        structure_desc_en="A 13x8 tall serpentine layout with alternating side openings and long horizontal corridors.",
        structure_desc_zh="一个 13x8 的纵向蛇形布局，包含交替侧向开口和多条长横向走廊。",
    ),
    "test-layout-01": _build_local_variant(
        variant_name="test-layout-01",
        env_name="PointMaze Test Layout 01",
        structure_desc_en="An 8x13 wide test layout with staggered gates, branching corridors, and multiple horizontal crossings.",
        structure_desc_zh="一个 8x13 的横向测试布局，包含错位门洞、分支走廊和多条横向连接通道。",
    ),
    "test-layout-02": _build_local_variant(
        variant_name="test-layout-02",
        env_name="PointMaze Test Layout 02",
        structure_desc_en="A 12x8 tall test layout with alternating horizontal barriers and two long side passages.",
        structure_desc_zh="一个 12x8 的纵向测试布局，包含交替横向隔墙和两条较长的侧边通道。",
    ),
    "test-layout-03": _build_local_variant(
        variant_name="test-layout-03",
        env_name="PointMaze Test Layout 03",
        structure_desc_en="A 13x13 large test layout with repeated corridor modules, narrow gates, and lower cross-connections.",
        structure_desc_zh="一个 13x13 的大型测试布局，包含重复走廊模块、狭窄门洞和底部交叉连接。",
    ),
    "local-layoutV2-01": _build_local_variant(
        variant_name="local-layoutV2-01",
        env_name="PointMaze Local Layout V2 01",
        structure_desc_en=_POINTMAZE_V2_DESCS["55-60"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["55-60"][1],
    ),
    "local-layoutV2-02": _build_local_variant(
        variant_name="local-layoutV2-02",
        env_name="PointMaze Local Layout V2 02",
        structure_desc_en=_POINTMAZE_V2_DESCS["55-60"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["55-60"][1],
    ),
    "local-layoutV2-03": _build_local_variant(
        variant_name="local-layoutV2-03",
        env_name="PointMaze Local Layout V2 03",
        structure_desc_en=_POINTMAZE_V2_DESCS["55-60"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["55-60"][1],
    ),
    "local-layoutV2-04": _build_local_variant(
        variant_name="local-layoutV2-04",
        env_name="PointMaze Local Layout V2 04",
        structure_desc_en=_POINTMAZE_V2_DESCS["55-60"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["55-60"][1],
    ),
    "local-layoutV2-05": _build_local_variant(
        variant_name="local-layoutV2-05",
        env_name="PointMaze Local Layout V2 05",
        structure_desc_en=_POINTMAZE_V2_DESCS["60-65"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["60-65"][1],
    ),
    "local-layoutV2-06": _build_local_variant(
        variant_name="local-layoutV2-06",
        env_name="PointMaze Local Layout V2 06",
        structure_desc_en=_POINTMAZE_V2_DESCS["60-65"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["60-65"][1],
    ),
    "local-layoutV2-07": _build_local_variant(
        variant_name="local-layoutV2-07",
        env_name="PointMaze Local Layout V2 07",
        structure_desc_en=_POINTMAZE_V2_DESCS["60-65"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["60-65"][1],
    ),
    "local-layoutV2-08": _build_local_variant(
        variant_name="local-layoutV2-08",
        env_name="PointMaze Local Layout V2 08",
        structure_desc_en=_POINTMAZE_V2_DESCS["60-65"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["60-65"][1],
    ),
    "local-layoutV2-09": _build_local_variant(
        variant_name="local-layoutV2-09",
        env_name="PointMaze Local Layout V2 09",
        structure_desc_en=_POINTMAZE_V2_DESCS["65-70"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["65-70"][1],
    ),
    "local-layoutV2-10": _build_local_variant(
        variant_name="local-layoutV2-10",
        env_name="PointMaze Local Layout V2 10",
        structure_desc_en=_POINTMAZE_V2_DESCS["65-70"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["65-70"][1],
    ),
    "local-layoutV2-11": _build_local_variant(
        variant_name="local-layoutV2-11",
        env_name="PointMaze Local Layout V2 11",
        structure_desc_en=_POINTMAZE_V2_DESCS["65-70"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["65-70"][1],
    ),
    "local-layoutV2-12": _build_local_variant(
        variant_name="local-layoutV2-12",
        env_name="PointMaze Local Layout V2 12",
        structure_desc_en=_POINTMAZE_V2_DESCS["65-70"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["65-70"][1],
    ),
    "test-layoutV2-01": _build_local_variant(
        variant_name="test-layoutV2-01",
        env_name="PointMaze Test Layout V2 01",
        structure_desc_en=_POINTMAZE_V2_DESCS["55-60"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["55-60"][1],
    ),
    "test-layoutV2-02": _build_local_variant(
        variant_name="test-layoutV2-02",
        env_name="PointMaze Test Layout V2 02",
        structure_desc_en=_POINTMAZE_V2_DESCS["55-60"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["55-60"][1],
    ),
    "test-layoutV2-03": _build_local_variant(
        variant_name="test-layoutV2-03",
        env_name="PointMaze Test Layout V2 03",
        structure_desc_en=_POINTMAZE_V2_DESCS["60-65"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["60-65"][1],
    ),
    "test-layoutV2-04": _build_local_variant(
        variant_name="test-layoutV2-04",
        env_name="PointMaze Test Layout V2 04",
        structure_desc_en=_POINTMAZE_V2_DESCS["60-65"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["60-65"][1],
    ),
    "test-layoutV2-05": _build_local_variant(
        variant_name="test-layoutV2-05",
        env_name="PointMaze Test Layout V2 05",
        structure_desc_en=_POINTMAZE_V2_DESCS["65-70"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["65-70"][1],
    ),
    "test-layoutV2-06": _build_local_variant(
        variant_name="test-layoutV2-06",
        env_name="PointMaze Test Layout V2 06",
        structure_desc_en=_POINTMAZE_V2_DESCS["65-70"][0],
        structure_desc_zh=_POINTMAZE_V2_DESCS["65-70"][1],
    ),
    "local-EXturn": _build_local_variant(
        variant_name="local-EXturn",
        env_name="PointMaze Local EXturn",
        structure_desc_en=_POINTMAZE_EXTURN_DESC[0],
        structure_desc_zh=_POINTMAZE_EXTURN_DESC[1],
    ),
    "test-EXturn": _build_local_variant(
        variant_name="test-EXturn",
        env_name="PointMaze Test EXturn",
        structure_desc_en=_POINTMAZE_EXTURN_DESC[0],
        structure_desc_zh=_POINTMAZE_EXTURN_DESC[1],
    ),
    "local-EXturn-simple": _build_local_variant(
        variant_name="local-EXturn-simple",
        env_name="PointMaze Local EXturn Simple",
        structure_desc_en=_POINTMAZE_EXTURN_SIMPLE_DESC[0],
        structure_desc_zh=_POINTMAZE_EXTURN_SIMPLE_DESC[1],
    ),
}


def get_pointmaze_variant_type(meta: dict) -> str:
    variant_type = meta.get("varient_type", meta.get("variant_type", "remote"))
    if variant_type not in {"remote", "local"}:
        raise ValueError(f"Unsupported PointMaze variant type: {variant_type!r}")
    return variant_type


def resolve_local_dataset_path(
    dataset_path: str | Path,
    local_dataset_root: str | Path | None = None,
    *,
    reward_type: str | None = None,
    default_reward_type: str = "sparse",
) -> Path:
    default_path = (
        reward_typed_dataset_path(
            dataset_path,
            reward_type=reward_type,
            default_reward_type=default_reward_type,
        )
        if reward_type is not None
        else Path(dataset_path).expanduser()
    )
    if local_dataset_root is None:
        path = default_path
    else:
        root_path = Path(local_dataset_root).expanduser()
        path = (
            root_path
            if root_path.name == default_path.name or (root_path / "data").is_dir()
            else root_path / default_path.name
        )
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path
