"""Deterministic procedural maze maps addressed by integer seeds.

The public feature name is ``seed_map``. Callers depend only on the versioned
seed-map contract:

``(seed_map_version, size config, map_seed) -> one exact 0/1 maze_map``.

Public map sizes are the final CrossMaze matrix dimensions, including the
outer wall boundary.  Random sizes are square and sampled from the configured
odd sizes.  Fixed sizes may be rectangular, but both dimensions must be odd so
the room/corridor lattice is unambiguous.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from crossmaze.layout import format_raw_matrix, format_visual_map, maze_shape_text
from crossmaze.reward import normalize_reward_type


SEED_MAP_VERSION = "v1"
SEED_MAP_SIZE_MODES = ("random", "fixed")
DEFAULT_SEED_MAP_MIN_SIZE = 5
DEFAULT_SEED_MAP_MAX_SIZE = 15


@dataclass(frozen=True)
class SeedMapSpec:
    """Versioned configuration that maps an integer seed to one maze."""

    version: str = SEED_MAP_VERSION
    size_mode: str = "random"
    min_size: int = DEFAULT_SEED_MAP_MIN_SIZE
    max_size: int = DEFAULT_SEED_MAP_MAX_SIZE
    fixed_rows: int | None = None
    fixed_cols: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return int(value)


def _validate_odd_size(value: Any, *, field: str) -> int:
    size = _require_int(value, field=field, minimum=5)
    if size % 2 == 0:
        raise ValueError(
            f"{field} must be odd because seed-map room and corridor cells alternate, "
            f"got {size}"
        )
    return size


def normalize_seed_map_spec(value: SeedMapSpec | Mapping[str, Any] | None) -> SeedMapSpec:
    """Validate a seed-map generator config and return its canonical form."""

    if value is None:
        return SeedMapSpec()
    from_spec_object = isinstance(value, SeedMapSpec)
    if from_spec_object:
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError(
            "seed_map spec must be a mapping or SeedMapSpec, "
            f"got {type(value).__name__}"
        )

    aliases = {
        "seed_map_version": "version",
        "seed_map_size_mode": "size_mode",
        "seed_map_min_size": "min_size",
        "seed_map_max_size": "max_size",
        "seed_map_fixed_rows": "fixed_rows",
        "seed_map_fixed_cols": "fixed_cols",
    }
    for alias, canonical in aliases.items():
        if alias in raw:
            if canonical in raw and raw[canonical] != raw[alias]:
                raise ValueError(
                    f"Conflicting seed-map fields {canonical!r} and {alias!r}"
                )
            raw[canonical] = raw.pop(alias)

    allowed = {
        "version",
        "size_mode",
        "min_size",
        "max_size",
        "fixed_rows",
        "fixed_cols",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown seed_map generator fields: {unknown}")

    version = str(raw.get("version", SEED_MAP_VERSION)).strip()
    if version != SEED_MAP_VERSION:
        raise ValueError(
            f"Unsupported seed_map_version={version!r}; supported: {SEED_MAP_VERSION!r}"
        )
    size_mode = str(raw.get("size_mode", "random")).strip()
    if size_mode not in SEED_MAP_SIZE_MODES:
        raise ValueError(
            f"seed_map_size_mode must be one of {list(SEED_MAP_SIZE_MODES)}, "
            f"got {size_mode!r}"
        )

    if size_mode == "random":
        min_size = _validate_odd_size(
            raw.get("min_size", DEFAULT_SEED_MAP_MIN_SIZE),
            field="seed_map_min_size",
        )
        max_size = _validate_odd_size(
            raw.get("max_size", DEFAULT_SEED_MAP_MAX_SIZE),
            field="seed_map_max_size",
        )
        if min_size > max_size:
            raise ValueError(
                "seed_map_min_size must be <= seed_map_max_size, "
                f"got {min_size} > {max_size}"
            )
        if raw.get("fixed_rows") is not None or raw.get("fixed_cols") is not None:
            raise ValueError(
                "seed_map_fixed_rows/cols are only valid when "
                "seed_map_size_mode='fixed'"
            )
        return SeedMapSpec(
            version=version,
            size_mode=size_mode,
            min_size=min_size,
            max_size=max_size,
        )

    fixed_rows = _validate_odd_size(
        raw.get("fixed_rows"),
        field="seed_map_fixed_rows",
    )
    fixed_cols = _validate_odd_size(
        raw.get("fixed_cols"),
        field="seed_map_fixed_cols",
    )
    if not from_spec_object and ("min_size" in raw or "max_size" in raw):
        canonical_fixed_sizes = (
            raw.get("min_size") == fixed_rows
            and raw.get("max_size") == fixed_rows
        )
        if not canonical_fixed_sizes:
            raise ValueError(
                "seed_map_min_size/max_size are only valid when "
                "seed_map_size_mode='random'"
            )
    return SeedMapSpec(
        version=version,
        size_mode=size_mode,
        min_size=fixed_rows,
        max_size=fixed_rows,
        fixed_rows=fixed_rows,
        fixed_cols=fixed_cols,
    )


def stable_seed_map_int(*parts: Any, bits: int = 64) -> int:
    """Return a process- and platform-stable non-negative integer."""

    if bits < 1 or bits > 256:
        raise ValueError(f"bits must be in [1, 256], got {bits}")
    payload = json.dumps(
        list(parts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return value & ((1 << bits) - 1)


def seed_map_shape(map_seed: int, spec: SeedMapSpec | Mapping[str, Any] | None = None) -> tuple[int, int]:
    """Resolve the final CrossMaze matrix shape for ``map_seed``."""

    map_seed = _require_int(map_seed, field="map_seed", minimum=0)
    resolved = normalize_seed_map_spec(spec)
    if resolved.size_mode == "fixed":
        assert resolved.fixed_rows is not None and resolved.fixed_cols is not None
        return resolved.fixed_rows, resolved.fixed_cols

    sizes = list(range(resolved.min_size, resolved.max_size + 1, 2))
    index = stable_seed_map_int(
        resolved.version,
        resolved.to_dict(),
        map_seed,
        "size",
    ) % len(sizes)
    size = sizes[index]
    return size, size


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def generate_seed_map(
    map_seed: int,
    spec: SeedMapSpec | Mapping[str, Any] | None = None,
) -> list[list[int]]:
    """Generate one connected perfect maze for ``map_seed``."""

    map_seed = _require_int(map_seed, field="map_seed", minimum=0)
    resolved = normalize_seed_map_spec(spec)
    rows, cols = seed_map_shape(map_seed, resolved)

    maze_map = [[1 for _ in range(cols)] for _ in range(rows)]
    room_positions = [
        (row, col)
        for row in range(1, rows - 1, 2)
        for col in range(1, cols - 1, 2)
    ]
    room_index = {cell: idx for idx, cell in enumerate(room_positions)}
    for row, col in room_positions:
        maze_map[row][col] = 0

    edges = []
    for row, col in room_positions:
        for delta_row, delta_col in ((0, 2), (2, 0)):
            neighbor = (row + delta_row, col + delta_col)
            if neighbor not in room_index:
                continue
            wall = (row + delta_row // 2, col + delta_col // 2)
            priority = stable_seed_map_int(
                resolved.version,
                resolved.to_dict(),
                map_seed,
                "edge",
                row,
                col,
                neighbor[0],
                neighbor[1],
                bits=256,
            )
            edges.append((priority, (row, col), neighbor, wall))
    edges.sort(key=lambda item: (item[0], item[1], item[2]))

    sets = _DisjointSet(len(room_positions))
    carved = 0
    for _, left, right, wall in edges:
        if sets.union(room_index[left], room_index[right]):
            maze_map[wall[0]][wall[1]] = 0
            carved += 1
    if carved != len(room_positions) - 1:
        raise RuntimeError(
            "seed-map generator failed to connect every room: "
            f"rooms={len(room_positions)}, carved_edges={carved}"
        )

    validate_seed_map(maze_map)
    return maze_map


def validate_seed_map(maze_map: list[list[int]]) -> None:
    """Validate the binary, bounded, connected CrossMaze map contract."""

    if not isinstance(maze_map, list) or not maze_map or not maze_map[0]:
        raise ValueError("seed-map maze_map must be a non-empty rectangular list")
    rows = len(maze_map)
    cols = len(maze_map[0])
    if rows < 5 or cols < 5:
        raise ValueError(f"seed-map maze_map must be at least 5x5, got {rows}x{cols}")
    if any(not isinstance(row, list) or len(row) != cols for row in maze_map):
        raise ValueError("seed-map maze_map must be rectangular")
    invalid = sorted(
        {
            cell
            for row in maze_map
            for cell in row
            if isinstance(cell, bool) or cell not in (0, 1)
        },
        key=repr,
    )
    if invalid:
        raise ValueError(f"seed-map maze_map must contain only integer 0/1, got {invalid}")
    if any(cell != 1 for cell in maze_map[0] + maze_map[-1]):
        raise ValueError("seed-map maze_map top/bottom boundaries must be walls")
    if any(row[0] != 1 or row[-1] != 1 for row in maze_map):
        raise ValueError("seed-map maze_map left/right boundaries must be walls")

    free = {
        (row, col)
        for row, values in enumerate(maze_map)
        for col, cell in enumerate(values)
        if cell == 0
    }
    if len(free) < 2:
        raise ValueError("seed-map maze_map must contain at least two free cells")
    start = min(free)
    visited = {start}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for next_cell in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if next_cell in free and next_cell not in visited:
                visited.add(next_cell)
                queue.append(next_cell)
    if visited != free:
        raise ValueError(
            "seed-map maze_map free cells must be connected: "
            f"reachable={len(visited)}, free={len(free)}"
        )


def seed_map_hash(maze_map: list[list[int]]) -> str:
    validate_seed_map(maze_map)
    payload = json.dumps(maze_map, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_map_generator_hash(
    spec: SeedMapSpec | Mapping[str, Any] | None,
) -> str:
    resolved = normalize_seed_map_spec(spec)
    payload = json.dumps(
        resolved.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_seed_map_prompt_vars(
    env_family: str,
    *,
    map_seed: int,
    maze_map: list[list[int]],
    reward_type: str,
) -> dict[str, Any]:
    """Build per-map prompt variables without registering a synthetic variant."""

    validate_seed_map(maze_map)
    reward_type = normalize_reward_type(reward_type)
    common = {
        "maze_seed": int(map_seed),
        "maze_map": [list(row) for row in maze_map],
        "maze_shape": maze_shape_text(maze_map),
        "maze_raw_matrix": format_raw_matrix(maze_map),
        "maze_visual": format_visual_map(maze_map),
        "reward_type": reward_type,
    }
    if env_family == "pointmaze":
        return {
            **common,
            "maze_size_scaling": 1.0,
            "env_name": f"PointMaze Seed Map {int(map_seed)}",
            "reward_desc_en": f"{reward_type} reward",
            "reward_desc_zh": "稀疏奖励" if reward_type == "sparse" else "稠密奖励",
            "structure_desc_en": (
                "A procedurally generated connected seed map with narrow corridors, "
                "branches, turns, and dead ends."
            ),
            "structure_desc_zh": (
                "一个按种子程序化生成的连通迷宫，包含狭窄走廊、分支、转弯和死路。"
            ),
        }
    if env_family == "antmaze":
        return {
            **common,
            "maze_size_scaling": 4.0,
            "env_name": f"AntMaze Seed Map {int(map_seed)}",
            "dataset_style": "seed-map trajectories with randomized reset and goal cells",
            "structure_desc_en": (
                "A procedurally generated connected seed map with narrow corridors, "
                "branches, turns, and dead ends."
            ),
        }
    raise ValueError(
        f"Unsupported seed-map env_family={env_family!r}; "
        "expected 'pointmaze' or 'antmaze'"
    )


def build_seed_map_env_paras(
    env_family: str,
    *,
    maze_map: list[list[int]],
    reward_type: str,
    max_episode_steps: int | None = None,
) -> dict[str, Any]:
    """Build Gymnasium Robotics kwargs for one generated seed map."""

    validate_seed_map(maze_map)
    reward_type = normalize_reward_type(reward_type)
    if env_family == "pointmaze":
        default_steps = 6 * len(maze_map) * len(maze_map[0])
        return {
            "id": "PointMaze_UMaze-v3",
            "maze_map": [list(row) for row in maze_map],
            "reward_type": reward_type,
            "continuing_task": True,
            "reset_target": True,
            "max_episode_steps": int(
                default_steps if max_episode_steps is None else max_episode_steps
            ),
        }
    if env_family == "antmaze":
        return {
            "id": "AntMaze_UMaze-v4",
            "maze_map": [list(row) for row in maze_map],
            "reward_type": reward_type,
            "continuing_task": True,
            "reset_target": False,
            "max_episode_steps": int(
                1000 if max_episode_steps is None else max_episode_steps
            ),
        }
    raise ValueError(
        f"Unsupported seed-map env_family={env_family!r}; "
        "expected 'pointmaze' or 'antmaze'"
    )
