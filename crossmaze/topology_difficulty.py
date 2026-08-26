"""Canonical CrossMaze topology-difficulty calculation.

Topology difficulty v3.3 is a whole-map metric based on maximal canonical
shortest paths.  It is deliberately separate from both evaluation path/map
difficulty in :mod:`crossmaze.eval_position` and the legacy repository-side
``utils.maze_metrics.static_difficulty`` contract.

This module is self-contained and standard-library-only so the installable
CrossMaze package does not depend on repository-level ``utils`` modules.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean


Cell = tuple[int, int]
MazeMap = list[list[object]]
Path = tuple[Cell, ...]

TOPOLOGY_DIFFICULTY_VERSION = "v3.3"
TOP_LEVEL_PATH_COUNT_LOG_SCALE = 100.0
MEAN_TOP_LEVEL_PATH_LEN_HALF_SCALE = 10.0
MEAN_TOP_LEVEL_TURNS_HALF_SCALE = 3.0
MAP_AREA_HALF_SCALE = 100.0
T_JUNCTION_COUNT_HALF_SCALE = 5.0
CROSS_JUNCTION_COUNT_HALF_SCALE = 2.0
COMPONENT_CONTRAST_GAMMA = 1.5
FINAL_SCORE_LOGISTIC_MIDPOINT = 45.0
FINAL_SCORE_LOGISTIC_TEMPERATURE = 10.0
TOPOLOGY_DIFFICULTY_WEIGHTS = {
    "path_count": 0.05,
    "mean_path_len": 0.25,
    "mean_turns": 0.15,
    "mean_detour_exposure": 0.25,
    "map_area": 0.20,
    "t_junctions": 0.05,
    "cross_junctions": 0.05,
}

# Compatibility names retained for callers of the earlier draft module.
TOP_LEVEL_PATH_DIFFICULTY_VERSION = TOPOLOGY_DIFFICULTY_VERSION
TOP_LEVEL_PATH_DIFFICULTY_WEIGHTS = TOPOLOGY_DIFFICULTY_WEIGHTS


@dataclass(frozen=True)
class TopLevelPathMetrics:
    rows: int
    cols: int
    map_area: int
    free_cells: int
    edge_count: int
    unordered_pair_count: int
    unique_canonical_path_count: int
    contained_path_count: int
    top_level_path_count: int
    mean_top_level_path_len: float
    max_top_level_path_len: int
    mean_top_level_turns: float
    max_top_level_turns: int
    mean_top_level_detour_ratio: float
    max_top_level_detour_ratio: float
    raw_degree_3_cells: int
    raw_degree_4_cells: int
    corridor_t_junctions: int
    corridor_cross_junctions: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TopLevelPathDifficulty:
    version: str
    score: float
    raw_score: float
    path_count_component: float
    mean_path_len_component: float
    mean_turns_component: float
    mean_detour_exposure_component: float
    map_area_component: float
    t_junction_component: float
    cross_junction_component: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["config"] = topology_difficulty_config()
        return payload


@dataclass(frozen=True)
class TopologyDifficultyResult:
    """One-call CrossMaze result containing metrics and the v3.3 score."""

    metrics: TopLevelPathMetrics
    difficulty: TopLevelPathDifficulty

    @property
    def version(self) -> str:
        return self.difficulty.version

    @property
    def score(self) -> float:
        return self.difficulty.score

    @property
    def raw_score(self) -> float:
        return self.difficulty.raw_score

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "difficulty": self.difficulty.to_dict(),
        }


def topology_difficulty_config() -> dict:
    """Return the complete, fixed v3.3 topology-difficulty contract."""

    return {
        "version": TOPOLOGY_DIFFICULTY_VERSION,
        "top_level_path_definition": (
            "maximal_contiguous_canonical_shortest_path_per_unordered_pair"
        ),
        "path_count_log_scale": float(TOP_LEVEL_PATH_COUNT_LOG_SCALE),
        "mean_path_len_half_scale": float(MEAN_TOP_LEVEL_PATH_LEN_HALF_SCALE),
        "mean_turns_half_scale": float(MEAN_TOP_LEVEL_TURNS_HALF_SCALE),
        "map_area_half_scale": float(MAP_AREA_HALF_SCALE),
        "mean_detour_exposure_definition": (
            "mean_top_level_detour_ratio * mean_path_len_component"
        ),
        "t_junction_count_half_scale": float(T_JUNCTION_COUNT_HALF_SCALE),
        "cross_junction_count_half_scale": float(CROSS_JUNCTION_COUNT_HALF_SCALE),
        "component_contrast_gamma": float(COMPONENT_CONTRAST_GAMMA),
        "component_contrast_definition": (
            "c^gamma / (c^gamma + (1-c)^gamma), applied independently "
            "to every normalized component before weighting"
        ),
        "raw_score_definition": (
            "100 * weighted_sum(contrast(normalized_component))"
        ),
        "final_score_definition": (
            "100 / (1 + exp(-(raw_score - midpoint) / temperature))"
        ),
        "final_score_logistic_midpoint": float(FINAL_SCORE_LOGISTIC_MIDPOINT),
        "final_score_logistic_temperature": float(
            FINAL_SCORE_LOGISTIC_TEMPERATURE
        ),
        "weights": dict(TOPOLOGY_DIFFICULTY_WEIGHTS),
    }


def top_level_path_difficulty_config() -> dict:
    """Compatibility alias for :func:`topology_difficulty_config`."""

    return topology_difficulty_config()


def _normalize_maze_map(maze_map: MazeMap) -> list[list[int]]:
    if not maze_map:
        raise ValueError("maze_map must be non-empty")
    cols = len(maze_map[0])
    if cols == 0:
        raise ValueError("maze_map rows must be non-empty")
    if any(len(row) != cols for row in maze_map):
        raise ValueError("maze_map rows must have equal length")
    return [
        [1 if cell == 1 else 0 for cell in row]
        for row in maze_map
    ]


def _validate_rectangular_boundary(maze_map: MazeMap) -> list[list[int]]:
    normalized = _normalize_maze_map(maze_map)
    rows = len(normalized)
    cols = len(normalized[0])
    if rows < 3 or cols < 3:
        raise ValueError("maze_map must be at least 3x3")
    if any(cell != 1 for cell in normalized[0] + normalized[-1]):
        raise ValueError("maze_map top and bottom boundaries must be walls")
    if any(row[0] != 1 or row[-1] != 1 for row in normalized):
        raise ValueError("maze_map left and right boundaries must be walls")
    return normalized


def _build_adjacency(maze_map: list[list[int]]) -> dict[Cell, list[Cell]]:
    free = {
        (row_index, col_index)
        for row_index, row in enumerate(maze_map)
        for col_index, cell in enumerate(row)
        if cell != 1
    }
    adjacency = {}
    for cell in free:
        row, col = cell
        adjacency[cell] = [
            neighbor
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            if neighbor in free
        ]
    return adjacency


def _connected_component_count(adjacency: dict[Cell, list[Cell]]) -> int:
    unseen = set(adjacency)
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        unseen.difference_update(_reachable_cells(adjacency, start))
    return count


def _reachable_cells(
    adjacency: dict[Cell, list[Cell]],
    start: Cell,
) -> set[Cell]:
    reached = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in adjacency[cell]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def _count_path_turns(path: Path) -> int:
    if len(path) < 3:
        return 0
    directions = [
        (right[0] - left[0], right[1] - left[1])
        for left, right in zip(path, path[1:])
    ]
    return sum(
        left_direction != right_direction
        for left_direction, right_direction in zip(directions, directions[1:])
    )


def _half_saturating(value: float, half_scale: float) -> float:
    value = max(float(value), 0.0)
    return float(value / (value + float(half_scale)))


def contrast_normalized_component(
    value: float,
    gamma: float = COMPONENT_CONTRAST_GAMMA,
) -> float:
    """Increase contrast around 0.5 while preserving 0, 0.5, and 1."""

    value = min(max(float(value), 0.0), 1.0)
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if value in {0.0, 1.0}:
        return value
    lower = value**gamma
    upper = (1.0 - value) ** gamma
    return float(lower / (lower + upper))


def logistic_final_score(raw_score: float) -> float:
    """Map a raw topology score to the final 0-100 difficulty scale."""

    exponent = -(
        float(raw_score) - FINAL_SCORE_LOGISTIC_MIDPOINT
    ) / FINAL_SCORE_LOGISTIC_TEMPERATURE
    return float(100.0 / (1.0 + math.exp(exponent)))


def compute_top_level_path_difficulty(
    metrics: TopLevelPathMetrics,
) -> TopLevelPathDifficulty:
    """Score precomputed top-level-path metrics with the v3.3 contract."""

    path_count_component = min(
        1.0,
        math.log1p(metrics.top_level_path_count)
        / math.log1p(TOP_LEVEL_PATH_COUNT_LOG_SCALE),
    )
    mean_path_len_component = _half_saturating(
        metrics.mean_top_level_path_len,
        MEAN_TOP_LEVEL_PATH_LEN_HALF_SCALE,
    )
    mean_turns_component = _half_saturating(
        metrics.mean_top_level_turns,
        MEAN_TOP_LEVEL_TURNS_HALF_SCALE,
    )
    mean_detour_exposure_component = (
        min(max(float(metrics.mean_top_level_detour_ratio), 0.0), 1.0)
        * mean_path_len_component
    )
    map_area_component = _half_saturating(
        metrics.map_area,
        MAP_AREA_HALF_SCALE,
    )
    t_junction_component = _half_saturating(
        metrics.corridor_t_junctions,
        T_JUNCTION_COUNT_HALF_SCALE,
    )
    cross_junction_component = _half_saturating(
        metrics.corridor_cross_junctions,
        CROSS_JUNCTION_COUNT_HALF_SCALE,
    )
    normalized_components = {
        "path_count": path_count_component,
        "mean_path_len": mean_path_len_component,
        "mean_turns": mean_turns_component,
        "mean_detour_exposure": mean_detour_exposure_component,
        "map_area": map_area_component,
        "t_junctions": t_junction_component,
        "cross_junctions": cross_junction_component,
    }
    components = {
        name: contrast_normalized_component(value)
        for name, value in normalized_components.items()
    }
    raw_score = 100.0 * sum(
        TOPOLOGY_DIFFICULTY_WEIGHTS[name] * value
        for name, value in components.items()
    )
    score = logistic_final_score(raw_score)
    return TopLevelPathDifficulty(
        version=TOPOLOGY_DIFFICULTY_VERSION,
        score=round(score, 2),
        raw_score=round(raw_score, 2),
        path_count_component=components["path_count"],
        mean_path_len_component=components["mean_path_len"],
        mean_turns_component=components["mean_turns"],
        mean_detour_exposure_component=components["mean_detour_exposure"],
        map_area_component=components["map_area"],
        t_junction_component=components["t_junctions"],
        cross_junction_component=components["cross_junctions"],
    )


def _canonical_undirected_path(path: Path) -> Path:
    reversed_path = tuple(reversed(path))
    return min(path, reversed_path)


def _path_detour_ratio(path: Path) -> float:
    path_len = len(path) - 1
    if path_len <= 0:
        return 0.0
    start, goal = path[0], path[-1]
    manhattan_distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return float(1.0 - manhattan_distance / path_len)


def _has_walled_diagonals(
    cell: Cell,
    maze_map: list[list[int]],
) -> bool:
    row, col = cell
    return all(
        maze_map[row + row_delta][col + col_delta] == 1
        for row_delta, col_delta in ((-1, -1), (-1, 1), (1, -1), (1, 1))
    )


def _bfs_parents(
    adjacency: dict[Cell, list[Cell]],
    start: Cell,
) -> dict[Cell, Cell | None]:
    parents: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in adjacency[cell]:
            if neighbor not in parents:
                parents[neighbor] = cell
                queue.append(neighbor)
    return parents


def _path_from_parents(parents: dict[Cell, Cell | None], goal: Cell) -> Path:
    if goal not in parents:
        raise ValueError(f"goal cell {goal} is not reachable")
    reversed_path = []
    cell: Cell | None = goal
    while cell is not None:
        reversed_path.append(cell)
        cell = parents[cell]
    return tuple(reversed(reversed_path))


def canonical_pair_paths(adjacency: dict[Cell, list[Cell]]) -> tuple[Path, ...]:
    """Return one deterministic shortest path per unordered free-cell pair."""

    cells = sorted(adjacency)
    paths = []
    for start_index, start in enumerate(cells):
        parents = _bfs_parents(adjacency, start)
        for goal in cells[start_index + 1 :]:
            path = _path_from_parents(parents, goal)
            paths.append(_canonical_undirected_path(path))
    return tuple(paths)


def maximal_contiguous_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Remove paths contained contiguously in another path, ignoring direction."""

    unique_paths = set(paths)
    contained: set[Path] = set()
    for path in unique_paths:
        cell_count = len(path)
        for start_index in range(cell_count - 1):
            for stop_index in range(start_index + 2, cell_count + 1):
                if start_index == 0 and stop_index == cell_count:
                    continue
                subpath = _canonical_undirected_path(path[start_index:stop_index])
                if subpath in unique_paths:
                    contained.add(subpath)
    return tuple(
        sorted(
            unique_paths - contained,
            key=lambda path: (-(len(path) - 1), path),
        )
    )


def compute_top_level_path_metrics(maze_map: MazeMap) -> TopLevelPathMetrics:
    """Compute the v3.3 top-level-path metrics for one connected maze."""

    normalized = _validate_rectangular_boundary(maze_map)
    adjacency = _build_adjacency(normalized)
    if not adjacency:
        raise ValueError("maze_map must contain at least one free cell")
    component_count = _connected_component_count(adjacency)
    if component_count != 1:
        raise ValueError(
            f"maze_map must be connected, got {component_count} components"
        )

    pair_paths = canonical_pair_paths(adjacency)
    top_level_paths = maximal_contiguous_paths(pair_paths)
    path_lengths = [len(path) - 1 for path in top_level_paths]
    turn_counts = [_count_path_turns(path) for path in top_level_paths]
    detour_ratios = [_path_detour_ratio(path) for path in top_level_paths]
    degrees = {
        cell: len(neighbors)
        for cell, neighbors in adjacency.items()
    }
    free_cell_count = len(adjacency)
    edge_count = sum(degrees.values()) // 2
    raw_degree_3_cells = sum(1 for degree in degrees.values() if degree == 3)
    raw_degree_4_cells = sum(1 for degree in degrees.values() if degree == 4)
    corridor_t_junctions = sum(
        1
        for cell, degree in degrees.items()
        if degree == 3 and _has_walled_diagonals(cell, normalized)
    )
    corridor_cross_junctions = sum(
        1
        for cell, degree in degrees.items()
        if degree == 4 and _has_walled_diagonals(cell, normalized)
    )

    return TopLevelPathMetrics(
        rows=len(normalized),
        cols=len(normalized[0]),
        map_area=len(normalized) * len(normalized[0]),
        free_cells=free_cell_count,
        edge_count=edge_count,
        unordered_pair_count=free_cell_count * (free_cell_count - 1) // 2,
        unique_canonical_path_count=len(set(pair_paths)),
        contained_path_count=len(set(pair_paths)) - len(top_level_paths),
        top_level_path_count=len(top_level_paths),
        mean_top_level_path_len=float(mean(path_lengths)) if path_lengths else 0.0,
        max_top_level_path_len=max(path_lengths, default=0),
        mean_top_level_turns=float(mean(turn_counts)) if turn_counts else 0.0,
        max_top_level_turns=max(turn_counts, default=0),
        mean_top_level_detour_ratio=(
            float(mean(detour_ratios)) if detour_ratios else 0.0
        ),
        max_top_level_detour_ratio=max(detour_ratios, default=0.0),
        raw_degree_3_cells=raw_degree_3_cells,
        raw_degree_4_cells=raw_degree_4_cells,
        corridor_t_junctions=corridor_t_junctions,
        corridor_cross_junctions=corridor_cross_junctions,
    )


def compute_topology_difficulty(maze_map: MazeMap) -> TopologyDifficultyResult:
    """Compute all topology metrics and the formal v3.3 score in one call."""

    metrics = compute_top_level_path_metrics(maze_map)
    difficulty = compute_top_level_path_difficulty(metrics)
    return TopologyDifficultyResult(metrics=metrics, difficulty=difficulty)


__all__ = [
    "COMPONENT_CONTRAST_GAMMA",
    "FINAL_SCORE_LOGISTIC_MIDPOINT",
    "FINAL_SCORE_LOGISTIC_TEMPERATURE",
    "TOPOLOGY_DIFFICULTY_VERSION",
    "TOPOLOGY_DIFFICULTY_WEIGHTS",
    "TopLevelPathDifficulty",
    "TopLevelPathMetrics",
    "TopologyDifficultyResult",
    "canonical_pair_paths",
    "compute_top_level_path_difficulty",
    "compute_top_level_path_metrics",
    "compute_topology_difficulty",
    "contrast_normalized_component",
    "logistic_final_score",
    "maximal_contiguous_paths",
    "topology_difficulty_config",
]
