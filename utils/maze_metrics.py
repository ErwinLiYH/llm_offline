from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from statistics import mean


Cell = tuple[int, int]
MazeMap = list[list[object]]


@dataclass(frozen=True)
class MazeMetrics:
    rows: int
    cols: int
    total_cells: int
    free_cells: int
    wall_cells: int
    wall_density: float
    edge_count: int
    connected_components: int
    start: Cell
    goal: Cell
    shortest_path_len: int
    normalized_path_len: float
    diameter: int
    solution_turns: int
    solution_turn_rate: float
    solution_junctions: int
    degree_1_deadends: int
    degree_2_corridors: int
    degree_3_forks: int
    degree_4_crosses: int
    junction_count: int
    cycle_rank: int
    articulation_points: int
    bridge_edges: int
    corridor_count: int
    mean_corridor_len: float
    max_corridor_len: int
    static_difficulty: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["start"] = list(self.start)
        payload["goal"] = list(self.goal)
        return payload


def normalize_maze_map(maze_map: MazeMap) -> list[list[int]]:
    if not maze_map:
        raise ValueError("maze_map must be non-empty")
    cols = len(maze_map[0])
    if cols == 0:
        raise ValueError("maze_map rows must be non-empty")
    normalized = []
    for row in maze_map:
        if len(row) != cols:
            raise ValueError("maze_map rows must have equal length")
        normalized.append([1 if cell == 1 else 0 for cell in row])
    return normalized


def format_maze_map(maze_map: MazeMap) -> str:
    normalized = normalize_maze_map(maze_map)
    return "\n".join(
        "".join("#" if cell == 1 else "." for cell in row)
        for row in normalized
    )


def parse_maze_text(text: str) -> list[list[int]]:
    rows = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]
    if not rows:
        raise ValueError("maze text must contain at least one non-empty row")
    width = len(rows[0])
    maze = []
    for row in rows:
        if len(row) != width:
            raise ValueError("maze text rows must have equal width")
        parsed_row = []
        for char in row:
            if char == "#":
                parsed_row.append(1)
            elif char in {".", "0", "r", "g", "c"}:
                parsed_row.append(0)
            else:
                raise ValueError(f"Unsupported maze character: {char!r}")
        maze.append(parsed_row)
    return maze


def find_marker(maze_map: MazeMap, marker: str) -> Cell | None:
    for row_idx, row in enumerate(maze_map):
        for col_idx, cell in enumerate(row):
            if cell == marker:
                return row_idx, col_idx
    return None


def validate_rectangular_boundary(maze_map: MazeMap) -> None:
    normalized = normalize_maze_map(maze_map)
    rows = len(normalized)
    cols = len(normalized[0])
    if rows < 3 or cols < 3:
        raise ValueError("maze_map must be at least 3x3")
    if any(cell != 1 for cell in normalized[0] + normalized[-1]):
        raise ValueError("maze_map top and bottom boundaries must be walls")
    for row in normalized:
        if row[0] != 1 or row[-1] != 1:
            raise ValueError("maze_map left and right boundaries must be walls")


def free_cells(maze_map: MazeMap) -> set[Cell]:
    normalized = normalize_maze_map(maze_map)
    return {
        (row_idx, col_idx)
        for row_idx, row in enumerate(normalized)
        for col_idx, cell in enumerate(row)
        if cell != 1
    }


def neighbor_cells(cell: Cell, free: set[Cell]) -> list[Cell]:
    row, col = cell
    neighbors = []
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = row + d_row, col + d_col
        if neighbor in free:
            neighbors.append(neighbor)
    return neighbors


def build_adjacency(maze_map: MazeMap) -> dict[Cell, list[Cell]]:
    free = free_cells(maze_map)
    return {cell: neighbor_cells(cell, free) for cell in free}


def connected_components(adjacency: dict[Cell, list[Cell]]) -> list[set[Cell]]:
    unseen = set(adjacency)
    components = []
    while unseen:
        start = unseen.pop()
        component = {start}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for neighbor in adjacency[cell]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def shortest_path(adjacency: dict[Cell, list[Cell]], start: Cell, goal: Cell) -> list[Cell]:
    if start not in adjacency:
        raise ValueError(f"start cell is not free: {start}")
    if goal not in adjacency:
        raise ValueError(f"goal cell is not free: {goal}")
    queue = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        for neighbor in adjacency[cell]:
            if neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    if goal not in parent:
        raise ValueError(f"goal cell {goal} is not reachable from start cell {start}")
    path = []
    cell: Cell | None = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return path


def farthest_cell(adjacency: dict[Cell, list[Cell]], start: Cell) -> tuple[Cell, int]:
    distances = bfs_distances(adjacency, start)
    return max(distances.items(), key=lambda item: (item[1], item[0]))


def bfs_distances(adjacency: dict[Cell, list[Cell]], start: Cell) -> dict[Cell, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in adjacency[cell]:
            if neighbor not in distances:
                distances[neighbor] = distances[cell] + 1
                queue.append(neighbor)
    return distances


def diameter_and_endpoints(adjacency: dict[Cell, list[Cell]]) -> tuple[int, Cell, Cell]:
    best_distance = -1
    best_start = next(iter(adjacency))
    best_goal = best_start
    for cell in adjacency:
        distances = bfs_distances(adjacency, cell)
        far_cell, far_distance = max(distances.items(), key=lambda item: (item[1], item[0]))
        if far_distance > best_distance:
            best_distance = far_distance
            best_start = cell
            best_goal = far_cell
    return best_distance, best_start, best_goal


def count_solution_turns(path: list[Cell]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    previous_direction = _direction(path[0], path[1])
    for idx in range(1, len(path) - 1):
        current_direction = _direction(path[idx], path[idx + 1])
        if current_direction != previous_direction:
            turns += 1
        previous_direction = current_direction
    return turns


def _direction(left: Cell, right: Cell) -> Cell:
    return right[0] - left[0], right[1] - left[1]


def articulation_points_and_bridges(adjacency: dict[Cell, list[Cell]]) -> tuple[set[Cell], set[tuple[Cell, Cell]]]:
    timer = 0
    visited: set[Cell] = set()
    tin: dict[Cell, int] = {}
    low: dict[Cell, int] = {}
    articulation: set[Cell] = set()
    bridges: set[tuple[Cell, Cell]] = set()

    def dfs(cell: Cell, parent: Cell | None) -> None:
        nonlocal timer
        visited.add(cell)
        tin[cell] = low[cell] = timer
        timer += 1
        children = 0
        for neighbor in adjacency[cell]:
            if neighbor == parent:
                continue
            if neighbor in visited:
                low[cell] = min(low[cell], tin[neighbor])
                continue
            dfs(neighbor, cell)
            low[cell] = min(low[cell], low[neighbor])
            if low[neighbor] >= tin[cell] and parent is not None:
                articulation.add(cell)
            if low[neighbor] > tin[cell]:
                bridges.add(_edge(cell, neighbor))
            children += 1
        if parent is None and children > 1:
            articulation.add(cell)

    for cell in adjacency:
        if cell not in visited:
            dfs(cell, None)
    return articulation, bridges


def corridor_lengths(adjacency: dict[Cell, list[Cell]]) -> list[int]:
    if not adjacency:
        return []
    visited_edges: set[tuple[Cell, Cell]] = set()
    lengths: list[int] = []

    def walk_segment(start: Cell, next_cell: Cell) -> int:
        length = 0
        previous = start
        current = next_cell
        visited_edges.add(_edge(previous, current))
        while True:
            length += 1
            if len(adjacency[current]) != 2:
                return length
            candidates = [cell for cell in adjacency[current] if cell != previous]
            if not candidates:
                return length
            candidate = candidates[0]
            edge = _edge(current, candidate)
            if edge in visited_edges:
                return length
            previous, current = current, candidate
            visited_edges.add(edge)

    endpoints = [cell for cell, neighbors in adjacency.items() if len(neighbors) != 2]
    for cell in endpoints:
        for neighbor in adjacency[cell]:
            edge = _edge(cell, neighbor)
            if edge not in visited_edges:
                lengths.append(walk_segment(cell, neighbor))

    for cell, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = _edge(cell, neighbor)
            if edge not in visited_edges:
                lengths.append(walk_segment(cell, neighbor))
    return lengths


def _edge(left: Cell, right: Cell) -> tuple[Cell, Cell]:
    return (left, right) if left <= right else (right, left)


def choose_farthest_free_pair(maze_map: MazeMap) -> tuple[Cell, Cell]:
    adjacency = build_adjacency(maze_map)
    components = connected_components(adjacency)
    if len(components) != 1:
        raise ValueError(f"maze_map must be connected, got {len(components)} components")
    _diameter, start, goal = diameter_and_endpoints(adjacency)
    return start, goal


def compute_maze_metrics(
    maze_map: MazeMap,
    *,
    start: Cell | None = None,
    goal: Cell | None = None,
) -> MazeMetrics:
    validate_rectangular_boundary(maze_map)
    normalized = normalize_maze_map(maze_map)
    rows = len(normalized)
    cols = len(normalized[0])
    adjacency = build_adjacency(normalized)
    if not adjacency:
        raise ValueError("maze_map must contain at least one free cell")
    components = connected_components(adjacency)
    if len(components) != 1:
        raise ValueError(f"maze_map must be connected, got {len(components)} components")

    if start is None or goal is None:
        diameter, auto_start, auto_goal = diameter_and_endpoints(adjacency)
        start = auto_start if start is None else start
        goal = auto_goal if goal is None else goal
    else:
        diameter, _diameter_start, _diameter_goal = diameter_and_endpoints(adjacency)

    path = shortest_path(adjacency, start, goal)
    path_len = len(path) - 1
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    degrees = {cell: len(neighbors) for cell, neighbors in adjacency.items()}
    deadends = sum(1 for degree in degrees.values() if degree == 1)
    corridors = sum(1 for degree in degrees.values() if degree == 2)
    forks = sum(1 for degree in degrees.values() if degree == 3)
    crosses = sum(1 for degree in degrees.values() if degree >= 4)
    junction_count = forks + crosses
    turns = count_solution_turns(path)
    solution_junctions = sum(1 for cell in path if degrees[cell] >= 3)
    articulation, bridges = articulation_points_and_bridges(adjacency)
    corr_lengths = corridor_lengths(adjacency)
    free_count = len(adjacency)
    total_cells = rows * cols
    wall_cells = total_cells - free_count
    cycle_rank = edge_count - free_count + len(components)
    static_difficulty = _static_difficulty(
        rows=rows,
        cols=cols,
        free_cells=free_count,
        edge_count=edge_count,
        wall_density=wall_cells / total_cells,
        shortest_path_len=path_len,
        solution_turns=turns,
        solution_junctions=solution_junctions,
        deadends=deadends,
        cycle_rank=cycle_rank,
        articulation_points=len(articulation),
        bridge_edges=len(bridges),
    )

    return MazeMetrics(
        rows=rows,
        cols=cols,
        total_cells=total_cells,
        free_cells=free_count,
        wall_cells=wall_cells,
        wall_density=wall_cells / total_cells,
        edge_count=edge_count,
        connected_components=len(components),
        start=start,
        goal=goal,
        shortest_path_len=path_len,
        normalized_path_len=path_len / max(free_count - 1, 1),
        diameter=diameter,
        solution_turns=turns,
        solution_turn_rate=turns / max(path_len - 1, 1),
        solution_junctions=solution_junctions,
        degree_1_deadends=deadends,
        degree_2_corridors=corridors,
        degree_3_forks=forks,
        degree_4_crosses=crosses,
        junction_count=junction_count,
        cycle_rank=cycle_rank,
        articulation_points=len(articulation),
        bridge_edges=len(bridges),
        corridor_count=len(corr_lengths),
        mean_corridor_len=float(mean(corr_lengths)) if corr_lengths else 0.0,
        max_corridor_len=max(corr_lengths) if corr_lengths else 0,
        static_difficulty=static_difficulty,
    )


def metrics_for_variant(meta: dict) -> MazeMetrics:
    eval_map = (meta.get("env_paras") or meta.get("env_kwargs") or {}).get("maze_map")
    maze_map = eval_map if eval_map is not None else meta["prompt_vars"]["maze_map"]
    start = find_marker(eval_map, "r") if eval_map is not None else None
    goal = find_marker(eval_map, "g") if eval_map is not None else None
    return compute_maze_metrics(maze_map, start=start, goal=goal)


def _static_difficulty(
    *,
    rows: int,
    cols: int,
    free_cells: int,
    edge_count: int,
    wall_density: float,
    shortest_path_len: int,
    solution_turns: int,
    solution_junctions: int,
    deadends: int,
    cycle_rank: int,
    articulation_points: int,
    bridge_edges: int,
) -> float:
    direct_scale = max(rows + cols - 4, 1)
    path_component = _clamp01(shortest_path_len / (2.5 * direct_scale))
    turn_component = _clamp01((solution_turns / max(shortest_path_len, 1)) / 0.45)
    solution_junction_component = _clamp01(
        (solution_junctions / max(shortest_path_len + 1, 1)) / 0.25
    )
    articulation_component = _clamp01((articulation_points / max(free_cells, 1)) / 0.30)
    deadend_component = _clamp01((deadends / max(free_cells, 1)) / 0.25)
    bridge_component = _clamp01((bridge_edges / max(edge_count, 1)) / 0.60)
    wall_component = _clamp01(wall_density / 0.65)
    cycle_component = _clamp01((cycle_rank / max(free_cells, 1)) / 0.20)
    score = (
        0.28 * path_component
        + 0.14 * turn_component
        + 0.14 * solution_junction_component
        + 0.12 * articulation_component
        + 0.10 * deadend_component
        + 0.10 * bridge_component
        + 0.07 * wall_component
        + 0.05 * cycle_component
    )
    return round(100.0 * score, 2)


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))
