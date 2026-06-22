"""Generate AntMaze grid layouts with design-centric topology targets."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from utils.maze_metrics import (
    Cell,
    choose_farthest_free_pair,
    compute_maze_metrics,
    format_maze_map,
)


@dataclass(frozen=True)
class Profile:
    name: str
    row_range: tuple[int, int]
    col_range: tuple[int, int]
    free_density_range: tuple[float, float]
    loop_chance_range: tuple[float, float]
    shortest_path_range: tuple[int, int]
    static_difficulty_range: tuple[float, float]
    min_deadends: int
    min_junctions: int
    max_open_2x2: int


PROFILES = {
    "large-like": Profile(
        name="large-like",
        row_range=(9, 11),
        col_range=(12, 14),
        free_density_range=(0.42, 0.58),
        loop_chance_range=(0.12, 0.32),
        shortest_path_range=(18, 42),
        static_difficulty_range=(30.0, 58.0),
        min_deadends=3,
        min_junctions=8,
        max_open_2x2=2,
    ),
    "hard": Profile(
        name="hard",
        row_range=(13, 16),
        col_range=(14, 16),
        free_density_range=(0.42, 0.56),
        loop_chance_range=(0.04, 0.18),
        shortest_path_range=(35, 90),
        static_difficulty_range=(45.0, 78.0),
        min_deadends=5,
        min_junctions=14,
        max_open_2x2=0,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("suite", "candidates"),
        default="suite",
        help="suite generates 9 local + 4 test layouts; candidates generates generic layouts.",
    )
    parser.add_argument(
        "--profile",
        choices=("large-like", "hard", "mixed"),
        default="mixed",
        help="Profile used in candidates mode.",
    )
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--candidates-per-layout", type=int, default=500)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--python-output", type=Path, default=None)
    return parser.parse_args()


def _suite_specs() -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    specs.extend((f"local-layout-{idx:02d}", "large-like") for idx in range(1, 6))
    specs.extend((f"local-layout-{idx:02d}", "hard") for idx in range(6, 10))
    specs.extend((f"test-layout-{idx:02d}", "large-like") for idx in range(1, 3))
    specs.extend((f"test-layout-{idx:02d}", "hard") for idx in range(3, 5))
    return specs


def _candidate_specs(count: int, profile_name: str) -> list[tuple[str, str]]:
    if count < 1:
        raise ValueError("--count must be >= 1")
    if profile_name == "mixed":
        large_count = (count + 1) // 2
        return [
            (f"generated-layout-{idx:02d}", "large-like" if idx <= large_count else "hard")
            for idx in range(1, count + 1)
        ]
    return [
        (f"generated-layout-{idx:02d}", profile_name)
        for idx in range(1, count + 1)
    ]


def generate_candidate(profile: Profile, rng: random.Random) -> list[list[int]]:
    rows = rng.randint(*profile.row_range)
    cols = rng.randint(*profile.col_range)
    density = rng.uniform(*profile.free_density_range)
    loop_chance = rng.uniform(*profile.loop_chance_range)
    return _spanning_corridor_map(
        rows=rows,
        cols=cols,
        target_free_density=density,
        loop_chance=loop_chance,
        avoid_open_blocks=profile.max_open_2x2 == 0,
        rng=rng,
    )


def _spanning_corridor_map(
    *,
    rows: int,
    cols: int,
    target_free_density: float,
    loop_chance: float,
    avoid_open_blocks: bool,
    rng: random.Random,
) -> list[list[int]]:
    maze = [[1 for _col in range(cols)] for _row in range(rows)]
    room_rows = _room_coordinates(rows)
    room_cols = _room_coordinates(cols)
    rooms = [(row, col) for row in room_rows for col in room_cols]
    start = rng.choice(rooms)
    visited = {start}
    stack = [start]
    carved_edges: set[tuple[Cell, Cell]] = set()
    maze[start[0]][start[1]] = 0

    while stack:
        current = stack[-1]
        candidates = [
            neighbor
            for neighbor in _room_neighbors(current, room_rows, room_cols)
            if neighbor not in visited
        ]
        if not candidates:
            stack.pop()
            continue
        neighbor = rng.choice(candidates)
        _carve_room_edge(maze, current, neighbor)
        carved_edges.add(_edge(current, neighbor))
        visited.add(neighbor)
        stack.append(neighbor)

    for room in rooms:
        for neighbor in _room_neighbors(room, room_rows, room_cols):
            edge = _edge(room, neighbor)
            if edge in carved_edges:
                continue
            if rng.random() < loop_chance:
                _carve_room_edge(maze, room, neighbor)
                carved_edges.add(edge)

    target_free = round((rows - 2) * (cols - 2) * target_free_density)
    _add_pockets(
        maze,
        target_free=target_free,
        avoid_open_blocks=avoid_open_blocks,
        rng=rng,
    )
    return maze


def _room_coordinates(size: int) -> list[int]:
    return list(range(1, size - 1, 2))


def _room_neighbors(
    room: Cell,
    room_rows: list[int],
    room_cols: list[int],
) -> list[Cell]:
    row, col = room
    row_idx = room_rows.index(row)
    col_idx = room_cols.index(col)
    neighbors = []
    if row_idx > 0:
        neighbors.append((room_rows[row_idx - 1], col))
    if row_idx + 1 < len(room_rows):
        neighbors.append((room_rows[row_idx + 1], col))
    if col_idx > 0:
        neighbors.append((row, room_cols[col_idx - 1]))
    if col_idx + 1 < len(room_cols):
        neighbors.append((row, room_cols[col_idx + 1]))
    return neighbors


def _carve_room_edge(maze: list[list[int]], left: Cell, right: Cell) -> None:
    row_a, col_a = left
    row_b, col_b = right
    maze[row_a][col_a] = 0
    maze[row_b][col_b] = 0
    d_row = _sign(row_b - row_a)
    d_col = _sign(col_b - col_a)
    row, col = row_a, col_a
    while (row, col) != (row_b, col_b):
        row += d_row
        col += d_col
        maze[row][col] = 0


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _edge(left: Cell, right: Cell) -> tuple[Cell, Cell]:
    return (left, right) if left <= right else (right, left)


def _add_pockets(
    maze: list[list[int]],
    *,
    target_free: int,
    avoid_open_blocks: bool,
    rng: random.Random,
) -> None:
    rows = len(maze)
    cols = len(maze[0])
    free_count = _free_count(maze)
    while free_count < target_free:
        candidates = []
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if maze[row][col] == 0:
                    continue
                if avoid_open_blocks and _would_create_open_2x2(maze, row, col):
                    continue
                adjacent_free = sum(
                    1
                    for n_row, n_col in _interior_neighbors((row, col), rows, cols)
                    if maze[n_row][n_col] == 0
                )
                if adjacent_free > 0:
                    candidates.append((adjacent_free, row, col))
        if not candidates:
            return
        candidates.sort(reverse=True)
        top_k = candidates[: max(1, min(len(candidates), 12))]
        _adjacent_free, row, col = rng.choice(top_k)
        maze[row][col] = 0
        free_count += 1


def _would_create_open_2x2(maze: list[list[int]], row: int, col: int) -> bool:
    rows = len(maze)
    cols = len(maze[0])
    for top in (row - 1, row):
        for left in (col - 1, col):
            if top < 1 or left < 1 or top + 1 >= rows - 1 or left + 1 >= cols - 1:
                continue
            if all(
                (r == row and c == col) or maze[r][c] == 0
                for r in (top, top + 1)
                for c in (left, left + 1)
            ):
                return True
    return False


def _free_count(maze: list[list[int]]) -> int:
    return sum(1 for row in maze for cell in row if cell != 1)


def _interior_neighbors(cell: Cell, rows: int, cols: int) -> list[Cell]:
    row, col = cell
    neighbors = []
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = row + d_row, col + d_col
        if 1 <= neighbor[0] < rows - 1 and 1 <= neighbor[1] < cols - 1:
            neighbors.append(neighbor)
    return neighbors


def choose_layout(
    profile: Profile,
    *,
    rng: random.Random,
    candidates_per_layout: int,
) -> dict:
    best: tuple[float, dict] | None = None
    for _idx in range(candidates_per_layout):
        maze = generate_candidate(profile, rng)
        try:
            start, goal = choose_farthest_free_pair(maze)
            metrics = compute_maze_metrics(maze, start=start, goal=goal)
        except ValueError:
            continue
        penalty = _profile_penalty(metrics.to_dict(), profile)
        penalty += 2.5 * _coverage_penalty(maze)
        penalty += 8.0 * _open_space_penalty(maze, profile)
        if best is None or penalty < best[0]:
            best = (
                penalty,
                {
                    "maze": maze,
                    "start": start,
                    "goal": goal,
                    "metrics": metrics.to_dict(),
                },
            )
    if best is None:
        raise RuntimeError(f"Failed to generate a valid layout for profile={profile.name}")
    return best[1]


def _profile_penalty(metrics: dict, profile: Profile) -> float:
    penalty = 0.0
    penalty += 5.0 * _range_penalty(
        metrics["shortest_path_len"],
        profile.shortest_path_range,
    )
    penalty += 3.0 * _range_penalty(
        metrics["static_difficulty"],
        profile.static_difficulty_range,
    )
    if metrics["degree_1_deadends"] < profile.min_deadends:
        penalty += 2.0 * (profile.min_deadends - metrics["degree_1_deadends"])
    if metrics["junction_count"] < profile.min_junctions:
        penalty += 1.5 * (profile.min_junctions - metrics["junction_count"])
    penalty += 0.1 * abs(metrics["wall_density"] - 0.52)
    return penalty


def _coverage_penalty(maze: list[list[int]]) -> float:
    rows = len(maze)
    cols = len(maze[0])
    thin_rows = sum(
        1
        for row in range(1, rows - 1)
        if sum(1 for col in range(1, cols - 1) if maze[row][col] == 0) == 0
    )
    thin_cols = sum(
        1
        for col in range(1, cols - 1)
        if sum(1 for row in range(1, rows - 1) if maze[row][col] == 0) == 0
    )
    wall_windows = 0
    for row in range(1, rows - 3):
        for col in range(1, cols - 3):
            if all(maze[r][c] == 1 for r in range(row, row + 3) for c in range(col, col + 3)):
                wall_windows += 1
    return thin_rows + thin_cols + wall_windows


def _open_space_penalty(maze: list[list[int]], profile: Profile) -> float:
    open_2x2 = _free_window_count(maze, height=2, width=2)
    penalty = 8.0 * max(0, open_2x2 - profile.max_open_2x2)
    penalty += 2.0 * _free_window_count(maze, height=2, width=3)
    penalty += 2.0 * _free_window_count(maze, height=3, width=2)
    return penalty


def _free_window_count(maze: list[list[int]], *, height: int, width: int) -> int:
    rows = len(maze)
    cols = len(maze[0])
    count = 0
    for row in range(1, rows - height):
        for col in range(1, cols - width):
            if all(
                maze[r][c] == 0
                for r in range(row, row + height)
                for c in range(col, col + width)
            ):
                count += 1
    return count


def _range_penalty(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if value < low:
        return low - value
    if value > high:
        return value - high
    center = (low + high) / 2
    half_width = max((high - low) / 2, 1e-6)
    return 0.05 * abs(value - center) / half_width


def _emit_layouts(specs: list[tuple[str, str]], args) -> list[dict]:
    rng = random.Random(args.seed)
    layouts = []
    seen_maps: set[str] = set()
    for name, profile_name in specs:
        profile = PROFILES[profile_name]
        for attempt in range(20):
            layout = choose_layout(
                profile,
                rng=rng,
                candidates_per_layout=args.candidates_per_layout,
            )
            text_map = format_maze_map(layout["maze"])
            if text_map not in seen_maps or attempt == 19:
                seen_maps.add(text_map)
                break
        layout["name"] = name
        layout["profile"] = profile_name
        layout["map_rows"] = text_map.splitlines()
        layouts.append(layout)
    return layouts


def _print_layouts(layouts: list[dict]) -> None:
    for layout in layouts:
        metrics = layout["metrics"]
        print(
            f"{layout['name']} profile={layout['profile']} "
            f"size={metrics['rows']}x{metrics['cols']} "
            f"free={metrics['free_cells']} path={metrics['shortest_path_len']} "
            f"turns={metrics['solution_turns']} junctions={metrics['junction_count']} "
            f"deadends={metrics['degree_1_deadends']} "
            f"score={metrics['static_difficulty']}"
        )
        print(format_maze_map(layout["maze"]))
        print(f"eval_reset_cell={tuple(layout['start'])} eval_goal_cell={tuple(layout['goal'])}")
        print()


def _write_json(path: Path, layouts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for layout in layouts:
        serializable.append(
            {
                "name": layout["name"],
                "profile": layout["profile"],
                "map_rows": layout["map_rows"],
                "eval_reset_cell": list(layout["start"]),
                "eval_goal_cell": list(layout["goal"]),
                "metrics": layout["metrics"],
            }
        )
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_python(path: Path, layouts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by generate_antmaze_layouts.py",
        "# Review these candidates before copying them into data/antmaze/variants.py.",
        "",
    ]
    for layout in layouts:
        const_name = "_" + layout["name"].upper().replace("-", "_")
        lines.append(f"{const_name} = _maze_from_strings([")
        for row in layout["map_rows"]:
            lines.append(f"    {row!r},")
        lines.append("])")
        lines.append(
            "# "
            f"profile={layout['profile']} "
            f"reset={tuple(layout['start'])} goal={tuple(layout['goal'])} "
            f"score={layout['metrics']['static_difficulty']} "
            f"path={layout['metrics']['shortest_path_len']}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.candidates_per_layout < 1:
        raise ValueError("--candidates-per-layout must be >= 1")
    specs = (
        _suite_specs()
        if args.mode == "suite"
        else _candidate_specs(args.count, args.profile)
    )
    layouts = _emit_layouts(specs, args)
    _print_layouts(layouts)
    if args.json_output is not None:
        _write_json(args.json_output, layouts)
    if args.python_output is not None:
        _write_python(args.python_output, layouts)


if __name__ == "__main__":
    main()
