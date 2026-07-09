"""Canonical evaluation start/goal selection for CrossMaze variants."""

from __future__ import annotations

import hashlib
import random
from functools import lru_cache
from typing import Any


def _clean_collection_map(maze_map: list[list[object]]) -> list[list[int]]:
    return [[1 if cell == 1 else 0 for cell in row] for row in maze_map]


def _free_cells(maze_map: list[list[object]]) -> list[tuple[int, int]]:
    return [
        (row_idx, col_idx)
        for row_idx, row in enumerate(maze_map)
        for col_idx, cell in enumerate(row)
        if cell != 1
    ]


def _cell_neighbors(
    cell: tuple[int, int],
    free_cell_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    row, col = cell
    candidates = (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    )
    return [candidate for candidate in candidates if candidate in free_cell_set]


def _shortest_path_parents(
    start: tuple[int, int],
    free_cell_set: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, int] | None]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = [start]
    for cell in queue:
        for neighbor in _cell_neighbors(cell, free_cell_set):
            if neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    return parent


def _path_from_parents(
    goal: tuple[int, int],
    parent: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]] | None:
    if goal not in parent:
        return None

    path = []
    cell: tuple[int, int] | None = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return path


def _path_away_steps(
    path: list[tuple[int, int]],
    goal: tuple[int, int],
) -> int:
    if len(path) < 2:
        return 0

    def manhattan(cell: tuple[int, int]) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    return sum(
        1
        for current, next_cell in zip(path, path[1:])
        if manhattan(next_cell) > manhattan(current)
    )


def build_hard_start_goal_pair_space(
    maze_map: list[list[object]],
    candidate_cells: list[tuple[int, int]],
    hard_sample_alpha: float,
    hard_sample_top_n: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Build the AntMaze hard-sampling pair space and difficulty metrics.

    The difficulty definition is shared by local AntMaze data generation and
    fixed/hard eval-position tables.
    """
    clean_map = _clean_collection_map(maze_map)
    free_cell_set = set(_free_cells(clean_map))
    candidates = sorted(
        {
            (int(cell[0]), int(cell[1]))
            for cell in candidate_cells
        }
    )
    if len(candidates) < 2:
        raise ValueError("--hard-sample requires at least two candidate free cells")

    invalid_cells = [cell for cell in candidates if cell not in free_cell_set]
    if invalid_cells:
        raise ValueError(f"Hard-sample candidate cells are not free: {invalid_cells}")

    records: list[dict[str, Any]] = []
    max_path_len = 0
    for start_cell in candidates:
        parent = _shortest_path_parents(start_cell, free_cell_set)
        for goal_cell in candidates:
            if start_cell == goal_cell:
                continue
            path = _path_from_parents(goal_cell, parent)
            if path is None:
                continue
            path_len = len(path) - 1
            away_steps = _path_away_steps(path, goal_cell)
            records.append(
                {
                    "start_cell": start_cell,
                    "goal_cell": goal_cell,
                    "path_len": int(path_len),
                    "away_steps": int(away_steps),
                    "away_frac": float(away_steps / max(path_len, 1)),
                }
            )
            max_path_len = max(max_path_len, path_len)

    if not records:
        raise ValueError("--hard-sample found no reachable ordered start/goal pairs")
    if max_path_len <= 0:
        raise ValueError("--hard-sample pair paths must have positive length")

    for record in records:
        path_score = float(record["path_len"] / max_path_len)
        difficulty = 0.5 * path_score + 0.5 * float(record["away_frac"])
        record["path_score"] = path_score
        record["difficulty"] = float(difficulty)

    records = sorted(
        records,
        key=lambda record: (
            float(record["difficulty"]),
            int(record["path_len"]),
            int(record["away_steps"]),
            record["start_cell"],
            record["goal_cell"],
        ),
    )
    total_reachable_pairs = len(records)
    if hard_sample_top_n > 0:
        records = records[-int(hard_sample_top_n):]

    rank_denominator = max(len(records) - 1, 1)
    for rank, record in enumerate(records):
        rank_score = float(rank / rank_denominator)
        record["rank_score"] = rank_score
        record["sample_weight"] = float(1.0 + float(hard_sample_alpha) * rank_score)
    return records, total_reachable_pairs


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _normalize_eval_seed(seed: int | None) -> int:
    return 0 if seed is None else int(seed)


def _public_pair_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_cell": [int(record["start_cell"][0]), int(record["start_cell"][1])],
        "goal_cell": [int(record["goal_cell"][0]), int(record["goal_cell"][1])],
        "difficulty": float(record["difficulty"]),
    }


def _find_pair_record(
    records: list[dict[str, Any]],
    *,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> dict[str, Any]:
    for record in records:
        if record["start_cell"] == start_cell and record["goal_cell"] == goal_cell:
            return record
    raise ValueError(f"Eval start/goal pair is not reachable: {start_cell}->{goal_cell}")


def _build_antmaze_eval_positions() -> dict[str, dict[str, Any]]:
    from crossmaze.variants import ANTMAZE_ENV_FACTS

    fixed_cells = {
        "umaze": ((1, 1), (3, 1)),
        "umaze-diverse": ((1, 1), (3, 1)),
        "medium-play": ((1, 1), (6, 6)),
        "medium-diverse": ((1, 1), (6, 6)),
        "large-play": ((1, 1), (7, 9)),
        "large-diverse": ((1, 1), (7, 9)),
        "local-layout-01": ((1, 1), (7, 11)),
        "local-layout-02": ((7, 1), (5, 11)),
        "local-layout-03": ((7, 1), (1, 11)),
        "local-layout-04": ((7, 1), (1, 11)),
        "local-layout-05": ((1, 5), (7, 11)),
        "local-layout-06": ((3, 1), (7, 11)),
        "local-layout-07": ((1, 11), (1, 1)),
        "local-layout-08": ((1, 11), (7, 9)),
        "local-layout-09": ((6, 11), (1, 1)),
        "test-layout-01": ((5, 1), (1, 11)),
        "test-layout-02": ((7, 1), (1, 11)),
        "test-layout-03": ((7, 1), (7, 11)),
        "test-layout-04": ((7, 1), (5, 7)),
        "ultra": ((1, 1), (10, 14)),
    }
    missing = set(ANTMAZE_ENV_FACTS) - set(fixed_cells)
    extra = set(fixed_cells) - set(ANTMAZE_ENV_FACTS)
    if missing or extra:
        raise ValueError(
            f"AntMaze eval position table mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    positions: dict[str, dict[str, Any]] = {}
    for variant, facts in ANTMAZE_ENV_FACTS.items():
        start_cell, goal_cell = fixed_cells[variant]
        maze_map = facts["maze_map"]
        records, _ = build_hard_start_goal_pair_space(
            maze_map,
            _free_cells(_clean_collection_map(maze_map)),
            hard_sample_alpha=0.0,
        )
        positions[variant] = {
            "fix_start_goal": _public_pair_record(
                _find_pair_record(records, start_cell=start_cell, goal_cell=goal_cell)
            )
        }
    return positions


def _build_pointmaze_eval_positions(seed: int | None) -> dict[str, dict[str, Any]]:
    from crossmaze.variants import POINTMAZE_ENV_FACTS

    eval_seed = _normalize_eval_seed(seed)
    positions: dict[str, dict[str, Any]] = {}
    for variant, facts in POINTMAZE_ENV_FACTS.items():
        maze_map = facts["maze_map"]
        records, _ = build_hard_start_goal_pair_space(
            maze_map,
            _free_cells(_clean_collection_map(maze_map)),
            hard_sample_alpha=0.0,
        )
        hard_pool_size = min(400, len(records))
        hard_pool = records[-hard_pool_size:]
        sample_size = min(100, hard_pool_size)
        rng = random.Random(
            _stable_seed(f"pointmaze-eval-position-v1:{variant}:{eval_seed}")
        )
        selected_records = rng.sample(hard_pool, sample_size)
        positions[variant] = {
            "start_goal_list": [
                _public_pair_record(record)
                for record in selected_records
            ]
        }
    return positions


@lru_cache(maxsize=None)
def _pointmaze_eval_positions_for_seed(seed: int) -> dict[str, dict[str, Any]]:
    return _build_pointmaze_eval_positions(seed)


EVAL_POSITIONS: dict[str, dict[str, dict[str, Any]]] = {
    "antmaze": _build_antmaze_eval_positions(),
    "pointmaze": _pointmaze_eval_positions_for_seed(0),
}


def get_eval_position_config(
    env_family: str,
    variant: str,
    seed: int | None = None,
) -> dict[str, Any] | None:
    if env_family == "pointmaze":
        return _pointmaze_eval_positions_for_seed(_normalize_eval_seed(seed)).get(variant)
    return EVAL_POSITIONS.get(env_family, {}).get(variant)


def eval_position_count(env_family: str, variant: str) -> int:
    config = get_eval_position_config(env_family, variant)
    if config is None:
        return 0
    if "fix_start_goal" in config:
        return 1
    if "start_goal_list" in config:
        return len(config["start_goal_list"])
    return 0


def eval_position_selection_policy(env_family: str, variant: str) -> str:
    config = get_eval_position_config(env_family, variant)
    if config is None:
        return "env_default_random"
    if "fix_start_goal" in config:
        return "fixed"
    if "start_goal_list" in config:
        return "seeded_permutation_cycle"
    return "env_default_random"


def _list_record_index(
    *,
    env_family: str,
    variant: str,
    count: int,
    episode_index: int,
    seed: int | None,
) -> int:
    if episode_index < 0:
        raise ValueError(f"episode_index must be >= 0, got {episode_index}")
    order = list(range(count))
    rng = random.Random(
        _stable_seed(
            f"eval-position-selection-v1:{env_family}:{variant}:{int(seed or 0)}"
        )
    )
    rng.shuffle(order)
    return int(order[int(episode_index) % count])


def select_eval_position(
    env_family: str,
    variant: str,
    episode_index: int | None,
    seed: int | None,
) -> dict[str, Any] | None:
    """Select the eval start/goal record for one episode.

    For `start_goal_list` variants, `seed` is the run-level eval seed rather
    than the per-episode reset seed.
    """
    config = get_eval_position_config(env_family, variant, seed=seed)
    if config is None:
        return None
    if "fix_start_goal" in config:
        record = dict(config["fix_start_goal"])
        record["source"] = "fix_start_goal"
        record["index"] = 0
        return record
    if "start_goal_list" in config:
        records = config["start_goal_list"]
        if not records:
            return None
        if episode_index is None:
            return None
        index = _list_record_index(
            env_family=env_family,
            variant=variant,
            count=len(records),
            episode_index=int(episode_index),
            seed=seed,
        )
        record = dict(records[index])
        record["source"] = "start_goal_list"
        record["index"] = int(index)
        return record
    return None


def eval_reset_options(
    env_family: str,
    variant: str,
    episode_index: int | None = None,
    seed: int | None = None,
) -> dict[str, list[int]] | None:
    """Return `reset(options=...)` cells for the selected eval position."""
    record = select_eval_position(env_family, variant, episode_index, seed)
    if record is None:
        return None
    return {
        "reset_cell": [int(record["start_cell"][0]), int(record["start_cell"][1])],
        "goal_cell": [int(record["goal_cell"][0]), int(record["goal_cell"][1])],
    }
