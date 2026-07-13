"""Canonical evaluation start/goal selection for CrossMaze variants."""

from __future__ import annotations

import hashlib
import math
import random
from functools import lru_cache
from typing import Any


FIX_START_GOAL_MODE = "fix-start-goal"
HARD_SAMPLE_MODE = "hard-sample"
RANDOM_START_GOAL_MODE = "random-start-goal"

_EVAL_POSITION_MODES = {
    FIX_START_GOAL_MODE,
    HARD_SAMPLE_MODE,
    RANDOM_START_GOAL_MODE,
}

_DEFAULT_EVAL_POSITION_MODE_BY_FAMILY = {
    "antmaze": FIX_START_GOAL_MODE,
    "pointmaze": RANDOM_START_GOAL_MODE,
}

PATH_DIFFICULTY_VERSION = "v2"
PATH_LENGTH_SCALE = 20.0
PATH_LENGTH_WEIGHT = 0.4
PATH_BRANCH_WEIGHT = 0.3
PATH_DETOUR_WEIGHT = 0.3
MAP_DIFFICULTY_TOP_FRACTION = 0.10


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


def _shortest_path_tree(
    start: tuple[int, int],
    free_cell_set: set[tuple[int, int]],
) -> tuple[
    dict[tuple[int, int], tuple[int, int] | None],
    dict[tuple[int, int], int],
]:
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    distance = {start: 0}
    queue = [start]
    for cell in queue:
        for neighbor in _cell_neighbors(cell, free_cell_set):
            if neighbor not in parent:
                parent[neighbor] = cell
                distance[neighbor] = distance[cell] + 1
                queue.append(neighbor)
    return parent, distance


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


def _path_distractor_metrics(
    path: list[tuple[int, int]],
    goal_distances: dict[tuple[int, int], int],
    free_cell_set: set[tuple[int, int]],
) -> tuple[int, int, float]:
    """Count suboptimal non-backtracking exits along one canonical path."""
    path_len = len(path) - 1
    if path_len <= 0:
        return 0, 0, 0.0

    distractor_point_count = 0
    distractor_exit_count = 0
    normalized_sum = 0.0
    for path_index, cell in enumerate(path[:-1]):
        previous = path[path_index - 1] if path_index > 0 else None
        remaining_distance = path_len - path_index
        distractor_count = 0
        for neighbor in _cell_neighbors(cell, free_cell_set):
            if neighbor == previous:
                continue
            if goal_distances.get(neighbor) == remaining_distance - 1:
                continue
            distractor_count += 1

        if distractor_count > 0:
            distractor_point_count += 1
            distractor_exit_count += distractor_count
        max_distractors = 3 if path_index == 0 else 2
        normalized_sum += float(distractor_count / max_distractors)

    branch_score = float(normalized_sum / path_len)
    return distractor_point_count, distractor_exit_count, branch_score


def path_difficulty_config() -> dict[str, Any]:
    """Return the versioned constants used by every path-difficulty consumer."""
    return {
        "version": PATH_DIFFICULTY_VERSION,
        "length_scale": float(PATH_LENGTH_SCALE),
        "weights": {
            "length": float(PATH_LENGTH_WEIGHT),
            "branch": float(PATH_BRANCH_WEIGHT),
            "detour": float(PATH_DETOUR_WEIGHT),
        },
        "map_top_fraction": float(MAP_DIFFICULTY_TOP_FRACTION),
    }


def _eval_pair_difficulty_components(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": PATH_DIFFICULTY_VERSION,
        "path_len": int(record["path_len"]),
        "manhattan_distance": int(record["manhattan_distance"]),
        "length_scale": float(PATH_LENGTH_SCALE),
        "length_score": float(record["length_score"]),
        "distractor_point_count": int(record["distractor_point_count"]),
        "distractor_exit_count": int(record["distractor_exit_count"]),
        "branch_score": float(record["branch_score"]),
        "away_steps": int(record["away_steps"]),
        "away_frac": float(record["away_frac"]),
        "detour_score": float(record["detour_score"]),
        "weights": {
            "length": float(PATH_LENGTH_WEIGHT),
            "branch": float(PATH_BRANCH_WEIGHT),
            "detour": float(PATH_DETOUR_WEIGHT),
        },
    }


def _pair_space_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "difficulty_version": PATH_DIFFICULTY_VERSION,
        "difficulty_config": path_difficulty_config(),
        "map_difficulty": float(record["map_difficulty"]),
        "map_difficulty_top_fraction": float(MAP_DIFFICULTY_TOP_FRACTION),
        "map_difficulty_path_count": int(record["map_difficulty_path_count"]),
        "map_reachable_pair_count": int(record["map_reachable_pair_count"]),
        "map_diameter": int(record["map_diameter"]),
    }


def _build_eval_pair_record(
    *,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    parent: dict[tuple[int, int], tuple[int, int] | None],
    start_distances: dict[tuple[int, int], int],
    goal_distances: dict[tuple[int, int], int],
    free_cell_set: set[tuple[int, int]],
) -> dict[str, Any] | None:
    path = _path_from_parents(goal_cell, parent)
    if path is None:
        return None
    path_len = int(start_distances[goal_cell])
    away_steps = _path_away_steps(path, goal_cell)
    manhattan_distance = int(
        abs(start_cell[0] - goal_cell[0])
        + abs(start_cell[1] - goal_cell[1])
    )
    (
        distractor_point_count,
        distractor_exit_count,
        branch_score,
    ) = _path_distractor_metrics(
        path,
        goal_distances,
        free_cell_set,
    )
    length_score = float(path_len / (path_len + PATH_LENGTH_SCALE))
    detour_score = float(1.0 - manhattan_distance / path_len)
    difficulty = float(
        PATH_LENGTH_WEIGHT * length_score
        + PATH_BRANCH_WEIGHT * branch_score
        + PATH_DETOUR_WEIGHT * detour_score
    )
    return {
        "start_cell": start_cell,
        "goal_cell": goal_cell,
        "path_len": int(path_len),
        "manhattan_distance": manhattan_distance,
        "length_score": length_score,
        "distractor_point_count": int(distractor_point_count),
        "distractor_exit_count": int(distractor_exit_count),
        "branch_score": branch_score,
        "away_steps": int(away_steps),
        "away_frac": float(away_steps / max(path_len, 1)),
        "detour_score": detour_score,
        "difficulty": difficulty,
    }


def build_hard_start_goal_pair_space(
    maze_map: list[list[object]],
    candidate_cells: list[tuple[int, int]],
    hard_sample_alpha: float,
    hard_sample_top_n: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Build the legacy local AntMaze data-generation hard pair space.

    Eval path/map difficulty is deliberately separate and is implemented by
    `build_eval_start_goal_pair_space` below.
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
        record["sample_weight"] = float(
            1.0 + float(hard_sample_alpha) * rank_score
        )
    return records, total_reachable_pairs


def build_eval_start_goal_pair_space(
    maze_map: list[list[object]],
    candidate_cells: list[tuple[int, int]],
    hard_sample_alpha: float,
    hard_sample_top_n: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Build eval ordered pairs with canonical v2 path/map difficulty."""
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

    shortest_path_trees = {
        cell: _shortest_path_tree(cell, free_cell_set)
        for cell in candidates
    }
    records: list[dict[str, Any]] = []
    map_diameter = 0
    for start_cell in candidates:
        parent, start_distances = shortest_path_trees[start_cell]
        for goal_cell in candidates:
            if start_cell == goal_cell:
                continue
            goal_distances = shortest_path_trees[goal_cell][1]
            record = _build_eval_pair_record(
                start_cell=start_cell,
                goal_cell=goal_cell,
                parent=parent,
                start_distances=start_distances,
                goal_distances=goal_distances,
                free_cell_set=free_cell_set,
            )
            if record is None:
                continue
            records.append(record)
            map_diameter = max(map_diameter, int(record["path_len"]))

    if not records:
        raise ValueError("--hard-sample found no reachable ordered start/goal pairs")
    if map_diameter <= 0:
        raise ValueError("--hard-sample pair paths must have positive length")

    records = sorted(
        records,
        key=lambda record: (
            float(record["difficulty"]),
            int(record["path_len"]),
            int(record["distractor_exit_count"]),
            int(record["away_steps"]),
            record["start_cell"],
            record["goal_cell"],
        ),
    )
    total_reachable_pairs = len(records)
    map_difficulty_path_count = max(
        1,
        int(math.ceil(total_reachable_pairs * MAP_DIFFICULTY_TOP_FRACTION)),
    )
    map_difficulty = float(
        sum(
            float(record["difficulty"])
            for record in records[-map_difficulty_path_count:]
        )
        / map_difficulty_path_count
    )
    for record in records:
        record["map_diameter"] = int(map_diameter)
        record["path_score"] = float(record["path_len"] / map_diameter)
        record["map_difficulty"] = map_difficulty
        record["map_difficulty_path_count"] = int(map_difficulty_path_count)
        record["map_reachable_pair_count"] = int(total_reachable_pairs)

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
        "difficulty_components": _eval_pair_difficulty_components(record),
    }


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
        clean_map = _clean_collection_map(facts["maze_map"])
        free_cell_set = set(_free_cells(clean_map))
        parent, start_distances = _shortest_path_tree(start_cell, free_cell_set)
        _, goal_distances = _shortest_path_tree(goal_cell, free_cell_set)
        record = _build_eval_pair_record(
            start_cell=start_cell,
            goal_cell=goal_cell,
            parent=parent,
            start_distances=start_distances,
            goal_distances=goal_distances,
            free_cell_set=free_cell_set,
        )
        if record is None:
            raise ValueError(f"Eval start/goal pair is not reachable: {variant}")
        positions[variant] = {
            "difficulty_version": PATH_DIFFICULTY_VERSION,
            "difficulty_config": path_difficulty_config(),
            "fix_start_goal": _public_pair_record(record),
        }
    return positions


def _normalize_eval_position_mode(mode: object) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized not in _EVAL_POSITION_MODES:
        raise ValueError(
            "eval_start_goal_mode must be one of "
            f"{sorted(_EVAL_POSITION_MODES)}, got {mode!r}"
        )
    return normalized


def resolve_eval_position_mode(env_family: str, config: dict | None = None) -> str:
    config = config or {}
    raw_mode = config.get("eval_start_goal_mode")
    alias_mode = config.get("eval_position_mode")
    if raw_mode is not None and alias_mode is not None:
        if _normalize_eval_position_mode(raw_mode) != _normalize_eval_position_mode(alias_mode):
            raise ValueError(
                "eval_start_goal_mode and eval_position_mode specify different modes: "
                f"{raw_mode!r} vs {alias_mode!r}"
            )
    mode = (
        _normalize_eval_position_mode(raw_mode if raw_mode is not None else alias_mode)
        if raw_mode is not None or alias_mode is not None
        else _DEFAULT_EVAL_POSITION_MODE_BY_FAMILY.get(
            env_family,
            RANDOM_START_GOAL_MODE,
        )
    )
    if mode == FIX_START_GOAL_MODE and env_family != "antmaze":
        raise ValueError("fix-start-goal eval mode currently supports only antmaze")
    return mode


def _config_value(config: dict, *keys: str):
    for key in keys:
        if key in config:
            return config[key]
    return None


def _normalize_top_percent(value: object) -> float:
    top_percent = float(value)
    if not math.isfinite(top_percent) or top_percent <= 0.0:
        raise ValueError("eval_hard_sample_top_percent must be > 0")
    if top_percent > 1.0:
        if top_percent > 100.0:
            raise ValueError("eval_hard_sample_top_percent must be <= 1.0 or <= 100")
        top_percent = top_percent / 100.0
    return float(top_percent)


def _resolve_hard_sample_options(config: dict | None) -> dict[str, Any]:
    config = config or {}
    raw_top_percent = _config_value(
        config,
        "eval_hard_sample_top_percent",
        "eval_position_hard_sample_top_percent",
    )
    raw_top_n = _config_value(
        config,
        "eval_hard_sample_top_n",
        "eval_position_hard_sample_top_n",
    )
    has_top_percent = raw_top_percent is not None
    has_top_n = raw_top_n is not None
    if has_top_percent == has_top_n:
        raise ValueError(
            "hard-sample eval mode requires exactly one of "
            "eval_hard_sample_top_percent or eval_hard_sample_top_n"
        )

    top_percent = _normalize_top_percent(raw_top_percent) if has_top_percent else None
    top_n = None
    if has_top_n:
        top_n = int(raw_top_n)
        if top_n <= 0:
            raise ValueError("eval_hard_sample_top_n must be > 0")

    raw_alpha = _config_value(
        config,
        "eval_hard_sample_alpha",
        "eval_position_hard_sample_alpha",
        "hard_sample_alpha",
    )
    alpha = 0.0 if raw_alpha is None else float(raw_alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("eval_hard_sample_alpha must be >= 0")

    return {
        "top_percent": top_percent,
        "top_n": top_n,
        "alpha": alpha,
    }


def _hard_sample_cache_key(config: dict | None) -> tuple[float | None, int | None, float]:
    options = _resolve_hard_sample_options(config)
    return (
        options["top_percent"],
        options["top_n"],
        options["alpha"],
    )


def _hard_sample_pool_size(
    total_pairs: int,
    *,
    top_percent: float | None,
    top_n: int | None,
) -> int:
    if total_pairs <= 0:
        return 0
    if top_n is not None:
        return min(int(top_n), int(total_pairs))
    if top_percent is None:
        raise ValueError("top_percent is required when top_n is not set")
    return max(1, int(math.ceil(float(total_pairs) * float(top_percent))))


def _add_rank_sample_weights(
    records: list[dict[str, Any]],
    hard_sample_alpha: float,
) -> list[dict[str, Any]]:
    rank_denominator = max(len(records) - 1, 1)
    weighted_records = []
    for rank, record in enumerate(records):
        copied = dict(record)
        rank_score = float(rank / rank_denominator)
        copied["rank_score"] = rank_score
        copied["sample_weight"] = float(1.0 + float(hard_sample_alpha) * rank_score)
        weighted_records.append(copied)
    return weighted_records


def _weighted_sample_without_replacement(
    records: list[dict[str, Any]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pool = list(records)
    weights = [float(record.get("sample_weight", 1.0)) for record in pool]
    selected = []
    for _ in range(min(int(sample_size), len(pool))):
        total = sum(weight for weight in weights if weight > 0.0)
        if total <= 0.0 or not math.isfinite(total):
            index = rng.randrange(len(pool))
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            index = len(pool) - 1
            for candidate_idx, weight in enumerate(weights):
                cumulative += max(float(weight), 0.0)
                if cumulative >= threshold:
                    index = candidate_idx
                    break
        selected.append(pool.pop(index))
        weights.pop(index)
    return selected


@lru_cache(maxsize=4)
def _eval_pair_space_for_variant(
    env_family: str,
    variant: str,
) -> tuple[list[dict[str, Any]], int] | None:
    from crossmaze.variants import ENV_FACTS

    facts = ENV_FACTS.get(env_family, {}).get(variant)
    if facts is None:
        return None
    maze_map = facts["maze_map"]
    return build_eval_start_goal_pair_space(
        maze_map,
        _free_cells(_clean_collection_map(maze_map)),
        hard_sample_alpha=0.0,
    )


@lru_cache(maxsize=None)
def _hard_sample_eval_position_for_settings(
    env_family: str,
    variant: str,
    seed: int,
    top_percent: float | None,
    top_n: int | None,
    hard_sample_alpha: float,
) -> dict[str, Any] | None:
    pair_space = _eval_pair_space_for_variant(env_family, variant)
    if pair_space is None:
        return None

    eval_seed = _normalize_eval_seed(seed)
    records, _ = pair_space
    hard_pool_size = _hard_sample_pool_size(
        len(records),
        top_percent=top_percent,
        top_n=top_n,
    )
    hard_pool = _add_rank_sample_weights(
        records[-hard_pool_size:],
        hard_sample_alpha,
    )
    sample_size = min(100, hard_pool_size)
    rng = random.Random(
        _stable_seed(
            "eval-position-hard-sample-v2:"
            f"{env_family}:{variant}:{eval_seed}:{top_percent}:{top_n}:"
            f"{hard_sample_alpha}"
        )
    )
    selected_records = _weighted_sample_without_replacement(hard_pool, sample_size, rng)
    return {
        **_pair_space_metadata(records[0]),
        "start_goal_list": [
            _public_pair_record(record)
            for record in selected_records
        ]
    }


EVAL_POSITIONS: dict[str, dict[str, dict[str, Any]]] = {
    "antmaze": _build_antmaze_eval_positions(),
    "pointmaze": {},
}


def get_eval_position_config(
    env_family: str,
    variant: str,
    seed: int | None = None,
    config: dict | None = None,
) -> dict[str, Any] | None:
    mode = resolve_eval_position_mode(env_family, config)
    if mode == RANDOM_START_GOAL_MODE:
        return None
    if mode == FIX_START_GOAL_MODE:
        return EVAL_POSITIONS.get(env_family, {}).get(variant)
    if mode == HARD_SAMPLE_MODE:
        top_percent, top_n, hard_sample_alpha = _hard_sample_cache_key(config)
        return _hard_sample_eval_position_for_settings(
            env_family,
            variant,
            _normalize_eval_seed(seed),
            top_percent,
            top_n,
            hard_sample_alpha,
        )
    raise ValueError(f"Unsupported eval start/goal mode: {mode!r}")


def get_map_difficulty_config(
    env_family: str,
    variant: str,
) -> dict[str, Any] | None:
    """Return map difficulty derived from the hardest 10% of ordered pairs."""
    pair_space = _eval_pair_space_for_variant(env_family, variant)
    if pair_space is None:
        return None
    records, _ = pair_space
    return _pair_space_metadata(records[0])


def get_eval_position_pool_payload(
    env_family: str,
    variant: str,
    seed: int | None = None,
    config: dict | None = None,
) -> dict[str, Any] | None:
    """Return the persisted hard-sample pool payload for one variant."""
    if resolve_eval_position_mode(env_family, config) != HARD_SAMPLE_MODE:
        return None
    position_config = get_eval_position_config(
        env_family,
        variant,
        seed=seed,
        config=config,
    )
    if position_config is None:
        return None
    options = _resolve_hard_sample_options(config)
    records = list(position_config.get("start_goal_list") or [])
    return {
        "env_family": str(env_family),
        "variant": str(variant),
        "seed": _normalize_eval_seed(seed),
        "eval_position_mode": HARD_SAMPLE_MODE,
        "selection_policy": "seeded_weighted_hard_sample_permutation_cycle",
        "hard_sample_options": options,
        **{
            key: value
            for key, value in position_config.items()
            if key != "start_goal_list"
        },
        "selected_pair_count": len(records),
        "start_goal_list": records,
    }


def eval_position_count(
    env_family: str,
    variant: str,
    config: dict | None = None,
    seed: int | None = None,
) -> int:
    position_config = get_eval_position_config(
        env_family,
        variant,
        seed=seed,
        config=config,
    )
    if position_config is None:
        return 0
    if "fix_start_goal" in position_config:
        return 1
    if "start_goal_list" in position_config:
        return len(position_config["start_goal_list"])
    return 0


def eval_position_selection_policy(
    env_family: str,
    variant: str,
    config: dict | None = None,
    seed: int | None = None,
) -> str:
    mode = resolve_eval_position_mode(env_family, config)
    if mode == RANDOM_START_GOAL_MODE:
        return "env_default_random"
    position_config = get_eval_position_config(
        env_family,
        variant,
        seed=seed,
        config=config,
    )
    if position_config is None:
        return "env_default_random"
    if "fix_start_goal" in position_config:
        return "fixed"
    if "start_goal_list" in position_config:
        return (
            "seeded_weighted_hard_sample_permutation_cycle"
            if mode == HARD_SAMPLE_MODE
            else "seeded_permutation_cycle"
        )
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
    config: dict | None = None,
) -> dict[str, Any] | None:
    """Select the eval start/goal record for one episode.

    For `start_goal_list` variants, `seed` is the run-level eval seed rather
    than the per-episode reset seed.
    """
    mode = resolve_eval_position_mode(env_family, config)
    if mode == HARD_SAMPLE_MODE and episode_index is None:
        _resolve_hard_sample_options(config)
        return None
    position_config = get_eval_position_config(
        env_family,
        variant,
        seed=seed,
        config=config,
    )
    if position_config is None:
        return None
    if "fix_start_goal" in position_config:
        record = dict(position_config["fix_start_goal"])
        record["source"] = "fix_start_goal"
        record["index"] = 0
        return record
    if "start_goal_list" in position_config:
        records = position_config["start_goal_list"]
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
        record["source"] = "hard_sample" if mode == HARD_SAMPLE_MODE else "start_goal_list"
        record["index"] = int(index)
        return record
    return None


def eval_reset_options(
    env_family: str,
    variant: str,
    episode_index: int | None = None,
    seed: int | None = None,
    config: dict | None = None,
) -> dict[str, list[int]] | None:
    """Return `reset(options=...)` cells for the selected eval position."""
    record = select_eval_position(
        env_family,
        variant,
        episode_index,
        seed,
        config=config,
    )
    if record is None:
        return None
    return {
        "reset_cell": [int(record["start_cell"][0]), int(record["start_cell"][1])],
        "goal_cell": [int(record["goal_cell"][0]), int(record["goal_cell"][1])],
    }
