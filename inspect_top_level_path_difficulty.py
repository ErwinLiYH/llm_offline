"""Inspect the formal CrossMaze topology-difficulty metric."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from crossmaze import (
    compute_topology_difficulty,
    topology_difficulty_config,
)
from crossmaze.variants import ENV_FACTS


DEFAULT_COLUMNS = [
    "variant",
    "size",
    "area",
    "free",
    "pairs",
    "top_paths",
    "mean_len",
    "mean_turns",
    "mean_detour",
    "t_junctions",
    "crosses",
    "raw_score",
    "score",
    "seconds",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-family",
        choices=sorted(ENV_FACTS),
        default="pointmaze",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variant names to inspect. Defaults to every registered variant.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _variant_names(env_family: str, selected: list[str] | None) -> list[str]:
    facts = ENV_FACTS[env_family]
    if selected is None:
        return list(facts)
    missing = [name for name in selected if name not in facts]
    if missing:
        raise ValueError(
            f"Unknown {env_family} variants: {missing}. Available: {list(facts)}"
        )
    return list(selected)


def _row_for_variant(env_family: str, variant: str) -> dict:
    started = time.perf_counter()
    result = compute_topology_difficulty(
        ENV_FACTS[env_family][variant]["maze_map"]
    )
    metrics = result.metrics
    difficulty = result.difficulty
    elapsed = time.perf_counter() - started
    return {
        "variant": variant,
        "size": f"{metrics.rows}x{metrics.cols}",
        "area": metrics.map_area,
        "free": metrics.free_cells,
        "pairs": metrics.unordered_pair_count,
        "top_paths": metrics.top_level_path_count,
        "mean_len": round(metrics.mean_top_level_path_len, 4),
        "max_len": metrics.max_top_level_path_len,
        "mean_turns": round(metrics.mean_top_level_turns, 4),
        "max_turns": metrics.max_top_level_turns,
        "mean_detour": round(metrics.mean_top_level_detour_ratio, 4),
        "max_detour": round(metrics.max_top_level_detour_ratio, 4),
        "t_junctions": metrics.corridor_t_junctions,
        "crosses": metrics.corridor_cross_junctions,
        "raw_score": difficulty.raw_score,
        "score": difficulty.score,
        "seconds": round(elapsed, 6),
        "metrics": metrics.to_dict(),
        "difficulty": difficulty.to_dict(),
    }


def _format_table(rows: list[dict]) -> str:
    table_rows = [
        {column: row[column] for column in DEFAULT_COLUMNS}
        for row in rows
    ]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in table_rows))
        for column in DEFAULT_COLUMNS
    }
    header = "  ".join(column.ljust(widths[column]) for column in DEFAULT_COLUMNS)
    separator = "  ".join("-" * widths[column] for column in DEFAULT_COLUMNS)
    body = [
        "  ".join(str(row[column]).ljust(widths[column]) for column in DEFAULT_COLUMNS)
        for row in table_rows
    ]
    return "\n".join([header, separator, *body])


def main():
    args = parse_args()
    names = _variant_names(args.env_family, args.variants)
    rows = [_row_for_variant(args.env_family, name) for name in names]
    if args.json:
        text = json.dumps(
            {
                "env_family": args.env_family,
                "difficulty_config": topology_difficulty_config(),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    else:
        text = _format_table(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
