"""Inspect AntMaze layout topology and design-centric static difficulty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.antmaze.variants import ANTMAZE_VARIANTS
from utils.maze_metrics import metrics_for_variant


DEFAULT_COLUMNS = [
    "variant",
    "size",
    "free_cells",
    "wall_density",
    "shortest_path_len",
    "normalized_path_len",
    "diameter",
    "solution_turns",
    "solution_junctions",
    "degree_1_deadends",
    "junction_count",
    "cycle_rank",
    "articulation_points",
    "bridge_edges",
    "max_corridor_len",
    "static_difficulty",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variant names to inspect. Defaults to every registered AntMaze variant.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for JSON or table text.",
    )
    return parser.parse_args()


def _variant_names(selected: list[str] | None) -> list[str]:
    if selected is None:
        return list(ANTMAZE_VARIANTS)
    missing = [name for name in selected if name not in ANTMAZE_VARIANTS]
    if missing:
        raise ValueError(
            f"Unknown AntMaze variants: {missing}. "
            f"Available: {list(ANTMAZE_VARIANTS)}"
        )
    return list(selected)


def _row_for_variant(name: str) -> dict:
    metrics = metrics_for_variant(ANTMAZE_VARIANTS[name])
    payload = metrics.to_dict()
    payload["variant"] = name
    payload["size"] = f"{metrics.rows}x{metrics.cols}"
    payload["wall_density"] = round(metrics.wall_density, 4)
    payload["normalized_path_len"] = round(metrics.normalized_path_len, 4)
    payload["solution_turn_rate"] = round(metrics.solution_turn_rate, 4)
    payload["mean_corridor_len"] = round(metrics.mean_corridor_len, 2)
    return payload


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
    sep = "  ".join("-" * widths[column] for column in DEFAULT_COLUMNS)
    body = [
        "  ".join(str(row[column]).ljust(widths[column]) for column in DEFAULT_COLUMNS)
        for row in table_rows
    ]
    return "\n".join([header, sep, *body])


def main():
    args = parse_args()
    rows = [_row_for_variant(name) for name in _variant_names(args.variants)]
    text = (
        json.dumps(rows, ensure_ascii=False, indent=2)
        if args.json
        else _format_table(rows)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
