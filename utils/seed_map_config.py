"""Shared seed-map CLI/config normalization helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from crossmaze.reward import normalize_reward_type
from crossmaze.seed_map import (
    DEFAULT_SEED_MAP_MAX_SIZE,
    DEFAULT_SEED_MAP_MIN_SIZE,
    SEED_MAP_VERSION,
    SeedMapSpec,
    normalize_seed_map_spec,
)
from data.seed_map_corpus import seed_map_dataset_name


def add_seed_map_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-seed-map",
        action="store_true",
        help=(
            "Generate one aggregate procedural seed-map corpus instead of "
            "registered per-variant datasets."
        ),
    )
    parser.add_argument("--seed-map-start", type=int, default=None)
    parser.add_argument("--seed-map-end", type=int, default=None)
    parser.add_argument("--seed-map-trajectories-per-seed", type=int, default=None)
    parser.add_argument(
        "--seed-map-dataset-root",
        type=Path,
        default=None,
        help=(
            "Parent output directory or the exact derived <start>-<end>-<count> "
            "dataset directory."
        ),
    )
    parser.add_argument("--seed-map-version", default=SEED_MAP_VERSION)
    parser.add_argument(
        "--seed-map-size-mode",
        choices=("random", "fixed"),
        default="random",
    )
    parser.add_argument(
        "--seed-map-min-size",
        type=int,
        default=DEFAULT_SEED_MAP_MIN_SIZE,
    )
    parser.add_argument(
        "--seed-map-max-size",
        type=int,
        default=DEFAULT_SEED_MAP_MAX_SIZE,
    )
    parser.add_argument("--seed-map-fixed-rows", type=int, default=None)
    parser.add_argument("--seed-map-fixed-cols", type=int, default=None)


def seed_map_spec_from_args(args) -> SeedMapSpec:
    raw: dict[str, Any] = {
        "version": args.seed_map_version,
        "size_mode": args.seed_map_size_mode,
    }
    if args.seed_map_size_mode == "random":
        raw.update(
            {
                "min_size": args.seed_map_min_size,
                "max_size": args.seed_map_max_size,
            }
        )
    else:
        raw.update(
            {
                "fixed_rows": args.seed_map_fixed_rows,
                "fixed_cols": args.seed_map_fixed_cols,
            }
        )
    return normalize_seed_map_spec(raw)


def validate_seed_map_generation_args(args) -> SeedMapSpec:
    if args.seed_map_start is None:
        raise ValueError("--use-seed-map requires --seed-map-start")
    if args.seed_map_end is None:
        raise ValueError("--use-seed-map requires --seed-map-end")
    if args.seed_map_trajectories_per_seed is None:
        raise ValueError(
            "--use-seed-map requires --seed-map-trajectories-per-seed"
        )
    if args.seed_map_start < 0:
        raise ValueError("--seed-map-start must be >= 0")
    if args.seed_map_end <= args.seed_map_start:
        raise ValueError(
            "--seed-map-end must be greater than --seed-map-start "
            "for the half-open range"
        )
    if args.seed_map_trajectories_per_seed < 1:
        raise ValueError("--seed-map-trajectories-per-seed must be >= 1")
    return seed_map_spec_from_args(args)


def resolve_seed_map_generation_path(
    args,
    *,
    env_family: str,
    reward_type: str,
    repo_root: Path,
    spec: SeedMapSpec,
) -> Path:
    name = seed_map_dataset_name(
        args.seed_map_start,
        args.seed_map_end,
        args.seed_map_trajectories_per_seed,
    )
    if args.seed_map_dataset_root is not None:
        configured = args.seed_map_dataset_root.expanduser()
        path = configured if configured.name == name else configured / name
    else:
        reward_type = normalize_reward_type(reward_type)
        if spec.size_mode == "random":
            size_tag = f"random-size{spec.min_size}-{spec.max_size}"
        else:
            size_tag = f"fixed-{spec.fixed_rows}x{spec.fixed_cols}"
        path = (
            repo_root
            / "local_datasets"
            / f"{env_family}-seed-map-{spec.version}-{size_tag}-{reward_type}"
            / name
        )
    return path.resolve()


def seed_map_section_enabled(config: dict, section_name: str) -> bool:
    section = config.get(section_name)
    if section is None:
        return False
    if not isinstance(section, dict):
        raise ValueError(
            f"{section_name} must be a mapping or null, "
            f"got {type(section).__name__}"
        )
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"{section_name}.enabled must be true or false, "
            f"got {enabled!r}"
        )
    return enabled
