from __future__ import annotations

import argparse

from baselines.config import normalize_baseline_config
from utils.config_loader import load_merged_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train d3rlpy-based CrossMaze offline RL baselines."
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more layered YAML configs; later files override earlier files.",
    )
    parser.add_argument(
        "--experiment_id",
        default=None,
        help="Optional run directory name override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_merged_config(args.config)
    if args.experiment_id is not None:
        raw_config["experiment_id"] = args.experiment_id
    config = normalize_baseline_config(raw_config)

    # Importing the runner performs the d3rlpy import and is intentionally
    # delayed so --help and config errors work outside the baseline env.
    from baselines.runner import train_baseline

    train_baseline(config)


if __name__ == "__main__":
    main()
