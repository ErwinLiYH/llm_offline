from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import d3rlpy
import torch
import yaml

from baselines.artifacts import write_json
from baselines.evaluation import evaluate_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll out saved intermediate d3rlpy baseline checkpoints."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Intermediate update counts to evaluate. By default, evaluate every "
            "saved checkpoint before the final update."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _expected_episode_count(config: dict) -> int:
    return len(config["resolved_eval_variants"]) * config["evaluation"][
        "num_episodes"
    ]


def _complete_result(path: Path, *, config: dict, step: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rollout = payload["rollout"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if payload.get("experiment_id") != config["experiment_id"]:
        return False
    if payload.get("step") != step:
        return False
    if set(rollout.get("variants", {})) != set(config["resolved_eval_variants"]):
        return False
    expected_per_variant = config["evaluation"]["num_episodes"]
    for result in rollout["variants"].values():
        if result.get("num_episodes") != expected_per_variant:
            return False
        if len(result.get("episodes", [])) != expected_per_variant:
            return False
    return rollout.get("aggregate", {}).get("num_episodes") == _expected_episode_count(
        config
    )


def main() -> None:
    args = parse_args()
    run_dir = ROOT / "baseline_runs" / args.experiment_id
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing formal run config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != args.experiment_id:
        raise ValueError("Run directory and resolved experiment_id do not match")
    final_step = config["n_steps"]
    checkpoint_interval = (
        config["save_interval_epochs"] * config["n_steps_per_epoch"]
    )
    default_steps = range(checkpoint_interval, final_step, checkpoint_interval)
    requested_steps = sorted(set(args.steps if args.steps is not None else default_steps))
    invalid = [
        step
        for step in requested_steps
        if step <= 0 or step >= final_step or step % checkpoint_interval != 0
    ]
    if invalid:
        raise ValueError(
            f"Unsupported intermediate steps for final_step={final_step}, "
            f"checkpoint_interval={checkpoint_interval}: {invalid}"
        )

    output_dir = run_dir / "checkpoint_rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)
    for step in requested_steps:
        checkpoint = run_dir / "checkpoints" / f"step_{step}.d3"
        output = output_dir / f"step_{step}.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        if _complete_result(output, config=config, step=step):
            print(f"[checkpoint rollout] skip complete {args.experiment_id} step={step}")
            continue

        print(f"[checkpoint rollout] start {args.experiment_id} step={step}")
        algo = d3rlpy.load_learnable(str(checkpoint), device=args.device)
        rollout = evaluate_rollouts(
            algo,
            env_family=config["env_family"],
            variants=config["resolved_eval_variants"],
            reward_types=config["resolved_eval_reward_types"],
            evaluation_config=config["evaluation"],
            observation_config=config["observation"],
        )
        payload = {
            "experiment_id": args.experiment_id,
            "algorithm": config["algorithm"],
            "env_family": config["env_family"],
            "step": step,
            "checkpoint_path": str(checkpoint),
            "evaluation_seed": config["evaluation"]["seed"],
            "num_episodes_per_variant": config["evaluation"]["num_episodes"],
            "rollout": rollout,
        }
        write_json(output, payload)
        aggregate = rollout["aggregate"]
        print(
            f"[checkpoint rollout] finish {args.experiment_id} step={step} "
            f"success={aggregate['success_rate']:.4f} "
            f"episodes={aggregate['num_episodes']} output={output}"
        )
        del algo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
