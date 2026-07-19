#!/usr/bin/env python3
"""Audit and summarize the paper_obs_v1 baseline runs.

The command exits non-zero until all six runs are complete and every expected
episode-level rollout record satisfies the formal experiment contract.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SEED = 20260716
EXPECTED_UPDATES = 500_000
EXPECTED_EPISODES_PER_VARIANT = 100

POINT_TRAIN_VARIANTS = (
    "open",
    "umaze",
    "medium",
    "large",
    *(f"local-layoutV2-{index:02d}" for index in range(1, 13)),
)
POINT_TEST_VARIANTS = tuple(
    f"test-layoutV2-{index:02d}" for index in range(1, 7)
)
ANT_TRAIN_VARIANTS = (
    "umaze",
    "medium-diverse",
    "large-diverse",
    "ultra",
    *(f"local-layout-{index:02d}" for index in range(1, 13)),
)
ANT_TEST_VARIANTS = tuple(f"test-layout-{index:02d}" for index in range(1, 4))


@dataclass(frozen=True)
class RunSpec:
    family: str
    algorithm: str
    experiment_id: str
    train_variant_count: int
    test_variant_count: int
    network: list[int]


RUNS = (
    RunSpec(
        "pointmaze",
        "mlp_bc",
        "paperobs1-pointmaze-mlp_bc-e300-500k-r100-s20260716",
        16,
        6,
        [1024, 1024, 1024],
    ),
    RunSpec(
        "pointmaze",
        "iql",
        "paperobs1-pointmaze-iql-e300-500k-r100-s20260716",
        16,
        6,
        [512, 512, 512],
    ),
    RunSpec(
        "pointmaze",
        "td3_bc",
        "paperobs1-pointmaze-td3_bc-e300-500k-r100-s20260716",
        16,
        6,
        [512, 512, 512],
    ),
    RunSpec(
        "antmaze",
        "mlp_bc",
        "paperobs1-antmaze-mlp_bc-e300-500k-r100-s20260716",
        16,
        3,
        [1024, 1024, 1024],
    ),
    RunSpec(
        "antmaze",
        "iql",
        "paperobs1-antmaze-iql-e300-500k-r100-s20260716",
        16,
        3,
        [512, 512, 512],
    ),
    RunSpec(
        "antmaze",
        "td3_bc",
        "paperobs1-antmaze-td3_bc-e300-500k-r100-s20260716",
        16,
        3,
        [1024, 1024, 1024],
    ),
)


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def _episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [bool(episode["success"]) for episode in episodes]
    success_steps = [
        float(episode["first_success_step"])
        for episode in episodes
        if episode["first_success_step"] is not None
    ]
    returns = [float(episode["return"]) for episode in episodes]
    lengths = [float(episode["length"]) for episode in episodes]
    return {
        "num_episodes": len(episodes),
        "successful_episode_count": sum(successes),
        "success_rate": statistics.fmean(successes),
        "first_success_step_mean": _mean(success_steps),
        "first_success_step_std": _pstdev(success_steps),
        "return_mean": statistics.fmean(returns),
        "return_std": statistics.pstdev(returns),
        "length_mean": statistics.fmean(lengths),
    }


def _check_metric_block(
    recorded: dict[str, Any], computed: dict[str, Any], label: str
) -> None:
    exact_keys = ("num_episodes", "successful_episode_count")
    for key in exact_keys:
        if int(recorded[key]) != int(computed[key]):
            raise AssertionError(
                f"{label}.{key}: expected {computed[key]}, found {recorded[key]}"
            )
    for key in (
        "success_rate",
        "return_mean",
        "return_std",
        "length_mean",
    ):
        _assert_close(recorded[key], computed[key], f"{label}.{key}")
    for key in ("first_success_step_mean", "first_success_step_std"):
        if recorded[key] is None or computed[key] is None:
            if recorded[key] is not computed[key]:
                raise AssertionError(
                    f"{label}.{key}: expected {computed[key]}, found {recorded[key]}"
                )
        else:
            _assert_close(recorded[key], computed[key], f"{label}.{key}")


def _audit_config(config: dict[str, Any], spec: RunSpec) -> None:
    checks = {
        "algorithm": spec.algorithm,
        "env_family": spec.family,
        "episode_keep_num": 300,
        "balance_variant_episode_count": True,
        "sampling_seed": SEED,
        "train_data_ratio": 0.9,
        "seed": 0,
        "n_steps": EXPECTED_UPDATES,
        "n_steps_per_epoch": 10_000,
    }
    for key, expected in checks.items():
        if config[key] != expected:
            raise AssertionError(
                f"{spec.experiment_id} config {key}: expected {expected}, "
                f"found {config[key]}"
            )
    if config["network"]["hidden_units"] != spec.network:
        raise AssertionError(f"{spec.experiment_id}: unexpected network")
    expected_network_tail = {
        "activation": "relu",
        "use_batch_norm": False,
        "use_layer_norm": False,
        "dropout_rate": None,
    }
    for key, expected in expected_network_tail.items():
        if config["network"][key] != expected:
            raise AssertionError(f"{spec.experiment_id}: unexpected network {key}")
    if spec.algorithm == "mlp_bc":
        expected_algorithm_config = {
            "batch_size": 512,
            "learning_rate": 0.0003,
            "policy_type": "deterministic",
            "compile_graph": False,
        }
    elif spec.algorithm == "iql":
        expected_algorithm_config = {
            "batch_size": 512,
            "gamma": 0.99,
            "actor_learning_rate": 0.0001,
            "critic_learning_rate": 0.0001,
            "tau": 0.005,
            "n_critics": 2,
            "expectile": 0.9,
            "weight_temp": 3.0,
            "max_weight": 100.0,
            "reward_scaler": {"type": "constant_shift", "shift": -1.0},
            "compile_graph": False,
        }
    else:
        expected_algorithm_config = {
            "batch_size": 512,
            "gamma": 0.99,
            "actor_learning_rate": 0.0003,
            "critic_learning_rate": 0.0003,
            "tau": 0.005,
            "n_critics": 2,
            "target_smoothing_sigma": 0.2,
            "target_smoothing_clip": 0.5,
            "alpha": 2.5,
            "update_actor_interval": 2,
            "reward_scaler": None,
            "compile_graph": False,
        }
    if config["algorithm_config"] != expected_algorithm_config:
        raise AssertionError(f"{spec.experiment_id}: algorithm config mismatch")
    observation = config["observation"]
    for key in ("include_map", "include_location_sensing", "include_wall_sensing"):
        if observation[key] is not True:
            raise AssertionError(f"{spec.experiment_id}: {key} is not enabled")
    if observation["wall_sensing_version"] != "v3":
        raise AssertionError(f"{spec.experiment_id}: wall sensing is not v3")
    _assert_close(
        observation["map_sensing_boundary_risk_threshold"],
        0.1,
        f"{spec.experiment_id}.wall_threshold",
    )
    evaluation = config["evaluation"]
    if evaluation["num_episodes"] != EXPECTED_EPISODES_PER_VARIANT:
        raise AssertionError(f"{spec.experiment_id}: rollout count is not 100")
    if evaluation["seed"] != SEED:
        raise AssertionError(f"{spec.experiment_id}: unexpected rollout seed")
    if not evaluation["enabled"] or evaluation["every_epochs"] != 1000:
        raise AssertionError(f"{spec.experiment_id}: unexpected eval schedule")
    expected_mode = "random-start-goal" if spec.family == "pointmaze" else "fix-start-goal"
    if evaluation["env_config"]["eval_start_goal_mode"] != expected_mode:
        raise AssertionError(f"{spec.experiment_id}: unexpected eval position mode")
    expected_train = (
        POINT_TRAIN_VARIANTS if spec.family == "pointmaze" else ANT_TRAIN_VARIANTS
    )
    expected_test = (
        POINT_TEST_VARIANTS if spec.family == "pointmaze" else ANT_TEST_VARIANTS
    )
    if tuple(config["resolved_train_variants"]) != expected_train:
        raise AssertionError(f"{spec.experiment_id}: resolved train variants mismatch")
    if tuple(config["resolved_eval_variants"]) != expected_train + expected_test:
        raise AssertionError(f"{spec.experiment_id}: resolved eval variants mismatch")
    if config["reward_type"] != "sparse" or config["allow_mixed_reward_types"]:
        raise AssertionError(f"{spec.experiment_id}: unexpected reward protocol")


def _audit_run(root: Path, spec: RunSpec) -> dict[str, Any]:
    run_dir = root / spec.experiment_id
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.yaml"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Incomplete run, missing {summary_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Incomplete run, missing {config_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _audit_config(config, spec)

    if summary["algorithm"] != spec.algorithm or summary["env_family"] != spec.family:
        raise AssertionError(f"{spec.experiment_id}: summary identity mismatch")
    if summary["n_steps"] != EXPECTED_UPDATES or summary["epochs"] != 50:
        raise AssertionError(f"{spec.experiment_id}: training did not reach 500k")
    train_variants = list(summary["train_variants"])
    eval_variants = list(summary["eval_variants"])
    test_variants = [variant for variant in eval_variants if variant not in train_variants]
    expected_train = list(
        POINT_TRAIN_VARIANTS if spec.family == "pointmaze" else ANT_TRAIN_VARIANTS
    )
    expected_test = list(
        POINT_TEST_VARIANTS if spec.family == "pointmaze" else ANT_TEST_VARIANTS
    )
    if train_variants != expected_train:
        raise AssertionError(f"{spec.experiment_id}: wrong train variants")
    if test_variants != expected_test or eval_variants != expected_train + expected_test:
        raise AssertionError(f"{spec.experiment_id}: wrong eval variants")

    expected_transitions = (
        (2_928_498, 328_033)
        if spec.family == "pointmaze"
        else (4_509_000, 501_000)
    )
    dataset = summary["dataset"]
    if dataset["train_episode_count"] != 4_320:
        raise AssertionError(f"{spec.experiment_id}: wrong train episode count")
    if dataset["validation_episode_count"] != 480:
        raise AssertionError(f"{spec.experiment_id}: wrong validation episode count")
    if (
        dataset["train_transition_count"],
        dataset["validation_transition_count"],
    ) != expected_transitions:
        raise AssertionError(f"{spec.experiment_id}: wrong transition counts")

    manifest = json.loads(
        (run_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["balanced_episode_target"] != 300 or manifest["warnings"]:
        raise AssertionError(f"{spec.experiment_id}: unexpected dataset balance/warnings")
    expected_dimension = 239 if spec.family == "pointmaze" else 231
    if manifest["observation_schema"]["dimension"] != expected_dimension:
        raise AssertionError(f"{spec.experiment_id}: observation dimension mismatch")
    if set(manifest["variants"]) != set(expected_train):
        raise AssertionError(f"{spec.experiment_id}: manifest variants mismatch")
    for variant, variant_manifest in manifest["variants"].items():
        counts = (
            variant_manifest["initial_sampled_episode_target"],
            variant_manifest["sampled_episode_count"],
            variant_manifest["train_episode_count"],
            variant_manifest["validation_episode_count"],
        )
        if counts != (300, 300, 270, 30):
            raise AssertionError(f"{spec.experiment_id}/{variant}: data split mismatch")

    training_history = summary["training_history"]
    if len(training_history) != 50 or [row["epoch"] for row in training_history] != list(
        range(1, 51)
    ):
        raise AssertionError(f"{spec.experiment_id}: incomplete training history")
    for row in training_history:
        for key, value in row["metrics"].items():
            if not math.isfinite(float(value)):
                raise AssertionError(
                    f"{spec.experiment_id}: non-finite training metric {key}"
                )

    final = summary["final_evaluation"]
    if final is None or final["epoch"] != 50 or final["step"] != EXPECTED_UPDATES:
        raise AssertionError(f"{spec.experiment_id}: missing final 500k evaluation")
    rollout = final["rollout"]
    if set(rollout["variants"]) != set(eval_variants):
        raise AssertionError(f"{spec.experiment_id}: rollout variants mismatch")

    all_episodes: list[dict[str, Any]] = []
    train_episodes: list[dict[str, Any]] = []
    test_episodes: list[dict[str, Any]] = []
    variant_rows = []
    expected_mode = "random-start-goal" if spec.family == "pointmaze" else "fix-start-goal"
    expected_policy = "env_default_random" if spec.family == "pointmaze" else "fixed"
    for variant in eval_variants:
        metrics = rollout["variants"][variant]
        episodes = metrics["episodes"]
        if len(episodes) != EXPECTED_EPISODES_PER_VARIANT:
            raise AssertionError(f"{spec.experiment_id}/{variant}: not 100 episodes")
        if [episode["episode_index"] for episode in episodes] != list(range(100)):
            raise AssertionError(f"{spec.experiment_id}/{variant}: episode index mismatch")
        if [episode["seed"] for episode in episodes] != list(range(SEED, SEED + 100)):
            raise AssertionError(f"{spec.experiment_id}/{variant}: episode seed mismatch")

        cell_pairs = set()
        for episode in episodes:
            start_goal = episode["start_goal"]
            if start_goal["sampling_mode"] != expected_mode:
                raise AssertionError(f"{spec.experiment_id}/{variant}: sampling mode mismatch")
            if start_goal["selection_policy"] != expected_policy:
                raise AssertionError(f"{spec.experiment_id}/{variant}: selection policy mismatch")
            for key in ("start_cell", "goal_cell", "start_xy", "goal_xy"):
                if len(start_goal[key]) != 2:
                    raise AssertionError(f"{spec.experiment_id}/{variant}: invalid {key}")
            cell_pairs.add(
                (tuple(start_goal["start_cell"]), tuple(start_goal["goal_cell"]))
            )
            first_success_step = episode["first_success_step"]
            if bool(episode["success"]) != (first_success_step is not None):
                raise AssertionError(
                    f"{spec.experiment_id}/{variant}: success-step mismatch"
                )
            if first_success_step is not None and not (
                1 <= int(first_success_step) <= int(episode["length"])
            ):
                raise AssertionError(
                    f"{spec.experiment_id}/{variant}: invalid success step"
                )
        if metrics["unique_start_goal_count"] != len(cell_pairs):
            raise AssertionError(f"{spec.experiment_id}/{variant}: unique pair mismatch")
        if spec.family == "pointmaze" and len(cell_pairs) <= 1:
            raise AssertionError(f"{spec.experiment_id}/{variant}: PointMaze did not vary pairs")
        if spec.family == "antmaze" and len(cell_pairs) != 1:
            raise AssertionError(f"{spec.experiment_id}/{variant}: AntMaze cells are not fixed")

        computed = _episode_metrics(episodes)
        _check_metric_block(metrics, computed, f"{spec.experiment_id}/{variant}")
        split = "train" if variant in train_variants else "test"
        variant_rows.append(
            {
                "variant": variant,
                "split": split,
                "unique_start_goal_count": len(cell_pairs),
                **computed,
            }
        )
        all_episodes.extend(episodes)
        (train_episodes if split == "train" else test_episodes).extend(episodes)

    overall = _episode_metrics(all_episodes)
    _check_metric_block(rollout["aggregate"], overall, f"{spec.experiment_id}/aggregate")
    expected_total = (spec.train_variant_count + spec.test_variant_count) * 100
    if overall["num_episodes"] != expected_total:
        raise AssertionError(f"{spec.experiment_id}: wrong total episode count")

    if len(summary["evaluation_history"]) != 1 or summary["evaluation_history"][0] != final:
        raise AssertionError(f"{spec.experiment_id}: unexpected evaluation history")
    evaluation_lines = [
        json.loads(line)
        for line in (run_dir / "evaluation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if evaluation_lines != [final]:
        raise AssertionError(f"{spec.experiment_id}: evaluation.jsonl mismatch")
    for required_path in (
        run_dir / "model.d3",
        run_dir / "checkpoints" / "step_500000.d3",
    ):
        if not required_path.is_file() or required_path.stat().st_size == 0:
            raise AssertionError(f"{spec.experiment_id}: missing {required_path.name}")

    training_metrics = summary["training_history"][-1]["metrics"]
    return {
        "experiment_id": spec.experiment_id,
        "family": spec.family,
        "algorithm": spec.algorithm,
        "train_variants": train_variants,
        "test_variants": test_variants,
        "dataset": summary["dataset"],
        "validation": final["validation"],
        "training_final_metrics": training_metrics,
        "train": _episode_metrics(train_episodes),
        "test": _episode_metrics(test_episodes),
        "overall": overall,
        "variants": variant_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("baseline_runs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "protocol": {
            "seed": SEED,
            "training_seed": 0,
            "updates": EXPECTED_UPDATES,
            "episode_keep_num": 300,
            "rollout_episodes_per_variant": EXPECTED_EPISODES_PER_VARIANT,
        },
        "runs": [_audit_run(args.run_root, spec) for spec in RUNS],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Audited 6/6 runs; wrote {args.output}")


if __name__ == "__main__":
    main()
