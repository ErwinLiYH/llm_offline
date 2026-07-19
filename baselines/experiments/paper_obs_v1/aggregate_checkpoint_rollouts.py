from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "baseline_runs"
OUTPUT = ROOT / "reports/baseline_paper_obs_v1_checkpoint_curves.json"
SEED = 20260716
STEPS = (100000, 200000, 300000, 400000, 500000)
ALGORITHMS = ("mlp_bc", "iql", "td3_bc")
TRAIN_VARIANTS = {
    "pointmaze": [
        "open",
        "umaze",
        "medium",
        "large",
        *[f"local-layoutV2-{index:02d}" for index in range(1, 13)],
    ],
    "antmaze": [
        "umaze",
        "medium-diverse",
        "large-diverse",
        "ultra",
        *[f"local-layout-{index:02d}" for index in range(1, 13)],
    ],
}
HELD_OUT_VARIANTS = {
    "pointmaze": [f"test-layoutV2-{index:02d}" for index in range(1, 7)],
    "antmaze": [f"test-layout-{index:02d}" for index in range(1, 4)],
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_id(family: str, algorithm: str) -> str:
    return f"paperobs1-{family}-{algorithm}-e300-500k-r100-s{SEED}"


def _assert_close(actual: float, expected: float, *, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f"{context}: {actual} != {expected}")


def _audit_rollout(
    rollout: dict,
    *,
    family: str,
    variants: list[str],
    pair_reference: dict[str, list[tuple[tuple[int, int], tuple[int, int]]]],
) -> None:
    if set(rollout["variants"]) != set(variants):
        raise AssertionError(f"{family}: variant set mismatch")
    all_successes = []
    all_returns = []
    all_lengths = []
    all_success_steps = []
    for variant in variants:
        result = rollout["variants"][variant]
        episodes = result["episodes"]
        if result["num_episodes"] != 100 or len(episodes) != 100:
            raise AssertionError(f"{family}/{variant}: expected 100 episodes")
        successes = []
        returns = []
        lengths = []
        success_steps = []
        pairs = []
        for index, episode in enumerate(episodes):
            if episode["episode_index"] != index or episode["seed"] != SEED + index:
                raise AssertionError(f"{family}/{variant}: episode index/seed mismatch")
            success = bool(episode["success"])
            if success != (episode["first_success_step"] is not None):
                raise AssertionError(f"{family}/{variant}: success-step mismatch")
            if episode["first_success_step"] is not None:
                success_steps.append(float(episode["first_success_step"]))
            successes.append(float(success))
            returns.append(float(episode["return"]))
            lengths.append(float(episode["length"]))
            start_goal = episode["start_goal"]
            expected_mode = "random-start-goal" if family == "pointmaze" else "fix-start-goal"
            if start_goal["sampling_mode"] != expected_mode:
                raise AssertionError(f"{family}/{variant}: sampling mode mismatch")
            pairs.append(
                (
                    tuple(start_goal["start_cell"]),
                    tuple(start_goal["goal_cell"]),
                )
            )
        if family == "pointmaze" and len(set(pairs)) <= 1:
            raise AssertionError(f"{family}/{variant}: random pair diversity missing")
        if family == "antmaze" and len(set(pairs)) != 1:
            raise AssertionError(f"{family}/{variant}: fixed pair changed")
        if variant in pair_reference and pair_reference[variant] != pairs:
            raise AssertionError(f"{family}/{variant}: pair sequence changed across runs/steps")
        pair_reference.setdefault(variant, pairs)

        success_count = int(sum(successes))
        if result["successful_episode_count"] != success_count:
            raise AssertionError(f"{family}/{variant}: success count mismatch")
        _assert_close(result["success_rate"], mean(successes), context=f"{family}/{variant}/success")
        _assert_close(result["return_mean"], mean(returns), context=f"{family}/{variant}/return")
        _assert_close(result["return_std"], pstdev(returns), context=f"{family}/{variant}/return_std")
        _assert_close(result["length_mean"], mean(lengths), context=f"{family}/{variant}/length")
        if success_steps:
            _assert_close(
                result["first_success_step_mean"],
                mean(success_steps),
                context=f"{family}/{variant}/first_success",
            )
            _assert_close(
                result["first_success_step_std"],
                pstdev(success_steps),
                context=f"{family}/{variant}/first_success_std",
            )
        elif result["first_success_step_mean"] is not None or result["first_success_step_std"] is not None:
            raise AssertionError(f"{family}/{variant}: empty success-step metrics are not null")
        all_successes.extend(successes)
        all_returns.extend(returns)
        all_lengths.extend(lengths)
        all_success_steps.extend(success_steps)

    aggregate = rollout["aggregate"]
    if aggregate["num_episodes"] != len(all_successes):
        raise AssertionError(f"{family}: aggregate episode count mismatch")
    if aggregate["successful_episode_count"] != int(sum(all_successes)):
        raise AssertionError(f"{family}: aggregate success count mismatch")
    _assert_close(aggregate["success_rate"], mean(all_successes), context=f"{family}/aggregate/success")
    _assert_close(aggregate["return_mean"], mean(all_returns), context=f"{family}/aggregate/return")
    _assert_close(aggregate["return_std"], pstdev(all_returns), context=f"{family}/aggregate/return_std")
    _assert_close(aggregate["length_mean"], mean(all_lengths), context=f"{family}/aggregate/length")
    if all_success_steps:
        _assert_close(
            aggregate["first_success_step_mean"],
            mean(all_success_steps),
            context=f"{family}/aggregate/first_success",
        )
        _assert_close(
            aggregate["first_success_step_std"],
            pstdev(all_success_steps),
            context=f"{family}/aggregate/first_success_std",
        )


def _group_metrics(rollout: dict, variants: list[str]) -> dict:
    episodes = [
        episode
        for variant in variants
        for episode in rollout["variants"][variant]["episodes"]
    ]
    success_steps = [
        float(episode["first_success_step"])
        for episode in episodes
        if episode["first_success_step"] is not None
    ]
    returns = [float(episode["return"]) for episode in episodes]
    successes = [bool(episode["success"]) for episode in episodes]
    return {
        "num_variants": len(variants),
        "num_episodes": len(episodes),
        "successful_episode_count": sum(successes),
        "success_rate": mean(successes),
        "first_success_step_mean": mean(success_steps) if success_steps else None,
        "first_success_step_std": pstdev(success_steps) if success_steps else None,
        "return_mean": mean(returns),
        "return_std": pstdev(returns),
    }


def _compact_variants(rollout: dict) -> dict:
    return {
        variant: {
            key: result[key]
            for key in (
                "num_episodes",
                "successful_episode_count",
                "success_rate",
                "first_success_step_mean",
                "first_success_step_std",
                "return_mean",
                "return_std",
                "length_mean",
                "unique_start_goal_count",
            )
        }
        for variant, result in rollout["variants"].items()
    }


def main() -> None:
    payload = {
        "protocol": {
            "steps": list(STEPS),
            "num_episodes_per_variant": 100,
            "evaluation_seed": SEED,
            "training_seed": 0,
            "metric_aggregation": "episode-weighted",
            "checkpoint_selection": "report-all-no-posthoc-replacement",
            "num_intermediate_checkpoint_episodes": 49200,
            "num_final_checkpoint_episodes": 12300,
            "audited_episode_total": 61500,
        },
        "runs": {},
    }
    pair_reference = {"pointmaze": {}, "antmaze": {}}
    audited_episodes = 0
    for family in ("pointmaze", "antmaze"):
        payload["runs"][family] = {}
        expected_variants = TRAIN_VARIANTS[family] + HELD_OUT_VARIANTS[family]
        for algorithm in ALGORITHMS:
            run_id = _run_id(family, algorithm)
            run_dir = RUNS_ROOT / run_id
            summary = _load_json(run_dir / "summary.json")
            training_by_step = {
                int(entry["epoch"]) * 10000: entry["metrics"]
                for entry in summary["training_history"]
            }
            step_results = []
            for step in STEPS:
                if step == 500000:
                    rollout = summary["final_evaluation"]["rollout"]
                    source = str(run_dir / "summary.json")
                else:
                    path = run_dir / "checkpoint_rollouts" / f"step_{step}.json"
                    raw = _load_json(path)
                    if raw["experiment_id"] != run_id or raw["step"] != step:
                        raise AssertionError(f"{run_id}/{step}: raw result identity mismatch")
                    rollout = raw["rollout"]
                    source = str(path)
                _audit_rollout(
                    rollout,
                    family=family,
                    variants=expected_variants,
                    pair_reference=pair_reference[family],
                )
                audited_episodes += rollout["aggregate"]["num_episodes"]
                train_metrics = _group_metrics(rollout, TRAIN_VARIANTS[family])
                held_out_metrics = _group_metrics(rollout, HELD_OUT_VARIANTS[family])
                overall_metrics = _group_metrics(rollout, expected_variants)
                _assert_close(
                    overall_metrics["success_rate"],
                    rollout["aggregate"]["success_rate"],
                    context=f"{run_id}/{step}/overall_success",
                )
                step_results.append(
                    {
                        "step": step,
                        "source": source,
                        "training_metrics": training_by_step[step],
                        "train": train_metrics,
                        "held_out": held_out_metrics,
                        "overall": overall_metrics,
                        "variants": _compact_variants(rollout),
                    }
                )
            payload["runs"][family][algorithm] = {
                "experiment_id": run_id,
                "steps": step_results,
            }

    expected_total = 3 * 5 * (22 * 100 + 19 * 100)
    if audited_episodes != expected_total:
        raise AssertionError(
            f"Audited episode total mismatch: {audited_episodes} != {expected_total}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Audited {audited_episodes} checkpoint-curve episodes.")
    for family in ("pointmaze", "antmaze"):
        for algorithm in ALGORITHMS:
            rows = payload["runs"][family][algorithm]["steps"]
            rendered = ", ".join(
                f"{row['step']//1000}k={100*row['overall']['success_rate']:.2f}%"
                for row in rows
            )
            print(f"{family}/{algorithm}: {rendered}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
