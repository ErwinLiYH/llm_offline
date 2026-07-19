from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.experiments.paper_obs_v1.aggregate_checkpoint_rollouts import (
    HELD_OUT_VARIANTS,
    RUNS_ROOT,
    SEED,
    TRAIN_VARIANTS,
    _audit_rollout,
    _compact_variants,
    _group_metrics,
    _load_json,
)


OUTPUT = ROOT / "reports/baseline_paper_obs_v1_antmaze_2m_curves.json"
ALGORITHMS = ("mlp_bc", "iql", "td3_bc")
STEPS = tuple(range(100_000, 2_000_001, 100_000))
PREFIX_STEPS = tuple(range(100_000, 1_000_001, 100_000))


def _run_id(algorithm: str, budget: str) -> str:
    return f"paperobs1-antmaze-{algorithm}-e300-{budget}-r100-s{SEED}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rollout(run_dir: Path, summary: dict, step: int) -> tuple[dict, str]:
    if step == summary["n_steps"]:
        return summary["final_evaluation"]["rollout"], str(run_dir / "summary.json")
    path = run_dir / "checkpoint_rollouts" / f"step_{step}.json"
    raw = _load_json(path)
    if raw["experiment_id"] != run_dir.name or raw["step"] != step:
        raise AssertionError(f"{run_dir.name}/{step}: rollout identity mismatch")
    return raw["rollout"], str(path)


def _assert_configs_match(previous: dict, new: dict, *, algorithm: str) -> None:
    previous = dict(previous)
    new = dict(new)
    previous_id = previous.pop("experiment_id")
    new_id = new.pop("experiment_id")
    previous_steps = previous.pop("n_steps")
    new_steps = new.pop("n_steps")
    if previous_id != _run_id(algorithm, "1m") or new_id != _run_id(algorithm, "2m"):
        raise AssertionError(f"{algorithm}: config run identity mismatch")
    if previous_steps != 1_000_000 or new_steps != 2_000_000:
        raise AssertionError(f"{algorithm}: config training budget mismatch")
    if previous != new:
        differing = sorted(
            key for key in previous.keys() | new.keys() if previous.get(key) != new.get(key)
        )
        raise AssertionError(f"{algorithm}: config differs beyond budget/id: {differing}")


def _audit_training(summary: dict, *, run_id: str) -> dict[int, dict]:
    if summary["experiment_id"] != run_id:
        raise AssertionError(f"{run_id}: summary identity mismatch")
    if summary["n_steps"] != 2_000_000 or summary["epochs"] != 200:
        raise AssertionError(f"{run_id}: incomplete 2M training")
    if len(summary["training_history"]) != 200:
        raise AssertionError(f"{run_id}: incomplete training history")
    by_step = {}
    for expected_epoch, entry in enumerate(summary["training_history"], start=1):
        if entry["epoch"] != expected_epoch:
            raise AssertionError(f"{run_id}: training epoch sequence mismatch")
        step = expected_epoch * 10_000
        for key, value in entry["metrics"].items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise AssertionError(f"{run_id}/{step}: non-finite training metric {key}")
        by_step[step] = entry["metrics"]
    final = summary["final_evaluation"]
    if final["epoch"] != 200 or final["step"] != 2_000_000:
        raise AssertionError(f"{run_id}: missing final 2M evaluation")
    return by_step


def main() -> None:
    expected_variants = TRAIN_VARIANTS["antmaze"] + HELD_OUT_VARIANTS["antmaze"]
    pair_reference: dict[str, list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    payload = {
        "protocol": {
            "steps": list(STEPS),
            "num_episodes_per_variant": 100,
            "evaluation_seed": SEED,
            "training_seed": 0,
            "metric_aggregation": "episode-weighted",
            "training_mode": "fresh-from-scratch-2m",
            "checkpoint_selection": "report-all-no-posthoc-replacement",
            "curve_episode_total": 3 * len(STEPS) * len(expected_variants) * 100,
        },
        "runs": {},
    }
    audited_episodes = 0

    for algorithm in ALGORITHMS:
        source_500k_id = _run_id(algorithm, "500k")
        source_1m_id = _run_id(algorithm, "1m")
        new_id = _run_id(algorithm, "2m")
        source_500k_dir = RUNS_ROOT / source_500k_id
        source_1m_dir = RUNS_ROOT / source_1m_id
        new_dir = RUNS_ROOT / new_id
        source_500k_summary = _load_json(source_500k_dir / "summary.json")
        source_1m_summary = _load_json(source_1m_dir / "summary.json")
        new_summary = _load_json(new_dir / "summary.json")
        source_1m_config = yaml.safe_load(
            (source_1m_dir / "config.yaml").read_text(encoding="utf-8")
        )
        new_config = yaml.safe_load((new_dir / "config.yaml").read_text(encoding="utf-8"))
        _assert_configs_match(source_1m_config, new_config, algorithm=algorithm)
        source_manifest = _load_json(source_1m_dir / "dataset_manifest.json")
        if source_manifest != _load_json(new_dir / "dataset_manifest.json"):
            raise AssertionError(f"{algorithm}: dataset manifest changed from 1M to 2M")
        if source_manifest != _load_json(source_500k_dir / "dataset_manifest.json"):
            raise AssertionError(f"{algorithm}: 500k/1M source manifests disagree")
        training_by_step = _audit_training(new_summary, run_id=new_id)

        checkpoint_prefix = {}
        prefix_identical = True
        for step in PREFIX_STEPS:
            source_checkpoint = source_1m_dir / "checkpoints" / f"step_{step}.d3"
            new_checkpoint = new_dir / "checkpoints" / f"step_{step}.d3"
            source_hash = _sha256(source_checkpoint)
            new_hash = _sha256(new_checkpoint)
            identical = source_hash == new_hash
            prefix_identical &= identical
            checkpoint_prefix[str(step)] = {
                "identical": identical,
                "source_1m_sha256": source_hash,
                "new_2m_sha256": new_hash,
            }

        step_results = []
        for step in STEPS:
            if step <= 1_000_000 and checkpoint_prefix[str(step)]["identical"]:
                if step <= 500_000:
                    rollout, source = _load_rollout(
                        source_500k_dir, source_500k_summary, step
                    )
                else:
                    rollout, source = _load_rollout(source_1m_dir, source_1m_summary, step)
            else:
                rollout, source = _load_rollout(new_dir, new_summary, step)
            _audit_rollout(
                rollout,
                family="antmaze",
                variants=expected_variants,
                pair_reference=pair_reference,
            )
            audited_episodes += rollout["aggregate"]["num_episodes"]
            step_results.append(
                {
                    "step": step,
                    "rollout_source": source,
                    "checkpoint_path": str(new_dir / "checkpoints" / f"step_{step}.d3"),
                    "training_metrics": training_by_step[step],
                    "train": _group_metrics(rollout, TRAIN_VARIANTS["antmaze"]),
                    "held_out": _group_metrics(rollout, HELD_OUT_VARIANTS["antmaze"]),
                    "overall": _group_metrics(rollout, expected_variants),
                    "variants": _compact_variants(rollout),
                }
            )

        payload["runs"][algorithm] = {
            "experiment_id": new_id,
            "source_1m_experiment_id": source_1m_id,
            "source_500k_experiment_id": source_500k_id,
            "checkpoint_prefix_identical": prefix_identical,
            "checkpoint_prefix": checkpoint_prefix,
            "steps": step_results,
        }

    if audited_episodes != payload["protocol"]["curve_episode_total"]:
        raise AssertionError(
            f"Audited episode total mismatch: {audited_episodes} != "
            f"{payload['protocol']['curve_episode_total']}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Audited {audited_episodes} AntMaze 2M-curve episodes.")
    for algorithm in ALGORITHMS:
        run = payload["runs"][algorithm]
        rendered = ", ".join(
            f"{row['step']//1000}k={100*row['overall']['success_rate']:.2f}%"
            for row in run["steps"]
        )
        print(
            f"antmaze/{algorithm} prefix_identical={run['checkpoint_prefix_identical']}: "
            f"{rendered}"
        )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
