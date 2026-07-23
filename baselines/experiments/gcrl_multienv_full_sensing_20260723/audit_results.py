from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "baseline_runs"
OUTPUT = ROOT / "reports/gcrl_multienv_full_sensing_20260723_checkpoint_audit.json"
STEPS = (250_000, 500_000, 750_000, 1_000_000)
EPISODES_PER_VARIANT = 30
FAMILIES = {
    "pointmaze": {
        "train": [
            "open", "umaze", "medium", "large",
            *[f"local-layoutV2-{index:02d}" for index in range(1, 13)],
        ],
        "test": [f"test-layoutV2-{index:02d}" for index in range(1, 7)],
    },
    "antmaze": {
        "train": [
            "umaze", "medium-diverse", "large-diverse", "ultra",
            *[f"local-layout-{index:02d}" for index in range(1, 13)],
        ],
        "test": [f"test-layout-{index:02d}" for index in range(1, 5)],
    },
}
ALGORITHMS = ("crl", "hiql")
REQUIRED_OBSERVATION = {
    "include_map": True,
    "include_location_sensing": True,
    "include_wall_sensing": True,
    "wall_sensing_version": "v3",
    "map_sensing_boundary_risk_threshold": 0.10,
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_id(algorithm: str, family: str) -> str:
    return f"gcrl-full-sensing-{algorithm}-{family}-multienv-s0-1m-20260723"


def _group_metrics(rollout: dict, variants: list[str]) -> dict:
    episodes = [
        episode
        for variant in variants
        for episode in rollout["variants"][variant]["episodes"]
    ]
    successes = [bool(episode["success"]) for episode in episodes]
    returns = [float(episode["return"]) for episode in episodes]
    success_steps = [
        float(episode["first_success_step"])
        for episode in episodes
        if episode["first_success_step"] is not None
    ]
    return {
        "num_variants": len(variants),
        "num_episodes": len(episodes),
        "successful_episode_count": sum(successes),
        "success_rate": mean(successes),
        "return_mean": mean(returns),
        "return_std": pstdev(returns),
        "first_success_step_mean": mean(success_steps) if success_steps else None,
        "first_success_step_std": pstdev(success_steps) if success_steps else None,
    }


def _audit_rollout(rollout: dict, expected_variants: list[str]) -> None:
    if set(rollout["variants"]) != set(expected_variants):
        raise AssertionError("rollout variant set does not match configured train+test maps")
    for variant in expected_variants:
        record = rollout["variants"][variant]
        episodes = record["episodes"]
        if record["num_episodes"] != EPISODES_PER_VARIANT or len(episodes) != EPISODES_PER_VARIANT:
            raise AssertionError(f"{variant}: expected {EPISODES_PER_VARIANT} rollout episodes")
        if [episode["seed"] for episode in episodes] != list(range(EPISODES_PER_VARIANT)):
            raise AssertionError(f"{variant}: rollout reset seeds are not 0..29")
        success_count = sum(bool(episode["success"]) for episode in episodes)
        if record["successful_episode_count"] != success_count:
            raise AssertionError(f"{variant}: successful episode count mismatch")
    expected_episode_count = len(expected_variants) * EPISODES_PER_VARIANT
    if rollout["aggregate"]["num_episodes"] != expected_episode_count:
        raise AssertionError("aggregate rollout episode count mismatch")


def _variant_metrics(rollout: dict, variants: list[str]) -> dict[str, dict]:
    rows = {}
    for variant in variants:
        record = rollout["variants"][variant]
        rows[variant] = {
            "successful_episode_count": record["successful_episode_count"],
            "success_rate": record["success_rate"],
            "num_episodes": record["num_episodes"],
        }
    return rows


def main() -> None:
    output = {
        "protocol": {
            "checkpoint_steps": list(STEPS),
            "episodes_per_variant": EPISODES_PER_VARIANT,
            "observation": REQUIRED_OBSERVATION,
            "selection_metric": "held_out_test.success_rate",
        },
        "runs": {},
    }
    for family, groups in FAMILIES.items():
        expected_variants = groups["train"] + groups["test"]
        output["runs"][family] = {}
        for algorithm in ALGORITHMS:
            run_id = _run_id(algorithm, family)
            run_dir = RUNS_ROOT / run_id
            # Config is YAML; require exact values by inspecting the generated
            # manifest as a dependency-free JSON record.
            manifest = _load_json(run_dir / "dataset_manifest.json")
            if manifest["observation_config"] != REQUIRED_OBSERVATION:
                raise AssertionError(f"{run_id}: full-sensing observation config mismatch")
            summary = _load_json(run_dir / "summary.json")
            evaluations = {int(record["step"]): record for record in summary["evaluation_history"]}
            rows = []
            for step in STEPS:
                checkpoint = run_dir / "checkpoints" / f"step_{step}.msgpack"
                if not checkpoint.is_file():
                    raise AssertionError(f"{run_id}: missing checkpoint {checkpoint.name}")
                record = evaluations.get(step)
                if record is None:
                    raise AssertionError(f"{run_id}: missing evaluation at checkpoint step {step}")
                rollout = record["rollout"]
                _audit_rollout(rollout, expected_variants)
                rows.append({
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "train": _group_metrics(rollout, groups["train"]),
                    "held_out_test": _group_metrics(rollout, groups["test"]),
                    "per_variant": _variant_metrics(rollout, expected_variants),
                })
            output["runs"][family][algorithm] = {
                "experiment_id": run_id,
                "checkpoint_evaluations": rows,
            }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for family in FAMILIES:
        for algorithm in ALGORITHMS:
            rows = output["runs"][family][algorithm]["checkpoint_evaluations"]
            rendered = ", ".join(
                f"{row['step'] // 1000}k train={row['train']['success_rate']:.3f} "
                f"test={row['held_out_test']['success_rate']:.3f}"
                for row in rows
            )
            print(f"{family}/{algorithm}: {rendered}")
    print(f"Wrote checkpoint audit: {OUTPUT}")


if __name__ == "__main__":
    main()
