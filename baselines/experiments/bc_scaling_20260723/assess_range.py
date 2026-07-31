"""Classify a BC scaling range as stable, unstable, or invalid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


MIN_RELATIVE_LOSS_DECREASE = 0.10
MIN_ACTION_STD = 0.05
MIN_POSITIVE_GRADIENT_FRACTION = 0.80
MAX_VALIDATION_MSE_RATIO = 1.05


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_partial_training_history(run_dir: Path) -> list[dict]:
    path = run_dir / "d3rlpy_logs" / "training" / "loss.csv"
    if not path.is_file():
        return []
    history = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(",")
        if len(fields) != 3:
            raise ValueError(f"malformed loss row in {path}: {line!r}")
        epoch, step, loss = fields
        history.append(
            {
                "epoch": int(epoch),
                "step": int(step),
                "metrics": {"loss": float(loss)},
            }
        )
    return history


def load_range_summary(run_dir: Path) -> dict:
    """Load a completed summary or reconstruct an early-stopped partial run."""

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "_partial_without_summary": True,
        "training_history": _load_partial_training_history(run_dir),
        "training_diagnostics": _read_jsonl(
            run_dir / "training_diagnostics.jsonl"
        ),
        "evaluation_history": _read_jsonl(run_dir / "evaluation.jsonl"),
    }


def assess_range(
    summary: dict,
    config: dict,
    *,
    run_dir: Path | None = None,
    expected_n_steps: int = 50_000,
) -> dict:
    reasons: list[str] = []
    history = summary.get("training_history", [])
    diagnostics = summary.get("training_diagnostics", [])
    observed_step_values = [
        row.get("step")
        for row in (*history, *diagnostics)
        if _finite(row.get("step"))
    ]
    observed_training_steps = (
        max(observed_step_values) if observed_step_values else None
    )
    partial_without_summary = bool(summary.get("_partial_without_summary"))
    infrastructure_failure = (
        partial_without_summary
        and (
            observed_training_steps is None
            or observed_training_steps <= 0
        )
    )
    if infrastructure_failure:
        reasons.append(
            "run ended before any observed training update; this is invalid "
            "infrastructure evidence and must not trigger a learning-rate change"
        )
    elif partial_without_summary:
        reasons.append(
            "run ended before summary.json; partial artifacts are unstable evidence"
        )
    n_steps = config["n_steps"]
    steps_per_epoch = config["n_steps_per_epoch"]
    if n_steps != expected_n_steps or steps_per_epoch != 5_000:
        reasons.append(
            "range must use "
            f"{expected_n_steps // 1_000}k updates and 5k-update epochs"
        )
    expected_epochs = n_steps // steps_per_epoch

    if len(history) != expected_epochs:
        reasons.append(
            f"expected {expected_epochs} training-history rows, got {len(history)}"
        )
    if len(diagnostics) != expected_epochs:
        reasons.append(
            f"expected {expected_epochs} diagnostic rows, got {len(diagnostics)}"
        )

    losses = [row.get("metrics", {}).get("loss") for row in history]
    if not losses or any(not _finite(value) for value in losses):
        reasons.append("training loss is missing or non-finite")
        relative_loss_decrease = None
    else:
        relative_loss_decrease = (losses[0] - losses[-1]) / max(
            abs(losses[0]), 1e-12
        )
        if relative_loss_decrease < MIN_RELATIVE_LOSS_DECREASE:
            reasons.append(
                "training loss did not decrease by at least "
                f"{100 * MIN_RELATIVE_LOSS_DECREASE:.0f}%"
            )

    action_std_rows = [row.get("action_std") for row in diagnostics]
    valid_action_rows = (
        action_std_rows
        and all(
            isinstance(values, list)
            and values
            and all(_finite(value) for value in values)
            for values in action_std_rows
        )
        and len({len(values) for values in action_std_rows}) == 1
    )
    if not valid_action_rows:
        reasons.append("action std diagnostics are missing or non-finite")
        tail_min_action_std = None
    else:
        tail = action_std_rows[-min(3, len(action_std_rows)) :]
        tail_min_action_std = min(value for values in tail for value in values)
        if tail_min_action_std < MIN_ACTION_STD:
            reasons.append(
                "tail action std is below "
                f"{MIN_ACTION_STD:.2f}, indicating constant-action collapse"
            )

    gradient_values = [
        row.get("gradient_global_norm_mean") for row in diagnostics
    ]
    if not gradient_values or any(not _finite(value) for value in gradient_values):
        reasons.append("gradient diagnostics are missing or non-finite")
        positive_gradient_fraction = None
    else:
        positive_gradient_fraction = sum(
            value > 1e-8 for value in gradient_values
        ) / len(gradient_values)
        if positive_gradient_fraction < MIN_POSITIVE_GRADIENT_FRACTION:
            reasons.append(
                "too few epochs have non-zero sampled gradient norms"
            )

    model = summary.get("model", {})
    trainable_parameter_count = model.get("trainable_parameter_count")
    if (
        not isinstance(trainable_parameter_count, int)
        or trainable_parameter_count <= 0
    ):
        reasons.append("trainable parameter count is missing or invalid")
        trainable_parameter_count = None

    def resource_values(field: str) -> list[object]:
        return [row.get(field) for row in diagnostics]

    gpu_peak_values = resource_values("gpu_peak_allocated_bytes")
    rss_peak_values = resource_values("process_peak_rss_bytes")
    wall_time_values = resource_values("wall_time_seconds")
    update_rate_values = resource_values("updates_per_second")
    diagnostic_training_steps = (
        diagnostics[-1].get("step") if diagnostics else None
    )
    resources_valid = (
        diagnostics
        and _finite(diagnostic_training_steps)
        and diagnostic_training_steps > 0
        and all(_finite(value) and value >= 0 for value in gpu_peak_values)
        and all(_finite(value) and value > 0 for value in rss_peak_values)
        and all(_finite(value) and value > 0 for value in wall_time_values)
        and all(_finite(value) and value > 0 for value in update_rate_values)
    )
    if not resources_valid:
        reasons.append("resource diagnostics are missing, non-finite, or non-positive")
        peak_gpu_allocated_bytes = None
        peak_process_rss_bytes = None
        wall_time_seconds = None
        effective_updates_per_second = None
    else:
        peak_gpu_allocated_bytes = int(max(gpu_peak_values))
        peak_process_rss_bytes = int(max(rss_peak_values))
        wall_time_seconds = float(wall_time_values[-1])
        effective_updates_per_second = (
            float(diagnostic_training_steps) / wall_time_seconds
        )

    evaluation = summary.get("evaluation_history", [])
    every_epochs = config["evaluation"]["every_epochs"]
    expected_eval_steps = [
        epoch * steps_per_epoch
        for epoch in range(every_epochs, expected_epochs + 1, every_epochs)
    ]
    actual_eval_steps = [row.get("step") for row in evaluation]
    if actual_eval_steps != expected_eval_steps:
        reasons.append(
            f"evaluation steps {actual_eval_steps!r} do not match "
            f"{expected_eval_steps!r}"
        )

    expected_variants = config["eval_variants"]
    expected_episodes = config["evaluation"]["num_episodes"]
    for record in evaluation:
        rollout = record.get("rollout", {})
        variants = rollout.get("variants", {})
        if set(variants) != set(expected_variants):
            reasons.append(
                f"step {record.get('step')}: rollout variants mismatch"
            )
            continue
        for variant in expected_variants:
            variant_record = variants[variant]
            episodes = variant_record.get("episodes", [])
            if (
                variant_record.get("num_episodes") != expected_episodes
                or len(episodes) != expected_episodes
            ):
                reasons.append(
                    f"step {record.get('step')}: {variant} episode count mismatch"
                )
                continue
            if [episode.get("seed") for episode in episodes] != list(
                range(expected_episodes)
            ):
                reasons.append(
                    f"step {record.get('step')}: {variant} episode seeds mismatch"
                )
            successful_count = sum(
                bool(episode.get("success")) for episode in episodes
            )
            if (
                variant_record.get("successful_episode_count")
                != successful_count
            ):
                reasons.append(
                    f"step {record.get('step')}: "
                    f"{variant} successful episode count mismatch"
                )

    validation_mse = [
        row.get("validation", {}).get("action_mse_sum")
        for row in evaluation
    ]
    if not validation_mse or any(
        not _finite(value) for value in validation_mse
    ):
        reasons.append("validation action MSE is missing or non-finite")
        validation_mse_ratio = None
    else:
        validation_mse_ratio = validation_mse[-1] / max(
            validation_mse[0], 1e-12
        )
        if validation_mse_ratio > MAX_VALIDATION_MSE_RATIO:
            reasons.append(
                "validation action MSE worsened by more than 5%"
            )

    artifact_steps: list[int] = []
    if run_dir is not None:
        artifacts = summary.get("checkpoint_artifacts", {})
        for step in expected_eval_steps:
            record = artifacts.get(str(step), {})
            checkpoint = record.get("checkpoint")
            trainer_state = record.get("trainer_state")
            if checkpoint is None or not Path(checkpoint).is_file():
                reasons.append(f"step {step}: checkpoint artifact is missing")
            if trainer_state is None or not Path(trainer_state).is_file():
                reasons.append(f"step {step}: trainer-state artifact is missing")
            if (
                checkpoint is not None
                and trainer_state is not None
                and Path(checkpoint).is_file()
                and Path(trainer_state).is_file()
            ):
                artifact_steps.append(step)
        model_path = summary.get("model_path")
        if model_path is None or not Path(model_path).is_file():
            reasons.append("final model artifact is missing")

    return {
        "status": (
            "invalid"
            if infrastructure_failure
            else ("stable" if not reasons else "unstable")
        ),
        "reasons": reasons,
        "criteria": {
            "expected_n_steps": expected_n_steps,
            "minimum_relative_loss_decrease": MIN_RELATIVE_LOSS_DECREASE,
            "minimum_tail_action_std": MIN_ACTION_STD,
            "minimum_positive_gradient_fraction": (
                MIN_POSITIVE_GRADIENT_FRACTION
            ),
            "maximum_validation_mse_ratio": MAX_VALIDATION_MSE_RATIO,
        },
        "metrics": {
            "relative_loss_decrease": relative_loss_decrease,
            "tail_min_action_std": tail_min_action_std,
            "positive_gradient_fraction": positive_gradient_fraction,
            "validation_mse_ratio": validation_mse_ratio,
            "evaluation_steps": actual_eval_steps,
            "complete_artifact_steps": artifact_steps,
            "trainable_parameter_count": trainable_parameter_count,
            "peak_gpu_allocated_bytes": peak_gpu_allocated_bytes,
            "peak_process_rss_bytes": peak_process_rss_bytes,
            "wall_time_seconds": wall_time_seconds,
            "effective_updates_per_second": effective_updates_per_second,
            "observed_training_steps": observed_training_steps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--expected-n-steps",
        type=int,
        default=50_000,
        choices=(50_000, 100_000),
        help=(
            "Expected total update budget. 100k is reserved for an explicitly "
            "approved slow-start continuation; all stability thresholds stay "
            "unchanged."
        ),
    )
    args = parser.parse_args()
    summary = load_range_summary(args.run_dir)
    config = yaml.safe_load(
        (args.run_dir / "config.yaml").read_text(encoding="utf-8")
    )
    result = assess_range(
        summary,
        config,
        run_dir=args.run_dir,
        expected_n_steps=args.expected_n_steps,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output = args.run_dir / "range_assessment.json"
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"assessment: {output}")
    if result["status"] == "unstable":
        raise SystemExit(1)
    if result["status"] == "invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
