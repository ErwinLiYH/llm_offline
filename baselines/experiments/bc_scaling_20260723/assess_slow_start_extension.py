"""Gate a same-checkpoint 50k -> 100k slow-start range continuation.

This does not reclassify the original 50k range.  It only permits a longer
diagnostic when the lowest learning-rate candidate learned normally in every
respect except that exactly one action dimension still has low tail variance.
The 100k continuation must subsequently pass the unchanged full-range
stability criteria.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

if __package__:
    from .assess_range import MIN_ACTION_STD
else:
    # Keep the documented ``python path/to/script.py`` entry point working.
    from assess_range import MIN_ACTION_STD


SOURCE_N_STEPS = 50_000
EXTENDED_N_STEPS = 100_000
STEPS_PER_EPOCH = 5_000
TAIL_EPOCHS = 3
REQUIRED_ACTION_DIMS = 8
MIN_ACTIVE_ACTION_DIMS = 7
ACTION_STD_REASON_PREFIX = "tail action std is below"


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def assess_slow_start_extension(
    summary: dict,
    config: dict,
    range_assessment: dict,
) -> dict:
    reasons: list[str] = []
    if config.get("n_steps") != SOURCE_N_STEPS:
        reasons.append("source range must contain exactly 50k updates")
    if config.get("n_steps_per_epoch") != STEPS_PER_EPOCH:
        reasons.append("source range must use 5k-update epochs")
    if range_assessment.get("status") != "unstable":
        reasons.append("source 50k range must be classified unstable")

    original_reasons = range_assessment.get("reasons")
    if (
        not isinstance(original_reasons, list)
        or len(original_reasons) != 1
        or not isinstance(original_reasons[0], str)
        or not original_reasons[0].startswith(ACTION_STD_REASON_PREFIX)
    ):
        reasons.append(
            "the only 50k failure may be the unchanged tail action-std criterion"
        )

    diagnostics = summary.get("training_diagnostics", [])
    expected_epochs = SOURCE_N_STEPS // STEPS_PER_EPOCH
    if len(diagnostics) != expected_epochs:
        reasons.append(
            f"expected {expected_epochs} source diagnostic rows, "
            f"got {len(diagnostics)}"
        )

    tail = diagnostics[-TAIL_EPOCHS:]
    action_rows = [row.get("action_std") for row in tail]
    rows_valid = (
        len(action_rows) == TAIL_EPOCHS
        and all(
            isinstance(values, list)
            and len(values) == REQUIRED_ACTION_DIMS
            and all(_finite(value) for value in values)
            for values in action_rows
        )
    )
    if not rows_valid:
        reasons.append(
            "the final three diagnostics must contain eight finite action stds"
        )
        per_dimension_tail_min = None
        active_action_dims = None
    else:
        per_dimension_tail_min = [
            min(values[dimension] for values in action_rows)
            for dimension in range(REQUIRED_ACTION_DIMS)
        ]
        active_action_dims = sum(
            value >= MIN_ACTION_STD for value in per_dimension_tail_min
        )
        if active_action_dims < MIN_ACTIVE_ACTION_DIMS:
            reasons.append(
                "fewer than seven of eight action dimensions are active "
                "throughout the final three 50k epochs"
            )

    metrics = range_assessment.get("metrics", {})
    observed_steps = metrics.get("observed_training_steps")
    if observed_steps != SOURCE_N_STEPS:
        reasons.append("source assessment did not observe all 50k updates")

    return {
        "status": "eligible" if not reasons else "ineligible",
        "reasons": reasons,
        "criteria": {
            "source_n_steps": SOURCE_N_STEPS,
            "extended_n_steps": EXTENDED_N_STEPS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "tail_epochs": TAIL_EPOCHS,
            "required_action_dims": REQUIRED_ACTION_DIMS,
            "minimum_active_action_dims": MIN_ACTIVE_ACTION_DIMS,
            "minimum_action_std": MIN_ACTION_STD,
            "final_100k_stability_thresholds_unchanged": True,
        },
        "metrics": {
            "active_action_dims": active_action_dims,
            "per_dimension_tail_min_action_std": per_dimension_tail_min,
            "source_observed_training_steps": observed_steps,
            "source_relative_loss_decrease": metrics.get(
                "relative_loss_decrease"
            ),
            "source_positive_gradient_fraction": metrics.get(
                "positive_gradient_fraction"
            ),
            "source_validation_mse_ratio": metrics.get(
                "validation_mse_ratio"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    summary = json.loads(
        (args.run_dir / "summary.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load(
        (args.run_dir / "config.yaml").read_text(encoding="utf-8")
    )
    range_assessment = json.loads(
        (args.run_dir / "range_assessment.json").read_text(encoding="utf-8")
    )
    result = assess_slow_start_extension(summary, config, range_assessment)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output = args.run_dir / "slow_start_extension_assessment.json"
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"assessment: {output}")
    if result["status"] != "eligible":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
