from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from baselines.experiments.bc_scaling_20260723.assess_range import (
    assess_range,
    load_range_summary,
)
from baselines.experiments.bc_scaling_20260723.assess_slow_start_extension import (
    assess_slow_start_extension,
)


def _config() -> dict:
    return {
        "n_steps": 50_000,
        "n_steps_per_epoch": 5_000,
        "eval_variants": ["train-map", "test-map"],
        "evaluation": {
            "every_epochs": 2,
            "num_episodes": 30,
        },
    }


def _summary() -> dict:
    diagnostics = []
    history = []
    for epoch in range(1, 11):
        history.append(
            {
                "epoch": epoch,
                "metrics": {"loss": 0.50 - 0.025 * epoch},
            }
        )
        diagnostics.append(
            {
                "epoch": epoch,
                "step": epoch * 5_000,
                "action_std": [0.30 + 0.01 * epoch, 0.35],
                "gradient_global_norm_mean": 1.0,
                "gpu_peak_allocated_bytes": 1_000_000 + epoch,
                "process_peak_rss_bytes": 2_000_000 + epoch,
                "wall_time_seconds": 100.0 * epoch,
                "updates_per_second": 50.0,
            }
        )
    evaluation = []
    for index, step in enumerate(range(10_000, 50_001, 10_000)):
        evaluation.append(
            {
                "step": step,
                "validation": {"action_mse_sum": 0.8 - 0.1 * index},
                "rollout": {
                    "variants": {
                        variant: {
                            "num_episodes": 30,
                            "successful_episode_count": 15,
                            "episodes": [
                                {
                                    "seed": episode_index,
                                    "success": episode_index < 15,
                                }
                                for episode_index in range(30)
                            ],
                        }
                        for variant in ("train-map", "test-map")
                    }
                },
            }
        )
    return {
        "model": {"trainable_parameter_count": 12_345},
        "training_history": history,
        "training_diagnostics": diagnostics,
        "evaluation_history": evaluation,
    }


class ScalingRangeAssessmentTest(unittest.TestCase):
    def test_zero_update_partial_run_is_invalid_infrastructure_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "d3rlpy_logs" / "training").mkdir(parents=True)

            summary = load_range_summary(run_dir)
            result = assess_range(summary, _config(), run_dir=run_dir)

        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["metrics"]["observed_training_steps"])
        self.assertIn(
            "must not trigger a learning-rate change",
            " ".join(result["reasons"]),
        )

    def test_reconstructs_early_stopped_range_as_unstable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            log_dir = run_dir / "d3rlpy_logs" / "training"
            log_dir.mkdir(parents=True)
            (log_dir / "loss.csv").write_text(
                "1,5000,1.5759\n2,10000,1.5758\n",
                encoding="utf-8",
            )
            diagnostics = [
                {
                    "epoch": epoch,
                    "step": epoch * 5_000,
                    "action_std": [0.0, 0.0],
                    "gradient_global_norm_mean": 0.0,
                    "gpu_peak_allocated_bytes": 1_000_000,
                    "process_peak_rss_bytes": 2_000_000,
                    "wall_time_seconds": 100.0 * epoch,
                    "updates_per_second": 50.0,
                }
                for epoch in (1, 2)
            ]
            (run_dir / "training_diagnostics.jsonl").write_text(
                "".join(
                    f"{json.dumps(row)}\n"
                    for row in diagnostics
                ),
                encoding="utf-8",
            )

            summary = load_range_summary(run_dir)
            result = assess_range(summary, _config(), run_dir=run_dir)

        self.assertEqual(result["status"], "unstable")
        self.assertEqual(len(summary["training_history"]), 2)
        reasons = " ".join(result["reasons"])
        self.assertIn("ended before summary.json", reasons)
        self.assertIn("constant-action collapse", reasons)
        self.assertIn("non-zero sampled gradient", reasons)

    def test_accepts_complete_numerically_stable_range(self) -> None:
        result = assess_range(_summary(), _config())

        self.assertEqual(result["status"], "stable")
        self.assertEqual(result["reasons"], [])
        self.assertGreater(result["metrics"]["relative_loss_decrease"], 0.1)
        self.assertGreater(result["metrics"]["tail_min_action_std"], 0.05)
        self.assertEqual(
            result["metrics"]["trainable_parameter_count"], 12_345
        )
        self.assertEqual(
            result["metrics"]["peak_gpu_allocated_bytes"], 1_000_010
        )
        self.assertEqual(
            result["metrics"]["peak_process_rss_bytes"], 2_000_010
        )
        self.assertEqual(result["metrics"]["wall_time_seconds"], 1_000.0)
        self.assertEqual(
            result["metrics"]["effective_updates_per_second"], 50.0
        )

    def test_accepts_100k_continuation_without_changing_thresholds(self) -> None:
        config = _config()
        config["n_steps"] = 100_000
        summary = _summary()
        for epoch in range(11, 21):
            summary["training_history"].append(
                {
                    "epoch": epoch,
                    "metrics": {"loss": 0.25 - 0.005 * (epoch - 10)},
                }
            )
            summary["training_diagnostics"].append(
                {
                    "epoch": epoch,
                    "step": epoch * 5_000,
                    "action_std": [0.40, 0.35],
                    "gradient_global_norm_mean": 1.0,
                    "gpu_peak_allocated_bytes": 1_000_000 + epoch,
                    "process_peak_rss_bytes": 2_000_000 + epoch,
                    "wall_time_seconds": 100.0 * epoch,
                    "updates_per_second": 50.0,
                }
            )
        for index, step in enumerate(range(60_000, 100_001, 10_000), 5):
            summary["evaluation_history"].append(
                {
                    "step": step,
                    "validation": {"action_mse_sum": 0.8 - 0.05 * index},
                    "rollout": {
                        "variants": {
                            variant: {
                                "num_episodes": 30,
                                "successful_episode_count": 15,
                                "episodes": [
                                    {
                                        "seed": episode_index,
                                        "success": episode_index < 15,
                                    }
                                    for episode_index in range(30)
                                ],
                            }
                            for variant in ("train-map", "test-map")
                        }
                    },
                }
            )

        result = assess_range(
            summary, config, expected_n_steps=100_000
        )

        self.assertEqual(result["status"], "stable")
        self.assertEqual(result["criteria"]["expected_n_steps"], 100_000)
        self.assertEqual(
            result["criteria"]["minimum_tail_action_std"], 0.05
        )

    def test_slow_start_gate_requires_seven_active_ant_action_dims(self) -> None:
        config = _config()
        diagnostics = _summary()["training_diagnostics"]
        for row in diagnostics[-3:]:
            row["action_std"] = [0.30] * 7 + [0.0]
        summary = {"training_diagnostics": diagnostics}
        assessment = {
            "status": "unstable",
            "reasons": [
                "tail action std is below 0.05, indicating "
                "constant-action collapse"
            ],
            "metrics": {
                "observed_training_steps": 50_000,
                "relative_loss_decrease": 0.50,
                "positive_gradient_fraction": 1.0,
                "validation_mse_ratio": 0.70,
            },
        }

        eligible = assess_slow_start_extension(
            summary, config, assessment
        )
        for row in diagnostics[-3:]:
            row["action_std"][-2] = 0.0
        ineligible = assess_slow_start_extension(
            summary, config, assessment
        )

        self.assertEqual(eligible["status"], "eligible")
        self.assertEqual(eligible["metrics"]["active_action_dims"], 7)
        self.assertEqual(ineligible["status"], "ineligible")
        self.assertEqual(ineligible["metrics"]["active_action_dims"], 6)

    def test_rejects_constant_action_zero_gradient_and_flat_loss(self) -> None:
        summary = _summary()
        for row in summary["training_history"]:
            row["metrics"]["loss"] = 1.574
        for row in summary["training_diagnostics"]:
            row["action_std"] = [0.0, 0.0]
            row["gradient_global_norm_mean"] = 0.0

        result = assess_range(summary, _config())

        self.assertEqual(result["status"], "unstable")
        reasons = " ".join(result["reasons"])
        self.assertIn("training loss did not decrease", reasons)
        self.assertIn("constant-action collapse", reasons)
        self.assertIn("non-zero sampled gradient", reasons)

    def test_rejects_incomplete_rollout_and_worsening_validation(self) -> None:
        summary = _summary()
        summary["evaluation_history"][-1]["rollout"]["variants"][
            "test-map"
        ]["num_episodes"] = 29
        summary["evaluation_history"][-1]["validation"][
            "action_mse_sum"
        ] = 1.0

        result = assess_range(summary, _config())

        self.assertEqual(result["status"], "unstable")
        reasons = " ".join(result["reasons"])
        self.assertIn("episode count mismatch", reasons)
        self.assertIn("validation action MSE worsened", reasons)

    def test_rejects_non_finite_diagnostics(self) -> None:
        summary = copy.deepcopy(_summary())
        summary["training_diagnostics"][-1]["action_std"][0] = float("nan")

        result = assess_range(summary, _config())

        self.assertEqual(result["status"], "unstable")
        self.assertIn(
            "action std diagnostics are missing or non-finite",
            result["reasons"],
        )

    def test_rejects_missing_model_and_resource_diagnostics(self) -> None:
        summary = _summary()
        summary.pop("model")
        for row in summary["training_diagnostics"]:
            row.pop("gpu_peak_allocated_bytes")

        result = assess_range(summary, _config())

        self.assertEqual(result["status"], "unstable")
        self.assertIn(
            "trainable parameter count is missing or invalid",
            result["reasons"],
        )
        self.assertIn(
            "resource diagnostics are missing, non-finite, or non-positive",
            result["reasons"],
        )

    def test_rejects_wrong_episode_seeds_and_success_count(self) -> None:
        summary = _summary()
        record = summary["evaluation_history"][0]["rollout"]["variants"][
            "test-map"
        ]
        record["episodes"][-1]["seed"] = 999
        record["successful_episode_count"] = 14

        result = assess_range(summary, _config())

        reasons = " ".join(result["reasons"])
        self.assertIn("episode seeds mismatch", reasons)
        self.assertIn("successful episode count mismatch", reasons)

    def test_cli_mode_requires_checkpoint_sidecars_and_final_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = assess_range(
                _summary(), _config(), run_dir=Path(directory)
            )

        self.assertEqual(result["status"], "unstable")
        reasons = " ".join(result["reasons"])
        self.assertIn("checkpoint artifact is missing", reasons)
        self.assertIn("trainer-state artifact is missing", reasons)
        self.assertIn("final model artifact is missing", reasons)


if __name__ == "__main__":
    unittest.main()
