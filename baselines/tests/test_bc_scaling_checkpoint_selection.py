from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.experiments.bc_scaling_20260723.audit_results import (
    CONFIGS,
    REQUIRED_BUDGET_CONFIGS,
    _activated_formal_config_ids,
    _activated_budget_config_ids,
    _activated_extension_config_ids,
    _audit_ant_depth_recipe,
    _audit_conditional_range_requirement,
    _audit_core_width_rescue_ladder,
    _audit_depth_recipe,
    _audit_depth_range_ladder,
    _audit_followup_width_ladder,
    _audit_width_range,
    _select_width_gate_evidence,
    _conditional_prerequisite_failures,
    _assert_no_formal_collapse,
    _discover_extension_config_ids,
    _discover_run_config_ids,
    _group_budget_extension_recommendations,
    _resolve_run_dir,
    allowed_learning_rates,
    assess_axis_early_stop,
    assess_next_depth_range_gate,
    assess_next_width_range_gate,
    expected_learning_rate,
    hierarchical_bootstrap_checkpoint_difference,
    select_best_checkpoint,
)


STEPS = (100_000, 200_000, 300_000, 400_000, 500_000)


def _record(train: list[float], test: list[float]) -> dict:
    return {
        "status": "complete",
        "experiment_id": f"synthetic-{train[0]:.3f}",
        "checkpoints": [
            {
                "step": step,
                "checkpoint": f"/synthetic/step_{step}.d3",
                "trainer_state": f"/synthetic/trainer_state_step_{step}.pt",
                "train": {"success_macro": train_value},
                "test": {"success_macro": test_value},
            }
            for step, train_value, test_value in zip(
                STEPS, train, test, strict=True
            )
        ],
    }


def _bootstrap_record(seed: int, values: dict[str, list[bool]]) -> dict:
    return {
        "status": "complete",
        "experiment_id": f"bootstrap-s{seed}",
        "checkpoints": [
            {
                "step": 100_000,
                "test": {"per_variant_episode_success": values},
            }
        ],
    }


def _add_range_assessment(
    runs_root: Path, run_id: str, status: str
) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir()
    (run_dir / "range_assessment.json").write_text(
        json.dumps({"status": status, "metrics": {}, "reasons": []}),
        encoding="utf-8",
    )


class CheckpointSelectionTest(unittest.TestCase):
    def test_first_conditional_point_run_activates_three_seed_formal_matrix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            (
                runs_root
                / "bc-scaling-formal-pointmaze-r512-w256-s1-500k-20260726"
            ).mkdir()
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                activated = _activated_formal_config_ids("pointmaze")

        self.assertIn("r512-w256", activated)
        self.assertNotIn("r1024-w256", activated)
        self.assertNotIn("w5120-d4", activated)

    def test_conditional_w5120_is_registered_for_future_audit(self) -> None:
        recipe = CONFIGS["w5120-d4"]
        self.assertEqual(recipe["architecture"], "plain_mlp")
        self.assertEqual(recipe["width"], 5120)
        self.assertEqual(recipe["body_depth"], 4)
        self.assertEqual(
            allowed_learning_rates(recipe, "pointmaze"),
            (3e-4, 1e-4, 3e-5, 1e-5),
        )

    def test_conditional_w6144_is_registered_for_future_audit(self) -> None:
        recipe = CONFIGS["w6144-d4"]
        self.assertEqual(recipe["architecture"], "plain_mlp")
        self.assertEqual(recipe["width"], 6144)
        self.assertEqual(recipe["body_depth"], 4)
        self.assertEqual(
            allowed_learning_rates(recipe, "pointmaze"),
            (3e-4, 1e-4, 3e-5, 1e-5),
        )

    def test_conditional_w7168_is_registered_for_future_audit(self) -> None:
        recipe = CONFIGS["w7168-d4"]
        self.assertEqual(recipe["architecture"], "plain_mlp")
        self.assertEqual(recipe["width"], 7168)
        self.assertEqual(recipe["body_depth"], 4)
        self.assertEqual(
            allowed_learning_rates(recipe, "pointmaze"),
            (3e-4, 1e-4, 3e-5, 1e-5),
        )

    def test_conditional_w8192_is_registered_for_future_audit(self) -> None:
        recipe = CONFIGS["w8192-d4"]
        self.assertEqual(recipe["architecture"], "plain_mlp")
        self.assertEqual(recipe["width"], 8192)
        self.assertEqual(recipe["body_depth"], 4)
        self.assertEqual(
            allowed_learning_rates(recipe, "pointmaze"),
            (3e-4, 1e-4, 3e-5, 1e-5),
        )

    def test_r128_is_registered_between_r64_and_r256(self) -> None:
        self.assertEqual(
            CONFIGS["r128-w256"]["body_depth"],
            128,
        )
        self.assertEqual(
            allowed_learning_rates(
                CONFIGS["r128-w256"], "pointmaze"
            ),
            (3e-4, 1e-4, 3e-5, 1e-5),
        )

    def test_w3072_point_rescue_uses_lower_lr_ladder(self) -> None:
        self.assertEqual(
            allowed_learning_rates(CONFIGS["w3072-d4"], "pointmaze"),
            (1e-4, 3e-5, 1e-5),
        )
        self.assertEqual(
            allowed_learning_rates(CONFIGS["w3072-d4"], "antmaze"),
            (3e-4,),
        )
        self.assertEqual(
            allowed_learning_rates(CONFIGS["w4096-d4"], "pointmaze"),
            (1e-4, 3e-5, 1e-5),
        )

    def test_core_width_rescue_requires_collapse_then_first_stable_lr(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            pilot = (
                runs_root
                / "bc-scaling-formal-pointmaze-w3072-d4-s2-500k-x"
            )
            pilot.mkdir()
            (pilot / "config.yaml").write_text(
                "algorithm_config:\n  learning_rate: 0.0003\n",
                encoding="utf-8",
            )
            (pilot / "training_diagnostics.jsonl").write_text(
                json.dumps(
                    {
                        "step": 25_000,
                        "action_std": [0.0, 0.0],
                        "gradient_global_norm_mean": 4.3e-7,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                missing = _audit_core_width_rescue_ladder(
                    "pointmaze", "w3072-d4"
                )

            range_dir = (
                runs_root
                / "bc-scaling-range-pointmaze-w3072-d4-s2-lr1e4-50k-x"
            )
            range_dir.mkdir()
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                in_progress = _audit_core_width_rescue_ladder(
                    "pointmaze", "w3072-d4"
                )
            (range_dir / "range_assessment.json").write_text(
                json.dumps(
                    {"status": "stable", "metrics": {}, "reasons": []}
                ),
                encoding="utf-8",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                stable = _audit_core_width_rescue_ladder(
                    "pointmaze", "w3072-d4"
                )

        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["next_required_candidate"], "lr1e4")
        self.assertEqual(
            missing["range_evidence"][0]["status"], "collapsed"
        )
        self.assertEqual(in_progress["status"], "in_progress")
        self.assertEqual(stable["status"], "stable")
        self.assertEqual(stable["selected_learning_rate"], 1e-4)

    def test_w4096_waits_for_w3072_recipe_and_starts_from_its_lr(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                blocked = _audit_followup_width_ladder(
                    "pointmaze",
                    "w4096-d4",
                    {"status": "awaiting_formal_matrix"},
                )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-pointmaze-w4096-d4-s2-lr1e4-50k-x",
                "stable",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                stable = _audit_followup_width_ladder(
                    "pointmaze",
                    "w4096-d4",
                    {
                        "status": "valid",
                        "selected_learning_rate": 1e-4,
                    },
                )

        self.assertEqual(blocked["status"], "prerequisite_incomplete")
        self.assertEqual(stable["status"], "stable")
        self.assertEqual(stable["starting_learning_rate"], 1e-4)
        self.assertEqual(stable["selected_learning_rate"], 1e-4)

    def test_run_resolution_can_select_rescued_lr_without_deleting_pilot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            for date, learning_rate in (
                ("20260726", 3e-4),
                ("20260727", 1e-4),
            ):
                run_dir = (
                    runs_root
                    / (
                        "bc-scaling-formal-pointmaze-w3072-d4-"
                        f"s0-500k-{date}"
                    )
                )
                run_dir.mkdir()
                (run_dir / "config.yaml").write_text(
                    "algorithm_config:\n"
                    f"  learning_rate: {learning_rate}\n",
                    encoding="utf-8",
                )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                resolved = _resolve_run_dir(
                    "pointmaze",
                    "w3072-d4",
                    0,
                    required_learning_rate=1e-4,
                )

        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.name.endswith("20260727"))

    def test_depth_range_ladder_reports_next_candidate_and_first_stable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                missing = _audit_depth_range_ladder(
                    "pointmaze", "r128-w256"
                )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-pointmaze-r128-w256-s0-lr3e4-50k-x",
                "unstable",
            )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-pointmaze-r128-w256-s0-lr3e4-retry-50k-x",
                "unstable",
            )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-pointmaze-r128-w256-s0-lr1e4-50k-x",
                "stable",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                stable = _audit_depth_range_ladder(
                    "pointmaze", "r128-w256"
                )

        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["next_required_candidate"], "lr3e4")
        self.assertEqual(stable["status"], "stable")
        self.assertEqual(stable["selected_learning_rate"], 1e-4)

    def test_depth_range_invalid_retry_does_not_lower_learning_rate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-antmaze-r64-w256-s0-lr3e4-50k-x",
                "invalid",
            )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-antmaze-r64-w256-s0-lr3e4-retry-50k-x",
                "invalid",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                result = _audit_depth_range_ladder(
                    "antmaze", "r64-w256"
                )

        self.assertEqual(
            result["status"], "invalid_infrastructure_evidence"
        )
        self.assertEqual(
            result["next_required_candidate"], "lr3e4-retry2"
        )

    def test_ant_d256_lowest_lr_can_require_stable_100k_extension(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            for label in (
                "lr3e4",
                "lr3e4-retry",
                "lr1e4",
                "lr3e5",
                "lr1e5",
            ):
                _add_range_assessment(
                    runs_root,
                    "bc-scaling-range-antmaze-r256-w256-s0-"
                    f"{label}-50k-x",
                    "unstable",
                )
            source = (
                runs_root
                / "bc-scaling-range-antmaze-r256-w256-s0-lr1e5-50k-x"
            )
            (source / "slow_start_extension_assessment.json").write_text(
                json.dumps(
                    {
                        "status": "eligible",
                        "metrics": {"active_action_dims": 7},
                        "reasons": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                required = _audit_depth_range_ladder(
                    "antmaze", "r256-w256"
                )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-antmaze-r256-w256-s0-"
                "lr1e5-extend100k-x",
                "stable",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                stable = _audit_depth_range_ladder(
                    "antmaze", "r256-w256"
                )

        self.assertEqual(required["status"], "slow_start_extension_required")
        self.assertEqual(
            required["next_required_candidate"], "lr1e5-extend100k"
        )
        self.assertEqual(stable["status"], "stable")
        self.assertEqual(stable["selected_learning_rate"], 1e-5)
        self.assertEqual(len(stable["range_evidence"]), 6)

    def test_ant_d256_failed_extension_requires_posthoc_lr3e6_range(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            for label in (
                "lr3e4",
                "lr3e4-retry",
                "lr1e4",
                "lr3e5",
                "lr1e5",
            ):
                _add_range_assessment(
                    runs_root,
                    "bc-scaling-range-antmaze-r256-w256-s0-"
                    f"{label}-50k-x",
                    "unstable",
                )
            source = (
                runs_root
                / "bc-scaling-range-antmaze-r256-w256-s0-lr1e5-50k-x"
            )
            (source / "slow_start_extension_assessment.json").write_text(
                json.dumps(
                    {
                        "status": "eligible",
                        "metrics": {"active_action_dims": 7},
                        "reasons": [],
                    }
                ),
                encoding="utf-8",
            )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-antmaze-r256-w256-s0-"
                "lr1e5-extend100k-x",
                "unstable",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                required = _audit_depth_range_ladder(
                    "antmaze", "r256-w256"
                )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-antmaze-r256-w256-s0-lr3e6-50k-x",
                "stable",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                stable = _audit_depth_range_ladder(
                    "antmaze", "r256-w256"
                )

        self.assertEqual(
            required["status"], "posthoc_lr_rescue_required"
        )
        self.assertEqual(required["next_required_candidate"], "lr3e6")
        self.assertEqual(stable["status"], "stable")
        self.assertEqual(stable["selected_learning_rate"], 3e-6)
        self.assertEqual(len(stable["range_evidence"]), 7)

    def test_incomplete_formal_depth_recipe_retains_range_evidence(
        self,
    ) -> None:
        range_evidence = [
            {"experiment_id": "range-a", "status": "unstable"},
            {"experiment_id": "range-b", "status": "in_progress"},
        ]
        with patch(
            "baselines.experiments.bc_scaling_20260723.audit_results."
            "_audit_depth_range_ladder",
            return_value={
                "status": "in_progress",
                "selected_learning_rate": None,
                "next_required_candidate": "lr3e6",
                "range_evidence": range_evidence,
            },
        ):
            result = _audit_depth_recipe(
                "antmaze", "r256-w256", records=[]
            )

        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["range_status"], "in_progress")
        self.assertEqual(result["next_required_candidate"], "lr3e6")
        self.assertEqual(result["range_evidence"], range_evidence)

    def test_width_range_lowers_lr_only_after_observed_instability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-pointmaze-w5120-d4-s0-lr3e4-50k-x",
                "unstable",
            )
            _add_range_assessment(
                runs_root,
                "bc-scaling-range-pointmaze-w5120-d4-s0-lr1e4-50k-x",
                "stable",
            )
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                result = _audit_width_range(
                    "pointmaze", "w5120-d4"
                )

        self.assertEqual(result["status"], "stable")
        self.assertEqual(result["selected_learning_rate"], 1e-4)
        self.assertEqual(
            [row["status"] for row in result["range_evidence"]],
            ["unstable", "stable"],
        )

    def test_width_gate_prefers_completed_common_budget_pair(self) -> None:
        formal_runs = {"w4096-d4": [{"status": "complete"}]}
        formal_selections = {
            "w3072-d4": {"status": "complete", "selected_step": 500_000},
            "w4096-d4": {"status": "complete", "selected_step": 400_000},
        }
        budget_runs = {"w4096-d4": [{"status": "complete"}] * 3}
        budget_selections = {
            "w3072-d4": {"status": "complete", "selected_step": 700_000},
            "w4096-d4": {"status": "complete", "selected_step": 600_000},
        }

        runs, selections, stage = _select_width_gate_evidence(
            previous_config="w3072-d4",
            current_config="w4096-d4",
            formal_runs=formal_runs,
            formal_selections=formal_selections,
            budget_runs=budget_runs,
            budget_selections=budget_selections,
        )

        self.assertIs(runs, budget_runs)
        self.assertIs(selections, budget_selections)
        self.assertEqual(stage, "1m_common_budget")

    def test_axis_early_stop_preempts_conditional_range_requirement(
        self,
    ) -> None:
        result = _audit_conditional_range_requirement(
            family="pointmaze",
            config_id="r512-w256",
            axis="depth",
            gate={"status": "eligible_for_50k_range"},
            early_stop_gate={"status": "stop"},
        )

        self.assertEqual(result["status"], "stopped_by_axis_early_stop")

    def test_user_decision_preempts_conditional_range_requirement(
        self,
    ) -> None:
        result = _audit_conditional_range_requirement(
            family="pointmaze",
            config_id="w8192-d4",
            axis="width",
            gate={
                "status": "stopped_by_user_decision",
                "protocol_gate_status": "eligible_for_50k_range",
                "decision_reason": "observed width plateau",
            },
            early_stop_gate={"status": "continue"},
        )

        self.assertEqual(result["status"], "stopped_by_user_decision")
        self.assertEqual(
            result["protocol_gate_status"], "eligible_for_50k_range"
        )
        self.assertEqual(result["decision_reason"], "observed width plateau")

    def test_conditional_depth_requires_valid_previous_formal_recipe(
        self,
    ) -> None:
        requirements = {
            "pointmaze": {
                "r512-w256": {
                    "status": "stable",
                    "formal_recipe": {"status": "awaiting_formal_matrix"},
                }
            },
            "antmaze": {},
        }

        self.assertEqual(
            _conditional_prerequisite_failures(
                family="pointmaze",
                config_id="r1024-w256",
                requirements=requirements,
            ),
            ["pointmaze/r512-w256"],
        )
        requirements["pointmaze"]["r512-w256"][
            "formal_recipe"
        ] = {"status": "valid"}
        self.assertEqual(
            _conditional_prerequisite_failures(
                family="pointmaze",
                config_id="r1024-w256",
                requirements=requirements,
            ),
            [],
        )
        self.assertEqual(
            _conditional_prerequisite_failures(
                family="antmaze",
                config_id="r512-w256",
                requirements=requirements,
            ),
            [],
        )
        requirements["pointmaze"]["w5120-d4"] = {
            "status": "stable",
            "formal_recipe": {"status": "awaiting_formal_matrix"},
        }
        self.assertEqual(
            _conditional_prerequisite_failures(
                family="pointmaze",
                config_id="w6144-d4",
                requirements=requirements,
            ),
            ["pointmaze/w5120-d4"],
        )
        requirements["pointmaze"]["w5120-d4"][
            "formal_recipe"
        ] = {"status": "valid"}
        self.assertEqual(
            _conditional_prerequisite_failures(
                family="pointmaze",
                config_id="w6144-d4",
                requirements=requirements,
            ),
            [],
        )
        requirements["pointmaze"]["w6144-d4"] = {
            "status": "stable",
            "formal_recipe": {"status": "awaiting_formal_matrix"},
        }
        self.assertEqual(
            _conditional_prerequisite_failures(
                family="pointmaze",
                config_id="w7168-d4",
                requirements=requirements,
            ),
            ["pointmaze/w6144-d4"],
        )
        requirements["pointmaze"]["w6144-d4"][
            "formal_recipe"
        ] = {"status": "valid"}
        self.assertEqual(
            _conditional_prerequisite_failures(
                family="pointmaze",
                config_id="w7168-d4",
                requirements=requirements,
            ),
            [],
        )
        requirements["pointmaze"]["w7168-d4"] = {
            "status": "stable",
            "formal_recipe": {"status": "awaiting_formal_matrix"},
        }
        self.assertEqual(
            _conditional_prerequisite_failures(
                family="pointmaze",
                config_id="w8192-d4",
                requirements=requirements,
            ),
            ["pointmaze/w7168-d4"],
        )
        requirements["pointmaze"]["w7168-d4"][
            "formal_recipe"
        ] = {"status": "valid"}
        self.assertEqual(
            _conditional_prerequisite_failures(
                family="pointmaze",
                config_id="w8192-d4",
                requirements=requirements,
            ),
            [],
        )

    def test_first_1m_run_activates_three_seed_budget_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            (
                runs_root
                / "bc-scaling-budget-pointmaze-r256-w256-s2-u1m-20260726"
            ).mkdir()
            (
                runs_root
                / "bc-scaling-formal-pointmaze-r64-w256-s0-500k-20260726"
            ).mkdir()
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                discovered = _discover_run_config_ids(
                    "pointmaze", updates=1_000_000, run_kind="budget"
                )

        self.assertEqual(discovered, ("r256-w256",))

    def test_point_budget_recommendation_activates_matrix_before_run_exists(
        self,
    ) -> None:
        selections = {
            config_id: {
                "status": "complete",
                "budget_extension": {
                    "recommended": config_id == "p4-w256"
                },
            }
            for config_id in CONFIGS
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "baselines.experiments.bc_scaling_20260723."
            "audit_results.RUNS_ROOT",
            Path(directory),
        ):
            activated = _activated_budget_config_ids(
                "pointmaze", selections
            )

        self.assertEqual(
            activated,
            ("r4-w256", "p4-w256", "w512-d4"),
        )

    def test_ant_primary_budget_matrix_does_not_depend_on_recommendations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "baselines.experiments.bc_scaling_20260723."
            "audit_results.RUNS_ROOT",
            Path(directory),
        ):
            activated = _activated_budget_config_ids("antmaze", {})

        self.assertEqual(activated, REQUIRED_BUDGET_CONFIGS["antmaze"])

    def test_first_2m_run_activates_three_seed_extension_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            (
                runs_root
                / "bc-scaling-extension-antmaze-w2048-d4-s1-u2m-20260726"
            ).mkdir()
            (
                runs_root
                / "bc-scaling-preflight-antmaze-w4096-d4-s0-1update-20260726"
            ).mkdir()
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                discovered = _discover_extension_config_ids("antmaze")

        self.assertEqual(discovered, ("w2048-d4",))

    def test_ant_2m_recommendation_activates_pair_before_run_exists(self) -> None:
        selections = {
            config_id: {
                "status": "complete",
                "budget_extension": {
                    "recommended": config_id == "p4-w256"
                },
            }
            for config_id in CONFIGS
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "baselines.experiments.bc_scaling_20260723."
            "audit_results.RUNS_ROOT",
            Path(directory),
        ):
            activated = _activated_extension_config_ids(
                "antmaze", selections
            )

        self.assertEqual(activated, ("p4-w256", "w512-d4"))

    def test_point_does_not_infer_unplanned_2m_from_1m_selection(self) -> None:
        selections = {
            "p4-w256": {
                "status": "complete",
                "budget_extension": {"recommended": True},
            }
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "baselines.experiments.bc_scaling_20260723."
            "audit_results.RUNS_ROOT",
            Path(directory),
        ):
            activated = _activated_extension_config_ids(
                "pointmaze", selections
            )

        self.assertEqual(activated, ())

    def test_depth_rescue_learning_rate_is_family_specific(self) -> None:
        r64_recipe = CONFIGS["r64-w256"]
        r256_recipe = CONFIGS["r256-w256"]

        self.assertEqual(expected_learning_rate(r64_recipe, "pointmaze"), 1e-4)
        self.assertEqual(expected_learning_rate(r64_recipe, "antmaze"), 3e-4)
        self.assertEqual(expected_learning_rate(r256_recipe, "pointmaze"), 1e-5)
        self.assertEqual(expected_learning_rate(r256_recipe, "antmaze"), 3e-4)
        self.assertEqual(
            allowed_learning_rates(r64_recipe, "antmaze"),
            (3e-4, 1e-4, 3e-5, 1e-5),
        )
        self.assertEqual(
            allowed_learning_rates(r256_recipe, "pointmaze"), (1e-5,)
        )
        self.assertEqual(
            allowed_learning_rates(r256_recipe, "antmaze"),
            (3e-4, 1e-4, 3e-5, 1e-5, 3e-6),
        )

    def test_ant_depth_recipe_requires_failed_rerun_and_first_stable_lr(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)

            def add_assessment(run_id: str, status: str) -> None:
                run_dir = runs_root / run_id
                run_dir.mkdir()
                (run_dir / "range_assessment.json").write_text(
                    json.dumps(
                        {"status": status, "metrics": {}, "reasons": []}
                    ),
                    encoding="utf-8",
                )

            add_assessment(
                "bc-scaling-range-antmaze-r64-w256-s0-lr3e4-50k-20260726",
                "unstable",
            )
            add_assessment(
                "bc-scaling-range-antmaze-r64-w256-s0-lr3e4-retry-50k-20260726",
                "unstable",
            )
            add_assessment(
                "bc-scaling-range-antmaze-r64-w256-s0-lr1e4-50k-20260726",
                "stable",
            )
            records = [
                {"status": "complete", "learning_rate": 1e-4}
                for _ in range(3)
            ]
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                result = _audit_ant_depth_recipe("r64-w256", records)

        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["selected_learning_rate"], 1e-4)
        self.assertEqual(
            [row["status"] for row in result["range_evidence"]],
            ["unstable", "unstable", "stable"],
        )

    def test_ant_depth_recipe_rejects_skipping_an_earlier_stable_lr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)

            def add_assessment(run_id: str, status: str) -> None:
                run_dir = runs_root / run_id
                run_dir.mkdir()
                (run_dir / "range_assessment.json").write_text(
                    json.dumps(
                        {"status": status, "metrics": {}, "reasons": []}
                    ),
                    encoding="utf-8",
                )

            add_assessment(
                "bc-scaling-range-antmaze-r256-w256-s0-lr3e4-50k-20260726",
                "unstable",
            )
            add_assessment(
                "bc-scaling-range-antmaze-r256-w256-s0-lr3e4-retry-50k-20260726",
                "unstable",
            )
            add_assessment(
                "bc-scaling-range-antmaze-r256-w256-s0-lr1e4-50k-20260726",
                "stable",
            )
            add_assessment(
                "bc-scaling-range-antmaze-r256-w256-s0-lr3e5-50k-20260726",
                "stable",
            )
            records = [
                {"status": "complete", "learning_rate": 3e-5}
                for _ in range(3)
            ]
            with patch(
                "baselines.experiments.bc_scaling_20260723."
                "audit_results.RUNS_ROOT",
                runs_root,
            ):
                with self.assertRaisesRegex(
                    AssertionError, "first stable range learning rate"
                ):
                    _audit_ant_depth_recipe("r256-w256", records)

    def test_ant_d256_same_recipe_retry_resolves_formal_instability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory)
            retry_id = (
                "bc-scaling-range-antmaze-r256-w256-s2-"
                "lr3e6-formal-retry-50k-20260730"
            )
            retry_dir = runs_root / retry_id
            retry_dir.mkdir()
            (retry_dir / "range_assessment.json").write_text(
                json.dumps(
                    {
                        "status": "unstable",
                        "metrics": {"tail_min_action_std": 0.0},
                        "reasons": ["constant-action collapse"],
                    }
                ),
                encoding="utf-8",
            )
            failed_dir = (
                runs_root
                / "bc-scaling-budget-antmaze-r256-w256-s2-u1m-20260726"
            )
            failed_dir.mkdir()
            diagnostics = [
                {
                    "step": step,
                    "action_std": [0.2] * 7 + [0.0],
                }
                for step in (450_000, 475_000, 500_000)
            ]
            (failed_dir / "training_diagnostics.jsonl").write_text(
                "\n".join(json.dumps(row) for row in diagnostics) + "\n",
                encoding="utf-8",
            )
            range_recipe = {
                "status": "incomplete",
                "selected_learning_rate": 3e-6,
                "range_status": "stable",
                "range_evidence": [],
            }
            with (
                patch(
                    "baselines.experiments.bc_scaling_20260723."
                    "audit_results.RUNS_ROOT",
                    runs_root,
                ),
                patch(
                    "baselines.experiments.bc_scaling_20260723."
                    "audit_results._audit_depth_recipe",
                    return_value=range_recipe,
                ),
            ):
                result = _audit_ant_depth_recipe(
                    "r256-w256", records=[]
                )

        self.assertEqual(
            result["status"], "confirmed_cross_seed_instability"
        )
        self.assertEqual(result["selected_learning_rate"], 3e-6)
        self.assertEqual(result["failed_formal_seed"], 2)
        self.assertEqual(result["same_recipe_retry_tail_min_action_std"], 0.0)

    def test_next_depth_gate_allows_range_after_gain_and_plateau(self) -> None:
        previous_records = [
            _record([0.60] * 5, [0.20] * 5) for _ in range(3)
        ]
        current_records = [
            _record(
                [0.60, 0.61, 0.62, 0.63, 0.64],
                [0.24] * 5,
            )
            for _ in range(3)
        ]
        for record in current_records:
            record["model"] = {"trainable_parameter_count": 17_036_290}
            record["resources"] = {
                "peak_gpu_allocated_bytes": 500_000_000,
                "diagnostic_wall_time_seconds": 50_000.0,
            }
        result = assess_next_depth_range_gate(
            family="pointmaze",
            previous_config="r64-w256",
            current_config="r256-w256",
            next_config="r512-w256",
            runs={"r256-w256": current_records},
            selections={
                "r64-w256": select_best_checkpoint(previous_records),
                "r256-w256": select_best_checkpoint(current_records),
            },
        )

        self.assertEqual(result["status"], "eligible_for_50k_range")
        self.assertAlmostEqual(result["selected_test_delta"], 0.04)
        self.assertEqual(result["positive_seed_count"], 3)
        self.assertTrue(result["plateau_gate"])
        self.assertEqual(result["failed_gates"], [])
        self.assertEqual(
            result["resource_projection"]["next_parameter_count"],
            34_010_114,
        )

    def test_next_depth_gate_requires_common_budget_plateau(self) -> None:
        previous_records = [
            _record([0.60] * 5, [0.20] * 5) for _ in range(3)
        ]
        current_records = [
            _record(
                [0.60, 0.61, 0.62, 0.63, 0.64],
                [0.24, 0.24, 0.24, 0.26, 0.27],
            )
            for _ in range(3)
        ]
        for record in current_records:
            record["model"] = {"trainable_parameter_count": 17_036_290}
            record["resources"] = {
                "peak_gpu_allocated_bytes": 500_000_000,
                "diagnostic_wall_time_seconds": 50_000.0,
            }
        result = assess_next_depth_range_gate(
            family="pointmaze",
            previous_config="r64-w256",
            current_config="r256-w256",
            next_config="r512-w256",
            runs={"r256-w256": current_records},
            selections={
                "r64-w256": select_best_checkpoint(previous_records),
                "r256-w256": select_best_checkpoint(current_records),
            },
        )

        self.assertEqual(result["status"], "not_eligible")
        self.assertTrue(result["performance_gate"])
        self.assertFalse(result["plateau_gate"])
        self.assertIn("common_budget_plateau", result["failed_gates"])

    def test_w5120_gate_uses_two_point_gain_and_measured_w4096_cost(self) -> None:
        previous_records = [
            _record([0.60] * 5, [0.20] * 5) for _ in range(3)
        ]
        current_records = [
            _record(
                [0.60, 0.61, 0.62, 0.63, 0.64],
                [0.225] * 5,
            )
            for _ in range(3)
        ]
        for record in current_records:
            record["model"] = {"trainable_parameter_count": 68_157_442}
            record["resources"] = {
                "peak_gpu_allocated_bytes": 1_500_000_000,
                "diagnostic_wall_time_seconds": 80_000.0,
            }
        result = assess_next_width_range_gate(
            family="pointmaze",
            previous_config="w3072-d4",
            current_config="w4096-d4",
            next_config="w5120-d4",
            runs={"w4096-d4": current_records},
            selections={
                "w3072-d4": select_best_checkpoint(previous_records),
                "w4096-d4": select_best_checkpoint(current_records),
            },
        )

        self.assertEqual(result["status"], "eligible_for_50k_range")
        self.assertAlmostEqual(result["minimum_selected_test_delta"], 0.02)
        self.assertAlmostEqual(result["selected_test_delta"], 0.025)
        self.assertEqual(
            result["resource_projection"]["next_parameter_count"],
            106_168_322,
        )
        self.assertEqual(result["next_action"], "run_next_width_50k_range")

    def test_w6144_gate_uses_w4096_w5120_pair_and_measured_cost(self) -> None:
        previous_records = [
            _record([0.60] * 5, [0.38] * 5) for _ in range(3)
        ]
        current_records = [
            _record(
                [0.78, 0.79, 0.80, 0.80, 0.79],
                [0.40, 0.41, 0.42, 0.42, 0.41],
            )
            for _ in range(3)
        ]
        for record in current_records:
            record["learning_rate"] = 1e-4
            record["model"] = {"trainable_parameter_count": 106_168_322}
            record["resources"] = {
                "peak_gpu_allocated_bytes": 2_100_000_000,
                "diagnostic_wall_time_seconds": 100_000.0,
            }
        result = assess_next_width_range_gate(
            family="pointmaze",
            previous_config="w4096-d4",
            current_config="w5120-d4",
            next_config="w6144-d4",
            runs={"w5120-d4": current_records},
            selections={
                "w4096-d4": select_best_checkpoint(previous_records),
                "w5120-d4": select_best_checkpoint(current_records),
            },
        )

        self.assertEqual(result["status"], "eligible_for_50k_range")
        self.assertAlmostEqual(result["selected_test_delta"], 0.04)
        self.assertEqual(result["current_learning_rate"], 1e-4)
        self.assertEqual(
            result["resource_projection"]["next_parameter_count"],
            152_567_810,
        )

    def test_w7168_range_follows_two_consecutive_low_gain_rule(self) -> None:
        previous_records = [
            _record([0.79] * 5, [0.42] * 5) for _ in range(3)
        ]
        current_records = [
            _record(
                [0.72, 0.76, 0.79, 0.79, 0.77],
                [0.24, 0.29, 0.32, 0.31, 0.29],
            )
            for _ in range(3)
        ]
        for record in current_records:
            record["learning_rate"] = 3e-5
            record["model"] = {"trainable_parameter_count": 152_567_810}
            record["resources"] = {
                "peak_gpu_allocated_bytes": 3_100_000_000,
                "diagnostic_wall_time_seconds": 130_000.0,
            }
        result = assess_next_width_range_gate(
            family="pointmaze",
            previous_config="w5120-d4",
            current_config="w6144-d4",
            next_config="w7168-d4",
            runs={"w6144-d4": current_records},
            selections={
                "w5120-d4": select_best_checkpoint(previous_records),
                "w6144-d4": select_best_checkpoint(current_records),
            },
            continue_until_two_low_gains=True,
        )

        self.assertEqual(result["status"], "eligible_for_50k_range")
        self.assertFalse(result["current_gain_required"])
        self.assertFalse(result["plateau_required"])
        self.assertLess(result["selected_test_delta"], 0.0)
        self.assertEqual(result["current_learning_rate"], 3e-5)
        self.assertEqual(
            result["resource_projection"]["next_parameter_count"],
            207_355_906,
        )

    def test_w8192_range_is_resource_projected_when_axis_continues(self) -> None:
        previous_records = [
            _record([0.79] * 5, [0.32] * 5) for _ in range(3)
        ]
        current_records = [
            _record(
                [0.72, 0.76, 0.79, 0.79, 0.78],
                [0.33, 0.35, 0.36, 0.37, 0.38],
            )
            for _ in range(3)
        ]
        for record in current_records:
            record["learning_rate"] = 3e-5
            record["model"] = {"trainable_parameter_count": 207_355_906}
            record["resources"] = {
                "peak_gpu_allocated_bytes": 4_200_000_000,
                "diagnostic_wall_time_seconds": 160_000.0,
            }
        result = assess_next_width_range_gate(
            family="pointmaze",
            previous_config="w6144-d4",
            current_config="w7168-d4",
            next_config="w8192-d4",
            runs={"w7168-d4": current_records},
            selections={
                "w6144-d4": select_best_checkpoint(previous_records),
                "w7168-d4": select_best_checkpoint(current_records),
            },
            continue_until_two_low_gains=True,
        )

        self.assertEqual(result["status"], "eligible_for_50k_range")
        self.assertFalse(result["current_gain_required"])
        self.assertFalse(result["plateau_required"])
        self.assertEqual(result["current_learning_rate"], 3e-5)
        self.assertEqual(
            result["resource_projection"]["next_parameter_count"],
            270_532_610,
        )

    def test_axis_stops_after_two_consecutive_sub_two_pp_gains(self) -> None:
        selections = {
            "w2048-d4": select_best_checkpoint(
                [_record([0.6] * 5, [0.30] * 5) for _ in range(3)]
            ),
            "w3072-d4": select_best_checkpoint(
                [_record([0.6] * 5, [0.315] * 5) for _ in range(3)]
            ),
            "w4096-d4": select_best_checkpoint(
                [_record([0.6] * 5, [0.31] * 5) for _ in range(3)]
            ),
        }

        result = assess_axis_early_stop(
            axis="width",
            config_order=(
                "p4-w256",
                "w512-d4",
                "w1024-d4",
                "w2048-d4",
                "w3072-d4",
                "w4096-d4",
                "w5120-d4",
            ),
            next_config="w5120-d4",
            selections=selections,
        )

        self.assertEqual(result["status"], "stop")
        self.assertEqual(
            result["compared_configs"],
            ["w2048-d4", "w3072-d4", "w4096-d4"],
        )
        self.assertAlmostEqual(result["latest_two_deltas"][0], 0.015)
        self.assertAlmostEqual(result["latest_two_deltas"][1], -0.005)

    def test_axis_continues_when_either_gain_reaches_two_pp(self) -> None:
        selections = {
            "r64-w256": select_best_checkpoint(
                [_record([0.6] * 5, [0.20] * 5) for _ in range(3)]
            ),
            "r128-w256": select_best_checkpoint(
                [_record([0.6] * 5, [0.22] * 5) for _ in range(3)]
            ),
            "r256-w256": select_best_checkpoint(
                [_record([0.6] * 5, [0.21] * 5) for _ in range(3)]
            ),
        }

        result = assess_axis_early_stop(
            axis="depth",
            config_order=(
                "r4-w256",
                "r16-w256",
                "r64-w256",
                "r128-w256",
                "r256-w256",
                "r512-w256",
            ),
            next_config="r512-w256",
            selections=selections,
        )

        self.assertEqual(result["status"], "continue")

    def test_hierarchical_bootstrap_uses_paired_episode_success(self) -> None:
        variants = ["test-a", "test-b"]
        lower = [
            _bootstrap_record(
                seed,
                {variant: [False] * 30 for variant in variants},
            )
            for seed in range(3)
        ]
        upper = [
            _bootstrap_record(
                seed,
                {variant: [True] * 30 for variant in variants},
            )
            for seed in range(3)
        ]

        result = hierarchical_bootstrap_checkpoint_difference(
            lower,
            upper,
            lower_step=100_000,
            upper_step=100_000,
            variants=variants,
            comparison_id="synthetic:lower->upper",
            replicates=200,
        )

        self.assertEqual(result["observed_test_delta"], 1.0)
        self.assertEqual(result["confidence_interval"], [1.0, 1.0])
        self.assertFalse(result["confidence_interval_contains_zero"])
        self.assertEqual(result["individual_seed_deltas"], [1.0, 1.0, 1.0])
        self.assertEqual(result["positive_variant_count"], 2)
        self.assertEqual(result["negative_variant_count"], 0)

    def test_hierarchical_bootstrap_is_deterministic_and_contains_zero(self) -> None:
        variants = ["test-a"]
        records = [
            _bootstrap_record(
                seed,
                {"test-a": [episode % 2 == 0 for episode in range(30)]},
            )
            for seed in range(3)
        ]
        kwargs = {
            "lower_step": 100_000,
            "upper_step": 100_000,
            "variants": variants,
            "comparison_id": "synthetic:same",
            "replicates": 200,
        }

        first = hierarchical_bootstrap_checkpoint_difference(
            records, records, **kwargs
        )
        second = hierarchical_bootstrap_checkpoint_difference(
            records, records, **kwargs
        )

        self.assertEqual(first, second)
        self.assertEqual(first["confidence_interval"], [0.0, 0.0])
        self.assertTrue(first["confidence_interval_contains_zero"])

    def test_formal_collapse_gate_rejects_low_std_and_zero_gradients(self) -> None:
        stable = [
            {
                "action_std": [0.30, 0.40],
                "gradient_global_norm_mean": 1.0,
            }
            for _ in range(10)
        ]
        _assert_no_formal_collapse(stable, run_id="stable")

        collapsed = [dict(record) for record in stable]
        collapsed[-1] = {
            "action_std": [0.01, 0.40],
            "gradient_global_norm_mean": 1.0,
        }
        with self.assertRaisesRegex(AssertionError, "tail action std"):
            _assert_no_formal_collapse(collapsed, run_id="collapsed")

        zero_gradient = [dict(record) for record in stable]
        for record in zero_gradient[:3]:
            record["gradient_global_norm_mean"] = 0.0
        with self.assertRaisesRegex(AssertionError, "positive-gradient"):
            _assert_no_formal_collapse(
                zero_gradient, run_id="zero-gradient"
            )

    def test_selects_shared_pre_overfit_checkpoint(self) -> None:
        records = [
            _record(
                [0.70, 0.75, 0.80, 0.82, 0.83],
                [0.30, 0.35, 0.40, 0.37, 0.34],
            ),
            _record(
                [0.69, 0.76, 0.81, 0.83, 0.84],
                [0.32, 0.36, 0.42, 0.38, 0.35],
            ),
            _record(
                [0.71, 0.74, 0.79, 0.81, 0.82],
                [0.31, 0.37, 0.41, 0.39, 0.36],
            ),
        ]

        selection = select_best_checkpoint(records)

        self.assertEqual(selection["selected_step"], 300_000)
        self.assertEqual(selection["endpoint_step"], 500_000)
        self.assertEqual(
            [row["seed"] for row in selection["selected_checkpoints"]],
            [0, 1, 2],
        )
        self.assertTrue(
            all(
                row["checkpoint"] == "/synthetic/step_300000.d3"
                for row in selection["selected_checkpoints"]
            )
        )
        self.assertTrue(
            all(
                row["trainer_state"]
                == "/synthetic/trainer_state_step_300000.pt"
                for row in selection["selected_checkpoints"]
            )
        )
        self.assertEqual(
            selection["candidate_steps"], [100_000, 200_000, 300_000]
        )
        self.assertTrue(selection["overfit"]["detected"])
        self.assertEqual(selection["overfit"]["peak_step"], 300_000)
        self.assertEqual(selection["overfit"]["decline_start_step"], 400_000)
        self.assertAlmostEqual(
            selection["overfit"]["test_decline_peak_to_endpoint"], -0.06
        )
        self.assertGreater(selection["selected_to_endpoint"]["train"], 0.0)
        self.assertLess(selection["selected_to_endpoint"]["test"], 0.0)
        self.assertGreater(
            selection["selected_to_endpoint"]["generalization_gap"], 0.0
        )
        self.assertAlmostEqual(
            selection["selected"]["generalization_gap"]["mean"],
            selection["selected"]["train"]["mean"]
            - selection["selected"]["test"]["mean"],
        )
        self.assertFalse(selection["budget_extension"]["recommended"])
        self.assertTrue(
            selection["budget_extension"]["blocked_by_sustained_overfit"]
        )

    def test_recovery_prevents_false_sustained_overfit(self) -> None:
        records = [
            _record(
                [0.30, 0.40, 0.45, 0.50, 0.55],
                [0.30, 0.40, 0.35, 0.34, 0.405],
            ),
            _record(
                [0.31, 0.41, 0.46, 0.51, 0.56],
                [0.31, 0.41, 0.36, 0.35, 0.415],
            ),
            _record(
                [0.29, 0.39, 0.44, 0.49, 0.54],
                [0.29, 0.39, 0.34, 0.33, 0.395],
            ),
        ]

        selection = select_best_checkpoint(records)

        self.assertFalse(selection["overfit"]["detected"])
        self.assertEqual(selection["candidate_steps"], list(STEPS))
        self.assertEqual(selection["selected_step"], 500_000)

    def test_joint_tie_prefers_test_then_earlier_step(self) -> None:
        record = _record(
            [0.80, 0.70, 0.70, 0.60, 0.60],
            [0.20, 0.30, 0.30, 0.25, 0.30],
        )

        selection = select_best_checkpoint([record, record, record])

        self.assertFalse(selection["overfit"]["detected"])
        self.assertEqual(selection["selected_step"], 200_000)

    def test_budget_extension_accepts_train_gain_without_test_regression(self) -> None:
        records = [
            _record(
                [0.20, 0.25, 0.30, 0.32, 0.35],
                [0.20, 0.22, 0.25, 0.248, 0.245],
            ),
            _record(
                [0.21, 0.26, 0.31, 0.33, 0.36],
                [0.21, 0.23, 0.26, 0.258, 0.255],
            ),
            _record(
                [0.19, 0.24, 0.29, 0.31, 0.34],
                [0.19, 0.21, 0.24, 0.238, 0.235],
            ),
        ]

        assessment = select_best_checkpoint(records)["budget_extension"]

        self.assertTrue(assessment["recommended"])
        self.assertTrue(assessment["train_trigger"])
        self.assertFalse(assessment["test_trigger"])
        self.assertAlmostEqual(assessment["train_delta"], 0.05)
        self.assertAlmostEqual(assessment["test_delta"], -0.005)
        self.assertEqual(assessment["window_steps"], list(STEPS[-3:]))

    def test_budget_extension_can_be_marked_not_applicable(self) -> None:
        record = _record(
            [0.20, 0.25, 0.30, 0.35, 0.40],
            [0.20, 0.25, 0.30, 0.35, 0.40],
        )

        assessment = select_best_checkpoint(
            [record, record, record], budget_extension_eligible=False
        )["budget_extension"]

        self.assertFalse(assessment["recommended"])
        self.assertEqual(assessment["status"], "not_applicable")

    def test_budget_extension_recommendations_are_paired_and_deduplicated(self) -> None:
        selections = {
            config_id: {
                "status": "complete",
                "budget_extension": {
                    "recommended": config_id
                    in {"p4-w256", "w2048-d4", "r256-w256"}
                },
            }
            for config_id in CONFIGS
        }

        recommendations = _group_budget_extension_recommendations(selections)

        self.assertEqual(
            recommendations["triggered_configs"],
            ["p4-w256", "w2048-d4", "r256-w256"],
        )
        self.assertEqual(
            [record["configs"] for record in recommendations["pairs"]],
            [
                ["p4-w256", "w512-d4"],
                ["w1024-d4", "w2048-d4"],
                ["r128-w256", "r256-w256"],
            ],
        )
        self.assertEqual(
            recommendations["config_union"],
            [
                "p4-w256",
                "w512-d4",
                "w1024-d4",
                "w2048-d4",
                "r128-w256",
                "r256-w256",
            ],
        )

    def test_requires_exactly_three_complete_seeds(self) -> None:
        record = _record([0.1] * 5, [0.1] * 5)

        with self.assertRaisesRegex(
            AssertionError, "requires 3 complete seeds"
        ):
            select_best_checkpoint([record, record])

        invalid = dict(record)
        invalid["status"] = "invalid"
        with self.assertRaisesRegex(
            AssertionError, "requires complete audited runs"
        ):
            select_best_checkpoint([record, record, invalid])

    def test_rejects_mismatched_checkpoint_steps(self) -> None:
        records = [
            _record([0.1] * 5, [0.1] * 5),
            _record([0.1] * 5, [0.1] * 5),
            _record([0.1] * 5, [0.1] * 5),
        ]
        records[-1]["checkpoints"][-1]["step"] = 600_000

        with self.assertRaisesRegex(AssertionError, "steps differ"):
            select_best_checkpoint(records)


if __name__ == "__main__":
    unittest.main()
