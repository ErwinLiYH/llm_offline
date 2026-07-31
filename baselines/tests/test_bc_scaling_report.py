from __future__ import annotations

import unittest

from baselines.experiments.bc_scaling_20260723.audit_results import (
    select_best_checkpoint,
)
from baselines.experiments.bc_scaling_20260723.render_report import (
    TWO_MILLION_STEPS,
    _append_axis_early_stop_table,
    _append_completion_table,
    _append_conditional_range_table,
    _append_budget_extension_recommendations,
    _append_macro_table,
    _append_individual_seed_table,
    _append_next_depth_gate_table,
    _append_next_width_gate_table,
    _append_pareto_table,
    _append_resource_table,
    _append_scaling_comparison_table,
    _append_selection_table,
    _append_width_recipe_table,
)


def _record(seed: int) -> dict:
    return {
        "status": "complete",
        "experiment_id": f"synthetic-extension-s{seed}",
        "model": {"trainable_parameter_count": 1234},
        "resources": {
            "peak_gpu_allocated_bytes": (500 + seed) * 1024 ** 2,
            "peak_process_rss_bytes": (10 + seed) * 1024 ** 3,
            "diagnostic_wall_time_seconds": (3600 + seed * 360),
            "effective_updates_per_second": 50.0 + seed,
            "processed_examples": 512_000_000,
            "wall_time_scope": "through_final_epoch_callback_before_final_rollout",
        },
        "checkpoints": [
            {
                "step": step,
                "checkpoint": f"/synthetic/s{seed}/step_{step}.d3",
                "trainer_state": (
                    f"/synthetic/s{seed}/trainer_state_step_{step}.pt"
                ),
                "train": {"success_macro": 0.5},
                "test": {"success_macro": 0.25},
            }
            for step in TWO_MILLION_STEPS
        ],
    }


class ScalingReportTest(unittest.TestCase):
    def test_completion_table_can_hide_inactive_conditional_cells(self) -> None:
        lines: list[str] = []
        runs = {
            "pointmaze": {
                "r256-w256": [
                    {"status": "complete"},
                    {"status": "complete"},
                    {"status": "complete"},
                ],
                "r512-w256": [
                    {"status": "missing"},
                    {"status": "missing"},
                    {"status": "missing"},
                ],
            }
        }

        _append_completion_table(
            lines,
            "Active formal matrix",
            runs,
            included_configs={"pointmaze": ["r256-w256"]},
        )
        report = "\n".join(lines)

        self.assertIn("`r256-w256`", report)
        self.assertNotIn("`r512-w256`", report)

    def test_budget_extension_recommendation_table_renders_pairs(self) -> None:
        lines: list[str] = []
        recommendations = {
            "pointmaze_500k_to_1m": {
                "triggered_configs": ["p4-w256"],
                "pairs": [
                    {
                        "axis": "width",
                        "trigger": "p4-w256",
                        "configs": ["p4-w256", "w512-d4"],
                    }
                ],
                "config_union": ["p4-w256", "w512-d4"],
            },
            "antmaze_1m_to_2m": {
                "triggered_configs": [],
                "pairs": [],
                "config_union": [],
            },
        }

        _append_budget_extension_recommendations(lines, recommendations)
        report = "\n".join(lines)

        self.assertIn("PointMaze 500k → 1M", report)
        self.assertIn("width: `p4-w256`/`w512-d4`", report)
        self.assertIn("| AntMaze 1M → 2M | 无 | 无 | 无 |", report)

    def test_2m_extension_tables_include_endpoint_and_selection(self) -> None:
        records = [_record(seed) for seed in range(3)]
        runs = {"antmaze": {"w2048-d4": records}}
        selections = {
            "antmaze": {
                "w2048-d4": select_best_checkpoint(records),
            }
        }
        lines: list[str] = []

        _append_completion_table(lines, "2M completion", runs)
        _append_macro_table(lines, "2M macros", runs, TWO_MILLION_STEPS)
        _append_selection_table(lines, "2M selection", selections)
        report = "\n".join(lines)

        self.assertIn("| antmaze | `w2048-d4` | 3/3 |", report)
        self.assertIn("| Family / 配置 | 100k", report)
        self.assertIn("| 2000k | 参数量 |", report)
        self.assertIn("50.00% ± 0.00 pp / 25.00% ± 0.00 pp", report)
        self.assertIn("| antmaze / `w2048-d4` | 100k |", report)
        self.assertIn("Endpoint train / test", report)
        self.assertIn("Gap selected / endpoint", report)
        self.assertIn("Trainer state", report)
        self.assertIn("| 3/3 / 3/3 |", report)

    def test_selection_table_exposes_non_resumable_checkpoints(self) -> None:
        records = [_record(seed) for seed in range(3)]
        for record in records:
            for checkpoint in record["checkpoints"]:
                checkpoint["trainer_state"] = None
        selections = {
            "pointmaze": {
                "r64-w256": select_best_checkpoint(records),
            }
        }
        lines: list[str] = []

        _append_selection_table(lines, "500k selection", selections)
        report = "\n".join(lines)

        self.assertIn("| 0/3 / 0/3 |", report)
        self.assertIn("预算延长必须从头训练", report)

    def test_resource_table_renders_measured_three_seed_costs(self) -> None:
        records = [_record(seed) for seed in range(3)]
        lines: list[str] = []

        _append_resource_table(
            lines,
            "2M training resources",
            {"antmaze": {"w2048-d4": records}},
        )
        report = "\n".join(lines)

        self.assertIn("GPU peak", report)
        self.assertIn("502.0 MiB", report)
        self.assertIn("12.00 GiB", report)
        self.assertIn("1.10 ± 0.10 h", report)
        self.assertIn("51.00 ± 1.00", report)
        self.assertIn("512,000,000", report)

    def test_next_depth_gate_table_renders_frozen_plateau_rule(self) -> None:
        lines: list[str] = []
        gates = {
            "pointmaze": {
                "r512-w256": {
                    "status": "not_eligible",
                    "selected_test_delta": 0.04,
                    "positive_seed_count": 3,
                    "tail_train_delta": 0.02,
                    "tail_test_delta": 0.03,
                    "failed_gates": ["common_budget_plateau"],
                }
            }
        }

        _append_next_depth_gate_table(lines, gates)
        report = "\n".join(lines)

        self.assertIn("train 变化绝对值不超过 3 pp", report)
        self.assertIn("test 不超过 2 pp", report)
        self.assertIn(
            "| pointmaze | `r512-w256` | not_eligible | +4.00 pp | 3/3 | "
            "+2.00 pp / +3.00 pp | common_budget_plateau |",
            report,
        )
        self.assertIn(
            "| antmaze | `r1024-w256` | incomplete |", report
        )

    def test_w5120_gate_table_renders_two_point_gain_rule(self) -> None:
        lines: list[str] = []
        gates = {
            "pointmaze": {
                "w5120-d4": {
                    "status": "eligible_for_50k_range",
                    "evidence_stage": "1m_common_budget",
                    "selected_test_delta": 0.025,
                    "positive_seed_count": 2,
                    "tail_train_delta": 0.01,
                    "tail_test_delta": -0.005,
                    "failed_gates": [],
                }
            }
        }

        _append_next_width_gate_table(lines, gates)
        report = "\n".join(lines)

        self.assertIn("selected test 提升至少 2 pp", report)
        self.assertIn(
            "| pointmaze/w5120-d4 | `1m_common_budget` | "
            "eligible_for_50k_range | "
            "+2.50 pp | 2/3 | "
            "+1.00 pp / -0.50 pp | 无 |",
            report,
        )
        self.assertIn(
            "| antmaze/w5120-d4 | — | incomplete |", report
        )

    def test_axis_early_stop_table_renders_two_adjacent_deltas(self) -> None:
        lines: list[str] = []
        _append_axis_early_stop_table(
            lines,
            {
                "pointmaze": {
                    "depth": {
                        "r512-w256": {
                            "status": "stop",
                            "compared_configs": [
                                "r64-w256",
                                "r128-w256",
                                "r256-w256",
                            ],
                            "latest_two_deltas": [0.015, -0.01],
                        }
                    }
                }
            },
        )
        report = "\n".join(lines)

        self.assertIn("连续无增益早停", report)
        self.assertIn(
            "`r64-w256` → `r128-w256` → `r256-w256`",
            report,
        )
        self.assertIn("+1.50 pp / -1.00 pp", report)

    def test_core_width_rescue_table_keeps_pilot_and_range_evidence(self) -> None:
        lines: list[str] = []
        _append_width_recipe_table(
            lines,
            {
                "pointmaze": {
                    "w3072-d4": {
                        "status": "in_progress",
                        "range_evidence": [
                            {
                                "candidate": "pilot-lr3e4",
                                "experiment_id": "pilot-s2",
                                "status": "collapsed",
                            },
                            {
                                "candidate": "lr1e4",
                                "experiment_id": "range-s2",
                                "status": "in_progress",
                            },
                        ],
                    }
                }
            },
        )
        report = "\n".join(lines)

        self.assertIn("核心宽网络学习率 rescue 证据", report)
        self.assertIn("pilot-lr3e4 `pilot-s2`: collapsed", report)
        self.assertIn("lr1e4 `range-s2`: in_progress", report)
        self.assertIn("重新训练全部三个正式 seed", report)

    def test_conditional_range_table_renders_pending_candidate(self) -> None:
        lines: list[str] = []
        _append_conditional_range_table(
            lines,
            {
                "pointmaze": {
                    "r512-w256": {
                        "axis": "depth",
                        "status": "missing",
                        "next_required_candidate": "lr3e4-retry",
                        "range_evidence": [],
                    }
                }
            },
        )
        report = "\n".join(lines)

        self.assertIn("条件结构 50k range", report)
        self.assertIn("lr3e4-retry", report)

    def test_pareto_table_marks_cost_dominated_configurations(self) -> None:
        cheap = [_record(seed) for seed in range(3)]
        expensive = [_record(seed) for seed in range(3)]
        for record in cheap:
            record["resources"]["diagnostic_wall_time_seconds"] = 3600
            record["resources"]["peak_gpu_allocated_bytes"] = 400 * 1024 ** 2
        for record in expensive:
            record["resources"]["diagnostic_wall_time_seconds"] = 7200
            record["resources"]["peak_gpu_allocated_bytes"] = 800 * 1024 ** 2
        cheap_selection = select_best_checkpoint(cheap)
        expensive_selection = select_best_checkpoint(expensive)
        lines: list[str] = []

        _append_pareto_table(
            lines,
            "Synthetic Pareto",
            {"pointmaze": {"cheap": cheap, "expensive": expensive}},
            {
                "pointmaze": {
                    "cheap": cheap_selection,
                    "expensive": expensive_selection,
                }
            },
        )
        report = "\n".join(lines)

        self.assertIn(
            "| pointmaze / `cheap` | 25.00% | 20,259.32 | "
            "1.00 h | 400.0 MiB | 是 | 是 |",
            report,
        )
        self.assertIn(
            "| pointmaze / `expensive` | 25.00% | 20,259.32 | 2.00 h | "
            "800.0 MiB | 否 | 否 |",
            report,
        )

    def test_scaling_comparison_table_renders_bootstrap_interval(self) -> None:
        lines: list[str] = []
        comparisons = {
            "width": [
                {
                    "lower_config": "w1024-d4",
                    "upper_config": "w2048-d4",
                    "lower_step": 500_000,
                    "upper_step": 300_000,
                    "observed_test_delta": 0.0111,
                    "confidence_interval": [-0.02, 0.04],
                    "confidence_interval_contains_zero": True,
                    "individual_seed_deltas": [0.03, -0.01, 0.02],
                    "positive_variant_count": 4,
                    "negative_variant_count": 2,
                    "zero_variant_count": 0,
                }
            ],
            "depth": [],
        }

        _append_scaling_comparison_table(
            lines, "Synthetic comparisons", comparisons
        )
        report = "\n".join(lines)

        self.assertIn("hierarchical bootstrap", report)
        self.assertIn("`w1024-d4` → `w2048-d4`", report)
        self.assertIn("+1.11 pp", report)
        self.assertIn("[-2.00 pp, +4.00 pp]，含 0", report)
        self.assertIn("+3.00 pp, -1.00 pp, +2.00 pp", report)
        self.assertIn("| 4/2/0 |", report)

    def test_individual_seed_table_uses_shared_selected_step(self) -> None:
        records = [_record(seed) for seed in range(3)]
        selection = select_best_checkpoint(records)
        lines: list[str] = []

        _append_individual_seed_table(
            lines,
            "Synthetic seed results",
            {"pointmaze": {"w2048-d4": selection}},
        )
        report = "\n".join(lines)

        self.assertIn(
            "| pointmaze / `w2048-d4` | selected | "
            "100k | 0 | 50.00% | 25.00% | 37.50% | 25.00% |",
            report,
        )
        self.assertIn(
            "| pointmaze / `w2048-d4` | endpoint | "
            "2000k | 2 | 50.00% | 25.00% | 37.50% | 25.00% |",
            report,
        )


if __name__ == "__main__":
    unittest.main()
