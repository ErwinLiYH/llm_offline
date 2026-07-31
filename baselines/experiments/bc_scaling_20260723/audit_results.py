"""审计 BC scaling run 是否遵守固定的多环境训练/测试协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = ROOT / "baseline_runs"
OUTPUT = ROOT / "reports/bc_scaling_20260723_checkpoint_audit.json"
FORMAL_STEPS = (100_000, 200_000, 300_000, 400_000, 500_000)
MILLION_STEPS = tuple(range(100_000, 1_000_001, 100_000))
TWO_MILLION_STEPS = tuple(range(100_000, 2_000_001, 100_000))
EPISODES_PER_VARIANT = 30
ROLLOUT_BATCH_SIZE = 30
FORMAL_SEEDS = (0, 1, 2)
OVERFIT_REQUIRED_DECLINING_INTERVALS = 2
OVERFIT_MIN_TOTAL_TEST_DECLINE = 0.02
OVERFIT_RECOVERY_TOLERANCE = 0.01
BUDGET_WINDOW_CHECKPOINTS = 3
BUDGET_MIN_TEST_GAIN = 0.02
BUDGET_MIN_TRAIN_GAIN = 0.03
BUDGET_MAX_TEST_DROP_FOR_TRAIN_TRIGGER = 0.01
BOOTSTRAP_REPLICATES = 10_000
MIN_FORMAL_TAIL_ACTION_STD = 0.05
MIN_FORMAL_POSITIVE_GRADIENT_FRACTION = 0.80
DEPTH_LEARNING_RATE_LADDER = (3e-4, 1e-4, 3e-5, 1e-5)
DEPTH_LEARNING_RATE_LABELS = {
    3e-4: "lr3e4",
    1e-4: "lr1e4",
    3e-5: "lr3e5",
    1e-5: "lr1e5",
    3e-6: "lr3e6",
}
DEPTH_SLOW_START_EXTENSIONS = {
    ("antmaze", "r256-w256"): {
        "learning_rate": 1e-5,
        "source_label": "lr1e5",
        "extension_label": "lr1e5-extend100k",
        # Added post hoc after the 1e-5 continuation still had an exactly
        # saturated eighth action dimension through 90k.  It follows the
        # existing roughly threefold LR ladder and retains the full 50k gate.
        "fallback_learning_rate": 3e-6,
        "fallback_label": "lr3e6",
    },
}
CORE_WIDTH_RESCUE_LADDER = (1e-4, 3e-5, 1e-5)
CORE_WIDTH_RESCUES = {
    ("pointmaze", "w3072-d4"): {
        "pilot_learning_rate": 3e-4,
        "pilot_seed": 2,
    },
}
CORE_WIDTH_FOLLOWUPS = {
    ("pointmaze", "w4096-d4"): ("pointmaze", "w3072-d4"),
}
NEXT_DEPTH_MIN_TEST_GAIN = 0.03
NEXT_WIDTH_MIN_TEST_GAIN = 0.02
AXIS_EARLY_STOP_MIN_GAIN = 0.02
NEXT_DEPTH_REQUIRED_POSITIVE_SEEDS = 2
DEPTH_PLATEAU_MAX_ABS_TEST_DELTA = 0.02
DEPTH_PLATEAU_MAX_ABS_TRAIN_DELTA = 0.03
DEPTH_PARAMETER_PRECHECK = {
    "pointmaze": {
        "r128-w256": 8_549_378,
        "r512-w256": 34_010_114,
        "r1024-w256": 67_957_762,
    },
    "antmaze": {
        "r128-w256": 8_548_872,
        "r512-w256": 34_009_608,
        "r1024-w256": 67_957_256,
    },
}
WIDTH_PARAMETER_PRECHECK = {
    "pointmaze": {
        "w5120-d4": 106_168_322,
        "w6144-d4": 152_567_810,
        "w7168-d4": 207_355_906,
        "w8192-d4": 270_532_610,
    },
    "antmaze": {
        "w5120-d4": 106_158_088,
        "w6144-d4": 152_555_528,
        "w7168-d4": 207_341_576,
        "w8192-d4": 270_516_232,
    },
}
OBSERVATION = {
    "include_map": True,
    "include_location_sensing": True,
    "include_wall_sensing": True,
    "wall_sensing_version": "v3",
    "map_sensing_boundary_risk_threshold": 0.10,
}
TRAINING_DIAGNOSTICS = {
    "enabled": True,
    "gradient_sample_interval_updates": 1000,
    "action_probe_size": 1024,
}
FAMILIES = {
    "pointmaze": {
        "train": [
            "open", "umaze", "medium", "large",
            *[f"local-layoutV2-{index:02d}" for index in range(1, 13)],
        ],
        "test": [f"test-layoutV2-{index:02d}" for index in range(1, 7)],
        "eval_start_goal_mode": "random-start-goal",
    },
    "antmaze": {
        "train": [
            "umaze", "medium-diverse", "large-diverse", "ultra",
            *[f"local-layout-{index:02d}" for index in range(1, 13)],
        ],
        "test": [f"test-layout-{index:02d}" for index in range(1, 5)],
        "eval_start_goal_mode": "fix-start-goal",
    },
}
CONFIGS = {
    "legacy": {
        "architecture": "legacy_mlp",
        "hidden_units": [1024, 1024, 1024],
        "batch_size": 512,
    },
    "p4-w256": {"architecture": "plain_mlp", "width": 256, "body_depth": 4},
    "p4-w256-b512": {
        "architecture": "plain_mlp",
        "width": 256,
        "body_depth": 4,
        "batch_size": 512,
    },
    "p4-w256-b1024": {
        "architecture": "plain_mlp",
        "width": 256,
        "body_depth": 4,
        "batch_size": 1024,
    },
    "r4-w256": {"architecture": "residual_mlp", "width": 256, "body_depth": 4},
    "r4-w256-b512": {
        "architecture": "residual_mlp",
        "width": 256,
        "body_depth": 4,
        "batch_size": 512,
    },
    "r4-w256-b1024": {
        "architecture": "residual_mlp",
        "width": 256,
        "body_depth": 4,
        "batch_size": 1024,
    },
    "w512-d4": {"architecture": "plain_mlp", "width": 512, "body_depth": 4},
    "w1024-d4": {"architecture": "plain_mlp", "width": 1024, "body_depth": 4},
    "w2048-d4": {"architecture": "plain_mlp", "width": 2048, "body_depth": 4},
    "w3072-d4": {
        "architecture": "plain_mlp",
        "width": 3072,
        "body_depth": 4,
        # Point seed 2 在论文起始值 3e-4 的首个 25k 诊断中动作 std=0；
        # 正式比较改由 rescue ladder 的首个稳定学习率统一重跑三 seed。
        "learning_rate_candidates_by_family": {
            "pointmaze": CORE_WIDTH_RESCUE_LADDER,
        },
    },
    "w4096-d4": {
        "architecture": "plain_mlp",
        "width": 4096,
        "body_depth": 4,
        "learning_rate_candidates_by_family": {
            "pointmaze": CORE_WIDTH_RESCUE_LADDER,
        },
    },
    "w5120-d4": {
        "architecture": "plain_mlp",
        "width": 5120,
        "body_depth": 4,
        "learning_rate_candidates_by_family": {
            "pointmaze": DEPTH_LEARNING_RATE_LADDER,
            "antmaze": DEPTH_LEARNING_RATE_LADDER,
        },
    },
    "w6144-d4": {
        "architecture": "plain_mlp",
        "width": 6144,
        "body_depth": 4,
        "learning_rate_candidates_by_family": {
            "pointmaze": DEPTH_LEARNING_RATE_LADDER,
            "antmaze": DEPTH_LEARNING_RATE_LADDER,
        },
    },
    "w7168-d4": {
        "architecture": "plain_mlp",
        "width": 7168,
        "body_depth": 4,
        "learning_rate_candidates_by_family": {
            "pointmaze": DEPTH_LEARNING_RATE_LADDER,
            "antmaze": DEPTH_LEARNING_RATE_LADDER,
        },
    },
    "w8192-d4": {
        "architecture": "plain_mlp",
        "width": 8192,
        "body_depth": 4,
        "learning_rate_candidates_by_family": {
            "pointmaze": DEPTH_LEARNING_RATE_LADDER,
            "antmaze": DEPTH_LEARNING_RATE_LADDER,
        },
    },
    "r16-w256": {"architecture": "residual_mlp", "width": 256, "body_depth": 16},
    "r64-w256": {
        "architecture": "residual_mlp",
        "width": 256,
        "body_depth": 64,
        # 稳定学习率按 family 独立冻结。Point 的 3e-4 动作塌缩，50k
        # 诊断确认 1e-4 正常；Ant 仍须从论文起始值 3e-4 独立筛选。
        "learning_rate_by_family": {"pointmaze": 1e-4},
        "learning_rate_candidates_by_family": {
            "antmaze": DEPTH_LEARNING_RATE_LADDER
        },
    },
    "r128-w256": {
        "architecture": "residual_mlp",
        "width": 256,
        "body_depth": 128,
        "learning_rate_candidates_by_family": {
            "pointmaze": DEPTH_LEARNING_RATE_LADDER,
            "antmaze": DEPTH_LEARNING_RATE_LADDER,
        },
    },
    "r256-w256": {
        "architecture": "residual_mlp",
        "width": 256,
        "body_depth": 256,
        # Point 的 3e-4/1e-4/3e-5 均塌缩；1e-5 完整 50k range
        # loss、action distribution、gradient 和 rollout 均正常。
        "learning_rate_by_family": {"pointmaze": 1e-5},
        "learning_rate_candidates_by_family": {
            "antmaze": (*DEPTH_LEARNING_RATE_LADDER, 3e-6)
        },
    },
    "r512-w256": {
        "architecture": "residual_mlp",
        "width": 256,
        "body_depth": 512,
        "learning_rate_candidates_by_family": {
            "pointmaze": DEPTH_LEARNING_RATE_LADDER,
            "antmaze": DEPTH_LEARNING_RATE_LADDER,
        },
    },
    "r1024-w256": {
        "architecture": "residual_mlp",
        "width": 256,
        "body_depth": 1024,
        "learning_rate_candidates_by_family": {
            "pointmaze": DEPTH_LEARNING_RATE_LADDER,
            "antmaze": DEPTH_LEARNING_RATE_LADDER,
        },
    },
}

# 此处是冻结的正式比较矩阵，而不是可被审计器枚举到的所有网络。每个深度先用
# 3e-4 range；若出现动作塌缩，可只调整学习率，选第一个已验证稳定的学习率冻结
# 为该深度的正式配置。Point R64 因此使用 1e-4，Point R256 使用完整 50k
# range 已验证稳定的 1e-5；新增的 R128 独立做 range，避免 R64 直接跳到 R256。
# 三者都属于必须完成的 Point 主深度网格。
# R512/R1024 是逐级条件扩展点，不在门槛触发前加入 required matrix。
# AntMaze 500k 表先保留已完成的四点；正式 1M width/depth 网格另按 budget run 审计。
# 其余行仍写入审计 JSON，便于看到它们未运行，但不会让 --strict 把一个已按计划
# 完成的实验误报为缺失。
REQUIRED_FORMAL_CONFIGS = {
    "pointmaze": (
        "legacy",
        "p4-w256",
        "r4-w256",
        "w512-d4",
        "w1024-d4",
        "w2048-d4",
        "w3072-d4",
        "w4096-d4",
        "r16-w256",
        "r64-w256",
        "r128-w256",
        "r256-w256",
    ),
    "antmaze": ("p4-w256", "w1024-d4", "w2048-d4", "r16-w256"),
}
CONDITIONAL_FORMAL_CONFIGS = {
    "pointmaze": (
        "w5120-d4",
        "w6144-d4",
        "w7168-d4",
        "w8192-d4",
        "r512-w256",
        "r1024-w256",
    ),
    # AntMaze uses 1M as its common budget. Conditional Ant cells are therefore
    # discovered through the budget matrix rather than the 500k legacy table.
    "antmaze": (),
}

# Family-level common-budget curves. Point's completed R4 1M run is retained as
# the existing budget control. Ant's full width/depth scaling uses 1M as its
# primary budget; these are run from scratch under `budget/u1m.yaml`, so every
# setting has the same ten checkpoint opportunities and a clean 100k--1M curve.
REQUIRED_BUDGET_CONFIGS = {
    "pointmaze": ("r4-w256",),
    "antmaze": (
        "p4-w256",
        "w512-d4",
        "w1024-d4",
        "w2048-d4",
        "w3072-d4",
        "w4096-d4",
        "r4-w256",
        "r16-w256",
        "r64-w256",
        "r128-w256",
        "r256-w256",
    ),
}
BUDGET_EXTENSION_ELIGIBLE_CONFIGS = {
    "p4-w256",
    "w512-d4",
    "w1024-d4",
    "w2048-d4",
    "w3072-d4",
    "w4096-d4",
    "w5120-d4",
    "w6144-d4",
    "w7168-d4",
    "w8192-d4",
    "r4-w256",
    "r16-w256",
    "r64-w256",
    "r128-w256",
    "r256-w256",
    "r512-w256",
    "r1024-w256",
}
WIDTH_CONFIG_ORDER = (
    "p4-w256",
    "w512-d4",
    "w1024-d4",
    "w2048-d4",
    "w3072-d4",
    "w4096-d4",
    "w5120-d4",
    "w6144-d4",
    "w7168-d4",
    "w8192-d4",
)
DEPTH_CONFIG_ORDER = (
    "r4-w256",
    "r16-w256",
    "r64-w256",
    "r128-w256",
    "r256-w256",
    "r512-w256",
    "r1024-w256",
)

# 等样本数控制与 500k 正式矩阵分开保存：前者只按计划先跑一个 seed，
# 其终点分别为 250k / 125k，但仍使用同一数据、观测和 rollout 协议。
CONTROL_CONFIGS = {
    "r4-w256-b512-u250k": {
        "network": {
            "architecture": "residual_mlp",
            "width": 256,
            "body_depth": 4,
            "batch_size": 512,
        },
        "updates": 250_000,
        "checkpoint_steps": (100_000, 200_000, 250_000),
    },
    "r4-w256-b1024-u125k": {
        "network": {
            "architecture": "residual_mlp",
            "width": 256,
            "body_depth": 4,
            "batch_size": 1024,
        },
        "updates": 125_000,
        "checkpoint_steps": (100_000, 125_000),
    },
    "p4-w256-b512-u250k": {
        "network": {
            "architecture": "plain_mlp",
            "width": 256,
            "body_depth": 4,
            "batch_size": 512,
        },
        "updates": 250_000,
        "checkpoint_steps": (100_000, 200_000, 250_000),
    },
    "p4-w256-b1024-u125k": {
        "network": {
            "architecture": "plain_mlp",
            "width": 256,
            "body_depth": 4,
            "batch_size": 1024,
        },
        "updates": 125_000,
        "checkpoint_steps": (100_000, 125_000),
    },
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_prefix(
    family: str,
    config_id: str,
    seed: int,
    *,
    updates: int = 500_000,
    run_kind: str = "formal",
) -> str:
    """Return the stable run-id part, excluding its calendar suffix."""
    if run_kind == "formal" and updates == 500_000:
        # 已完成正式矩阵使用最初约定的 `-500k-<date>` 命名；保持兼容，
        # 以免审计脚本自身的演进让旧结果被误判为缺失。
        return f"bc-scaling-formal-{family}-{config_id}-s{seed}-500k-"
    if run_kind == "budget" and updates == 1_000_000:
        return f"bc-scaling-budget-{family}-{config_id}-s{seed}-u1m-"
    if run_kind == "extension" and updates == 2_000_000:
        return f"bc-scaling-extension-{family}-{config_id}-s{seed}-u2m-"
    return (
        f"bc-scaling-{run_kind}-{family}-{config_id}-s{seed}-"
        f"u{updates // 1000}k-"
    )


def _discover_run_config_ids(
    family: str, *, updates: int, run_kind: str
) -> tuple[str, ...]:
    """Treat the first conditional run as activation of its 3-seed matrix."""

    discovered = []
    for config_id in CONFIGS:
        if any(
            any(
                path.is_dir()
                for path in RUNS_ROOT.glob(
                    _run_prefix(
                        family,
                        config_id,
                        seed,
                        updates=updates,
                        run_kind=run_kind,
                    )
                    + "*"
                )
            )
            for seed in FORMAL_SEEDS
        ):
            discovered.append(config_id)
    return tuple(discovered)


def _discover_extension_config_ids(family: str) -> tuple[str, ...]:
    """Treat the first 2M run for a cell as activation of its 3-seed matrix."""

    return _discover_run_config_ids(
        family, updates=2_000_000, run_kind="extension"
    )


def _activated_extension_config_ids(
    family: str, budget_selections: dict
) -> tuple[str, ...]:
    recommended = ()
    if family == "antmaze":
        recommended = tuple(
            _group_budget_extension_recommendations(
                budget_selections
            )["config_union"]
        )
    return tuple(
        dict.fromkeys(
            (*recommended, *_discover_extension_config_ids(family))
        )
    )


def _activated_formal_config_ids(family: str) -> tuple[str, ...]:
    """Add a conditional Point cell to strict audit after its first run starts."""

    discovered = set(
        _discover_run_config_ids(
            family, updates=500_000, run_kind="formal"
        )
    )
    activated_conditionals = (
        config_id
        for config_id in CONDITIONAL_FORMAL_CONFIGS[family]
        if config_id in discovered
    )
    return tuple(
        dict.fromkeys(
            (*REQUIRED_FORMAL_CONFIGS[family], *activated_conditionals)
        )
    )


def _activated_budget_config_ids(
    family: str, formal_selections: dict
) -> tuple[str, ...]:
    """Activate a recommended Point extension before its first directory exists."""

    recommended = ()
    if family == "pointmaze":
        recommended = tuple(
            _group_budget_extension_recommendations(
                formal_selections
            )["config_union"]
        )
    discovered = _discover_run_config_ids(
        family, updates=1_000_000, run_kind="budget"
    )
    return tuple(
        dict.fromkeys(
            (
                *REQUIRED_BUDGET_CONFIGS[family],
                *recommended,
                *discovered,
            )
        )
    )


def _resolve_run_dir(
    family: str,
    config_id: str,
    seed: int,
    *,
    updates: int = 500_000,
    run_kind: str = "formal",
    required_learning_rate: float | None = None,
) -> Path | None:
    """Find exactly one completed-attempt directory for an audited cell.

    The experiment directory is date-stamped, so a long grid can cross midnight.
    The date is not experimental semantics; requiring one exact directory prefix
    prevents such a run from being falsely reported as missing.
    """
    prefix = _run_prefix(
        family, config_id, seed, updates=updates, run_kind=run_kind
    )
    matches = sorted(path for path in RUNS_ROOT.glob(f"{prefix}*") if path.is_dir())
    if required_learning_rate is not None:
        matching_recipe = []
        for path in matches:
            config_path = path / "config.yaml"
            if not config_path.is_file():
                continue
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if (
                config.get("algorithm_config", {}).get("learning_rate")
                == required_learning_rate
            ):
                matching_recipe.append(path)
        matches = matching_recipe
    if not matches:
        return None
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches)
        raise AssertionError(
            f"ambiguous formal run directories for {prefix!r}: {names}"
        )
    return matches[0]


def _group_metrics(rollout: dict, variants: list[str]) -> dict:
    per_variant = [rollout["variants"][variant] for variant in variants]
    return {
        "num_variants": len(variants),
        "num_episodes": sum(record["num_episodes"] for record in per_variant),
        "success_macro": mean(record["success_rate"] for record in per_variant),
        "success_micro": sum(
            record["successful_episode_count"] for record in per_variant
        ) / sum(record["num_episodes"] for record in per_variant),
        "per_variant_success": {
            variant: rollout["variants"][variant]["success_rate"]
            for variant in variants
        },
        "per_variant_episode_success": {
            variant: [
                bool(episode["success"])
                for episode in rollout["variants"][variant]["episodes"]
            ]
            for variant in variants
        },
    }


def _summary(values: list[float]) -> dict:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "individual_seeds": values,
    }


def expected_learning_rate(expected_network: dict, family: str) -> float:
    """Resolve a depth recipe without leaking a rescue across env families."""

    return expected_network.get("learning_rate_by_family", {}).get(
        family, expected_network.get("learning_rate", 3e-4)
    )


def allowed_learning_rates(expected_network: dict, family: str) -> tuple[float, ...]:
    """Return the frozen LR, or the preregistered family-specific range ladder."""

    candidates = expected_network.get(
        "learning_rate_candidates_by_family", {}
    ).get(family)
    if candidates is None:
        return (expected_learning_rate(expected_network, family),)
    return tuple(candidates)


def _load_unique_range_assessment(
    pattern: str, *, required: bool, allow_in_progress: bool = False
) -> dict | None:
    matches = sorted(path for path in RUNS_ROOT.glob(pattern) if path.is_dir())
    if not matches:
        if required:
            raise AssertionError(f"missing required range run matching {pattern!r}")
        return None
    if len(matches) != 1:
        raise AssertionError(
            f"ambiguous range runs matching {pattern!r}: "
            + ", ".join(path.name for path in matches)
        )
    assessment_path = matches[0] / "range_assessment.json"
    if not assessment_path.is_file():
        if allow_in_progress:
            return {
                "experiment_id": matches[0].name,
                "status": "in_progress",
                "metrics": {},
                "reasons": [],
            }
        raise AssertionError(
            f"{matches[0].name}: missing range_assessment.json"
        )
    assessment = _load_json(assessment_path)
    status = assessment.get("status")
    if status not in {"stable", "unstable", "invalid"}:
        raise AssertionError(
            f"{matches[0].name}: invalid range assessment status {status!r}"
        )
    return {
        "experiment_id": matches[0].name,
        "status": status,
        "metrics": assessment.get("metrics", {}),
        "reasons": assessment.get("reasons", []),
    }


def _audit_depth_range_ladder(family: str, config_id: str) -> dict:
    """Audit the preregistered retry/LR ladder without requiring formal runs."""

    evidence = []
    initial = _load_unique_range_assessment(
        f"bc-scaling-range-{family}-{config_id}-s0-lr3e4-50k-*",
        required=False,
    )
    if initial is None:
        return {
            "status": "missing",
            "next_required_candidate": "lr3e4",
            "range_evidence": evidence,
        }
    evidence.append(initial)
    if initial["status"] == "stable":
        return {
            "status": "stable",
            "selected_learning_rate": 3e-4,
            "range_evidence": evidence,
        }

    retry = _load_unique_range_assessment(
        f"bc-scaling-range-{family}-{config_id}-s0-lr3e4-retry-50k-*",
        required=False,
    )
    if retry is None:
        return {
            "status": "missing",
            "next_required_candidate": "lr3e4-retry",
            "range_evidence": evidence,
        }
    evidence.append(retry)
    if retry["status"] == "stable":
        return {
            "status": "stable",
            "selected_learning_rate": 3e-4,
            "range_evidence": evidence,
        }
    if retry["status"] == "invalid":
        return {
            "status": "invalid_infrastructure_evidence",
            "next_required_candidate": "lr3e4-retry2",
            "range_evidence": evidence,
        }

    for candidate_lr in DEPTH_LEARNING_RATE_LADDER[1:]:
        label = DEPTH_LEARNING_RATE_LABELS[candidate_lr]
        candidate = _load_unique_range_assessment(
            f"bc-scaling-range-{family}-{config_id}-s0-{label}-50k-*",
            required=False,
        )
        if candidate is None:
            return {
                "status": "missing",
                "next_required_candidate": label,
                "range_evidence": evidence,
            }
        evidence.append(candidate)
        if candidate["status"] == "stable":
            return {
                "status": "stable",
                "selected_learning_rate": candidate_lr,
                "range_evidence": evidence,
            }
        if candidate["status"] == "invalid":
            return {
                "status": "invalid_infrastructure_evidence",
                "next_required_candidate": f"{label}-retry",
                "range_evidence": evidence,
            }
    extension_spec = DEPTH_SLOW_START_EXTENSIONS.get((family, config_id))
    if extension_spec is not None:
        source = evidence[-1]
        eligibility_path = (
            RUNS_ROOT
            / source["experiment_id"]
            / "slow_start_extension_assessment.json"
        )
        if not eligibility_path.is_file():
            return {
                "status": "slow_start_gate_missing",
                "selected_learning_rate": None,
                "next_required_candidate": "assess-slow-start-extension",
                "range_evidence": evidence,
            }
        eligibility = _load_json(eligibility_path)
        eligibility_status = eligibility.get("status")
        if eligibility_status not in {"eligible", "ineligible"}:
            raise AssertionError(
                f"{source['experiment_id']}: invalid slow-start extension "
                f"status {eligibility_status!r}"
            )
        source["slow_start_extension"] = {
            "status": eligibility_status,
            "metrics": eligibility.get("metrics", {}),
            "reasons": eligibility.get("reasons", []),
        }
        if eligibility_status == "eligible":
            extension_label = extension_spec["extension_label"]
            extension = _load_unique_range_assessment(
                f"bc-scaling-range-{family}-{config_id}-s0-"
                f"{extension_label}-*",
                required=False,
                allow_in_progress=True,
            )
            if extension is None:
                return {
                    "status": "slow_start_extension_required",
                    "selected_learning_rate": None,
                    "next_required_candidate": extension_label,
                    "range_evidence": evidence,
                }
            evidence.append(extension)
            if extension["status"] == "stable":
                return {
                    "status": "stable",
                    "selected_learning_rate": extension_spec["learning_rate"],
                    "range_evidence": evidence,
                }
            if extension["status"] == "in_progress":
                return {
                    "status": "in_progress",
                    "selected_learning_rate": None,
                    "next_required_candidate": extension_label,
                    "range_evidence": evidence,
                }
            if extension["status"] == "invalid":
                return {
                    "status": "invalid_infrastructure_evidence",
                    "selected_learning_rate": None,
                    "next_required_candidate": f"{extension_label}-retry",
                    "range_evidence": evidence,
                }
            fallback_lr = extension_spec.get("fallback_learning_rate")
            fallback_label = extension_spec.get("fallback_label")
            if fallback_lr is not None and fallback_label is not None:
                fallback = _load_unique_range_assessment(
                    f"bc-scaling-range-{family}-{config_id}-s0-"
                    f"{fallback_label}-50k-*",
                    required=False,
                    allow_in_progress=True,
                )
                if fallback is None:
                    return {
                        "status": "posthoc_lr_rescue_required",
                        "selected_learning_rate": None,
                        "next_required_candidate": fallback_label,
                        "range_evidence": evidence,
                    }
                evidence.append(fallback)
                if fallback["status"] == "stable":
                    return {
                        "status": "stable",
                        "selected_learning_rate": fallback_lr,
                        "range_evidence": evidence,
                    }
                if fallback["status"] == "in_progress":
                    return {
                        "status": "in_progress",
                        "selected_learning_rate": None,
                        "next_required_candidate": fallback_label,
                        "range_evidence": evidence,
                    }
                if fallback["status"] == "invalid":
                    return {
                        "status": "invalid_infrastructure_evidence",
                        "selected_learning_rate": None,
                        "next_required_candidate": f"{fallback_label}-retry",
                        "range_evidence": evidence,
                    }
    return {
        "status": "exhausted_without_stable_recipe",
        "selected_learning_rate": None,
        "range_evidence": evidence,
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _audit_core_width_pilot_collapse(family: str, config_id: str) -> dict:
    """Prove the observed formal-pilot collapse that activates width rescue."""

    rescue = CORE_WIDTH_RESCUES[(family, config_id)]
    seed = rescue["pilot_seed"]
    learning_rate = rescue["pilot_learning_rate"]
    prefix = _run_prefix(family, config_id, seed)
    matches = []
    for path in sorted(RUNS_ROOT.glob(f"{prefix}*")):
        config_path = path / "config.yaml"
        if not path.is_dir() or not config_path.is_file():
            continue
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config["algorithm_config"]["learning_rate"] == learning_rate:
            matches.append(path)
    if len(matches) != 1:
        return {
            "status": "missing" if not matches else "invalid",
            "experiment_ids": [path.name for path in matches],
            "required_learning_rate": learning_rate,
            "pilot_seed": seed,
        }

    diagnostics = _read_jsonl(matches[0] / "training_diagnostics.jsonl")
    if not diagnostics:
        return {
            "status": "in_progress",
            "experiment_id": matches[0].name,
            "required_learning_rate": learning_rate,
            "pilot_seed": seed,
        }
    latest = diagnostics[-1]
    action_std = latest.get("action_std", [])
    collapsed = (
        latest.get("step", 0) >= 25_000
        and isinstance(action_std, list)
        and action_std
        and min(action_std) < MIN_FORMAL_TAIL_ACTION_STD
    )
    return {
        "status": "collapsed" if collapsed else "not_collapsed",
        "experiment_id": matches[0].name,
        "required_learning_rate": learning_rate,
        "pilot_seed": seed,
        "observed_step": latest.get("step"),
        "action_std": action_std,
        "gradient_global_norm_mean": latest.get("gradient_global_norm_mean"),
    }


def _audit_core_width_rescue_ladder(family: str, config_id: str) -> dict:
    """Select the first stable lower LR after an audited core-width collapse."""

    pilot = _audit_core_width_pilot_collapse(family, config_id)
    evidence = [{"candidate": "pilot-lr3e4", **pilot}]
    if pilot["status"] != "collapsed":
        return {
            "status": "pilot_collapse_not_proven",
            "range_evidence": evidence,
        }

    seed = CORE_WIDTH_RESCUES[(family, config_id)]["pilot_seed"]
    for candidate_lr in CORE_WIDTH_RESCUE_LADDER:
        label = DEPTH_LEARNING_RATE_LABELS[candidate_lr]
        assessment = _load_unique_range_assessment(
            f"bc-scaling-range-{family}-{config_id}-s{seed}-{label}-50k-*",
            required=False,
            allow_in_progress=True,
        )
        if assessment is None:
            return {
                "status": "missing",
                "next_required_candidate": label,
                "range_evidence": evidence,
            }
        evidence.append({"candidate": label, **assessment})
        if assessment["status"] == "in_progress":
            return {
                "status": "in_progress",
                "next_required_candidate": label,
                "range_evidence": evidence,
            }
        if assessment["status"] == "stable":
            return {
                "status": "stable",
                "selected_learning_rate": candidate_lr,
                "range_evidence": evidence,
            }
        if assessment["status"] == "invalid":
            return {
                "status": "invalid_infrastructure_evidence",
                "next_required_candidate": f"{label}-retry",
                "range_evidence": evidence,
            }
    return {
        "status": "exhausted_without_stable_recipe",
        "selected_learning_rate": None,
        "range_evidence": evidence,
    }


def _audit_followup_width_ladder(
    family: str,
    config_id: str,
    prerequisite_recipe: dict,
) -> dict:
    """Start a wider setting from the preceding width's approved recipe."""

    if prerequisite_recipe.get("status") != "valid":
        return {
            "status": "prerequisite_incomplete",
            "prerequisite": (
                f"{CORE_WIDTH_FOLLOWUPS[(family, config_id)][0]}/"
                f"{CORE_WIDTH_FOLLOWUPS[(family, config_id)][1]}"
            ),
            "range_evidence": [],
        }
    starting_lr = prerequisite_recipe["selected_learning_rate"]
    candidates = tuple(
        learning_rate
        for learning_rate in CORE_WIDTH_RESCUE_LADDER
        if learning_rate <= starting_lr
    )
    evidence = []
    for candidate_lr in candidates:
        label = DEPTH_LEARNING_RATE_LABELS[candidate_lr]
        assessment = _load_unique_range_assessment(
            f"bc-scaling-range-{family}-{config_id}-s2-{label}-50k-*",
            required=False,
            allow_in_progress=True,
        )
        if assessment is None:
            return {
                "status": "missing",
                "starting_learning_rate": starting_lr,
                "next_required_candidate": label,
                "range_evidence": evidence,
            }
        evidence.append({"candidate": label, **assessment})
        if assessment["status"] == "in_progress":
            return {
                "status": "in_progress",
                "starting_learning_rate": starting_lr,
                "next_required_candidate": label,
                "range_evidence": evidence,
            }
        if assessment["status"] == "stable":
            return {
                "status": "stable",
                "starting_learning_rate": starting_lr,
                "selected_learning_rate": candidate_lr,
                "range_evidence": evidence,
            }
        if assessment["status"] == "invalid":
            return {
                "status": "invalid_infrastructure_evidence",
                "starting_learning_rate": starting_lr,
                "next_required_candidate": f"{label}-retry",
                "range_evidence": evidence,
            }
    return {
        "status": "exhausted_without_stable_recipe",
        "starting_learning_rate": starting_lr,
        "selected_learning_rate": None,
        "range_evidence": evidence,
    }


def _audit_core_width_recipe(
    family: str,
    config_id: str,
    records: list[dict],
    range_audit: dict,
    selection_rule: str | None = None,
) -> dict:
    if range_audit["status"] != "stable":
        return {
            "status": range_audit["status"],
            "range_evidence": range_audit["range_evidence"],
            "next_required_candidate": range_audit.get(
                "next_required_candidate"
            ),
        }
    formal_recipe = _validate_conditional_formal_recipe(range_audit, records)
    return {
        "status": formal_recipe["status"],
        "selected_learning_rate": range_audit["selected_learning_rate"],
        "shared_across_formal_seeds": (
            formal_recipe.get("shared_across_formal_seeds", False)
        ),
        "complete_seed_count": formal_recipe.get("complete_seed_count"),
        "required_seed_count": formal_recipe.get("required_seed_count"),
        "error": formal_recipe.get("error"),
        "selection_rule": selection_rule
        or (
            "after an audited 3e-4 formal-pilot collapse, freeze the first "
            "stable LR in 1e-4 -> 3e-5 -> 1e-5 and rerun all three seeds"
        ),
        "range_evidence": range_audit["range_evidence"],
    }


def _audit_width_range(
    family: str,
    config_id: str,
    *,
    starting_learning_rate: float = 3e-4,
) -> dict:
    """Continue a wider network from its predecessor's approved LR recipe."""

    candidates = tuple(
        learning_rate
        for learning_rate in DEPTH_LEARNING_RATE_LADDER
        if learning_rate <= starting_learning_rate
    )
    evidence = []
    for candidate_lr in candidates:
        label = DEPTH_LEARNING_RATE_LABELS[candidate_lr]
        assessment = _load_unique_range_assessment(
            f"bc-scaling-range-{family}-{config_id}-s0-{label}-50k-*",
            required=False,
            allow_in_progress=True,
        )
        if assessment is None:
            return {
                "status": "missing",
                "starting_learning_rate": starting_learning_rate,
                "next_required_candidate": label,
                "range_evidence": evidence,
            }
        evidence.append({"candidate": label, **assessment})
        if assessment["status"] == "in_progress":
            return {
                "status": "in_progress",
                "starting_learning_rate": starting_learning_rate,
                "next_required_candidate": label,
                "range_evidence": evidence,
            }
        if assessment["status"] == "stable":
            return {
                "status": "stable",
                "starting_learning_rate": starting_learning_rate,
                "selected_learning_rate": candidate_lr,
                "range_evidence": evidence,
            }
        if assessment["status"] == "invalid":
            return {
                "status": "invalid_infrastructure_evidence",
                "starting_learning_rate": starting_learning_rate,
                "next_required_candidate": f"{label}-retry",
                "range_evidence": evidence,
            }
    return {
        "status": "exhausted_without_stable_recipe",
        "starting_learning_rate": starting_learning_rate,
        "selected_learning_rate": None,
        "range_evidence": evidence,
    }


def _audit_conditional_range_requirement(
    *,
    family: str,
    config_id: str,
    axis: str,
    gate: dict,
    early_stop_gate: dict | None = None,
) -> dict:
    if gate.get("status") == "stopped_by_user_decision":
        return {
            "status": "stopped_by_user_decision",
            "family": family,
            "config_id": config_id,
            "axis": axis,
            "gate_status": gate["status"],
            "protocol_gate_status": gate.get("protocol_gate_status"),
            "decision_reason": gate.get("decision_reason"),
        }
    if (
        early_stop_gate is not None
        and early_stop_gate.get("status") == "stop"
    ):
        return {
            "status": "stopped_by_axis_early_stop",
            "family": family,
            "config_id": config_id,
            "axis": axis,
            "gate_status": gate.get("status", "missing"),
            "early_stop_gate": early_stop_gate,
        }
    if gate.get("status") != "eligible_for_50k_range":
        return {
            "status": "not_required",
            "family": family,
            "config_id": config_id,
            "axis": axis,
            "gate_status": gate.get("status", "missing"),
        }
    try:
        result = (
            _audit_depth_range_ladder(family, config_id)
            if axis == "depth"
            else _audit_width_range(
                family,
                config_id,
                starting_learning_rate=gate.get(
                    "current_learning_rate"
                )
                or 3e-4,
            )
        )
        return {
            "family": family,
            "config_id": config_id,
            "axis": axis,
            "gate_status": gate["status"],
            **result,
        }
    except Exception as exc:
        return {
            "status": "invalid",
            "family": family,
            "config_id": config_id,
            "axis": axis,
            "gate_status": gate["status"],
            "error": str(exc),
        }


def _validate_conditional_formal_recipe(
    requirement: dict, records: list[dict]
) -> dict:
    """Tie a range-approved conditional cell to one shared formal learning rate."""

    if requirement.get("status") != "stable":
        return {"status": "not_applicable"}
    complete = [record for record in records if record["status"] == "complete"]
    if len(complete) != len(FORMAL_SEEDS):
        return {
            "status": "awaiting_formal_matrix",
            "complete_seed_count": len(complete),
            "required_seed_count": len(FORMAL_SEEDS),
        }
    selected_lr = requirement["selected_learning_rate"]
    learning_rates = {record["learning_rate"] for record in complete}
    if learning_rates != {selected_lr}:
        return {
            "status": "invalid",
            "selected_learning_rate": selected_lr,
            "formal_learning_rates": sorted(learning_rates),
            "error": (
                "formal seeds must all use the first stable range learning rate"
            ),
        }
    return {
        "status": "valid",
        "selected_learning_rate": selected_lr,
        "shared_across_formal_seeds": True,
    }


def _conditional_prerequisite_failures(
    *,
    family: str,
    config_id: str,
    requirements: dict,
) -> list[str]:
    """Require every earlier conditional depth to have a valid formal recipe."""

    prerequisite_keys = []
    width_prerequisites = {
        "w6144-d4": "w5120-d4",
        "w7168-d4": "w6144-d4",
        "w8192-d4": "w7168-d4",
    }
    if config_id in width_prerequisites:
        prerequisite_keys.append(
            (family, width_prerequisites[config_id])
        )
    if config_id == "r1024-w256":
        prerequisite_keys.append((family, "r512-w256"))
    if family == "antmaze" and config_id in {
        "r512-w256",
        "r1024-w256",
    }:
        prerequisite_keys.append(("pointmaze", config_id))
    failures = []
    for prerequisite_family, prerequisite_config in prerequisite_keys:
        record = requirements.get(prerequisite_family, {}).get(
            prerequisite_config, {}
        )
        if (
            record.get("status") != "stable"
            or record.get("formal_recipe", {}).get("status") != "valid"
        ):
            failures.append(
                f"{prerequisite_family}/{prerequisite_config}"
            )
    return failures


def _audit_depth_recipe(
    family: str, config_id: str, records: list[dict]
) -> dict:
    """Prove that formal seeds share the first stable LR in the frozen ladder."""

    range_audit = _audit_depth_range_ladder(family, config_id)
    complete = [record for record in records if record["status"] == "complete"]
    if len(complete) != len(FORMAL_SEEDS):
        return {
            "status": "incomplete",
            "complete_seed_count": len(complete),
            "required_seed_count": len(FORMAL_SEEDS),
            "range_status": range_audit["status"],
            "selected_learning_rate": range_audit.get(
                "selected_learning_rate"
            ),
            "next_required_candidate": range_audit.get(
                "next_required_candidate"
            ),
            "range_evidence": range_audit.get("range_evidence", []),
        }
    if range_audit["status"] != "stable":
        raise AssertionError(
            f"{family}/{config_id}: range ladder is "
            f"{range_audit['status']}"
        )
    formal_recipe = _validate_conditional_formal_recipe(
        range_audit, records
    )
    if formal_recipe["status"] != "valid":
        raise AssertionError(
            f"{family}/{config_id}: {formal_recipe.get('error', 'invalid recipe')}"
        )
    return {
        "status": "valid",
        "selected_learning_rate": range_audit["selected_learning_rate"],
        "shared_across_formal_seeds": True,
        "selection_rule": (
            "initial 3e-4; rerun once if unstable; then first stable in "
            "1e-4 -> 3e-5 -> 1e-5; Ant D256 may use the explicitly "
            "post-hoc 100k slow-start gate and then lr=3e-6"
        ),
        "range_evidence": range_audit["range_evidence"],
    }


def _audit_ant_depth_recipe(config_id: str, records: list[dict]) -> dict:
    """Compatibility wrapper for the preregistered Ant depth recipe audit."""

    if config_id not in {"r64-w256", "r128-w256", "r256-w256"}:
        return {"status": "not_applicable"}
    recipe = _audit_depth_recipe("antmaze", config_id, records)
    if config_id != "r256-w256":
        return recipe

    retry = _load_unique_range_assessment(
        "bc-scaling-range-antmaze-r256-w256-s2-"
        "lr3e6-formal-retry-50k-*",
        required=False,
    )
    if retry is None:
        return recipe
    if retry["status"] != "unstable":
        raise AssertionError(
            "Ant D256 seed2 same-recipe retry must remain unstable before "
            "declaring a cross-seed stability failure"
        )
    retry_tail_std = retry.get("metrics", {}).get("tail_min_action_std")
    if retry_tail_std is None or retry_tail_std >= MIN_FORMAL_TAIL_ACTION_STD:
        raise AssertionError(
            "Ant D256 seed2 retry does not prove constant-action collapse"
        )

    failed_runs = sorted(
        path
        for path in RUNS_ROOT.glob(
            "bc-scaling-budget-antmaze-r256-w256-s2-u1m-*"
        )
        if path.is_dir()
    )
    if len(failed_runs) != 1:
        raise AssertionError(
            "expected exactly one failed Ant D256 seed2 formal attempt"
        )
    failed_run = failed_runs[0]
    failed_diagnostics = _read_jsonl(
        failed_run / "training_diagnostics.jsonl"
    )
    if not failed_diagnostics or failed_diagnostics[-1].get("step") != 500_000:
        raise AssertionError(
            f"{failed_run.name}: formal collapse evidence must reach 500k"
        )
    failed_tail_std = min(
        value
        for row in failed_diagnostics[-3:]
        for value in row["action_std"]
    )
    if failed_tail_std >= MIN_FORMAL_TAIL_ACTION_STD:
        raise AssertionError(
            f"{failed_run.name}: formal tail does not prove action collapse"
        )

    return {
        "status": "confirmed_cross_seed_instability",
        "selected_learning_rate": recipe.get("selected_learning_rate"),
        "range_status": recipe.get("range_status"),
        "range_evidence": [
            *recipe.get("range_evidence", []),
            retry,
        ],
        "failed_formal_run_id": failed_run.name,
        "failed_formal_seed": 2,
        "failed_formal_observed_step": 500_000,
        "failed_formal_tail_min_action_std": failed_tail_std,
        "same_recipe_retry_run_id": retry["experiment_id"],
        "same_recipe_retry_tail_min_action_std": retry_tail_std,
        "resolution": (
            "stop D256 after the lowest approved 3e-6 recipe collapses in "
            "formal seed2 and repeats under the same seed/configuration"
        ),
    }


def _assess_budget_extension(
    metrics: list[dict],
    *,
    sustained_overfit_detected: bool,
    eligible: bool,
) -> dict:
    """Apply the frozen common-budget tail criterion to one 3-seed curve."""

    if not eligible:
        return {
            "recommended": False,
            "status": "not_applicable",
            "window_steps": [],
        }
    if len(metrics) < BUDGET_WINDOW_CHECKPOINTS:
        return {
            "recommended": False,
            "status": "insufficient_checkpoints",
            "window_steps": [row["step"] for row in metrics],
        }

    window = metrics[-BUDGET_WINDOW_CHECKPOINTS:]
    first = window[0]
    endpoint = window[-1]
    train_delta = endpoint["train"]["mean"] - first["train"]["mean"]
    test_delta = endpoint["test"]["mean"] - first["test"]["mean"]
    train_positive_seed_count = sum(
        endpoint_value > first_value
        for first_value, endpoint_value in zip(
            first["train"]["individual_seeds"],
            endpoint["train"]["individual_seeds"],
            strict=True,
        )
    )
    test_positive_seed_count = sum(
        endpoint_value > first_value
        for first_value, endpoint_value in zip(
            first["test"]["individual_seeds"],
            endpoint["test"]["individual_seeds"],
            strict=True,
        )
    )

    # With three equally spaced checkpoint steps, the sign of the least-squares
    # slope is the sign of endpoint - first; the middle point affects residuals
    # but cancels from the fitted slope numerator.
    test_trigger = (
        test_delta >= BUDGET_MIN_TEST_GAIN
        and test_positive_seed_count >= 2
    )
    train_trigger = (
        train_delta >= BUDGET_MIN_TRAIN_GAIN
        and test_delta >= -BUDGET_MAX_TEST_DROP_FOR_TRAIN_TRIGGER
        and train_positive_seed_count >= 2
    )
    recommended = (
        not sustained_overfit_detected and (test_trigger or train_trigger)
    )
    return {
        "recommended": recommended,
        "status": "complete",
        "window_steps": [row["step"] for row in window],
        "train_delta": train_delta,
        "test_delta": test_delta,
        "train_positive_seed_count": train_positive_seed_count,
        "test_positive_seed_count": test_positive_seed_count,
        "test_trigger": test_trigger,
        "train_trigger": train_trigger,
        "blocked_by_sustained_overfit": sustained_overfit_detected,
        "pairing_rule": (
            "extend_with_smaller_neighbor_or_larger_neighbor_at_lower_boundary"
        ),
    }


def _group_budget_extension_recommendations(selections: dict) -> dict:
    """Pair every triggered scaling cell with its nearest registered neighbor."""

    triggered = {
        config_id
        for config_id, selection in selections.items()
        if selection.get("status") == "complete"
        and selection.get("budget_extension", {}).get("recommended") is True
    }
    pairs = []
    config_union = []
    for axis_name, order in (
        ("width", WIDTH_CONFIG_ORDER),
        ("depth", DEPTH_CONFIG_ORDER),
    ):
        for index, config_id in enumerate(order):
            if config_id not in triggered:
                continue
            neighbor_index = index - 1 if index > 0 else index + 1
            pair = [order[neighbor_index], config_id]
            pair.sort(key=order.index)
            record = {
                "axis": axis_name,
                "trigger": config_id,
                "configs": pair,
            }
            if record not in pairs:
                pairs.append(record)
            for paired_config in pair:
                if paired_config not in config_union:
                    config_union.append(paired_config)
    return {
        "triggered_configs": [
            config_id
            for order in (WIDTH_CONFIG_ORDER, DEPTH_CONFIG_ORDER)
            for config_id in order
            if config_id in triggered
        ],
        "pairs": pairs,
        "config_union": config_union,
    }


def assess_axis_early_stop(
    *,
    axis: str,
    config_order: tuple[str, ...],
    next_config: str,
    selections: dict,
) -> dict:
    """Stop an axis after two consecutive selected-test gains below 2 pp."""

    next_index = config_order.index(next_config)
    if next_index < 3:
        raise ValueError(
            "two-extension early stopping requires three preceding settings"
        )
    compared_configs = list(config_order[next_index - 3 : next_index])
    compared_selections = [
        selections.get(config_id, {}) for config_id in compared_configs
    ]
    if any(
        selection.get("status") != "complete"
        for selection in compared_selections
    ):
        return {
            "status": "incomplete",
            "axis": axis,
            "next_config": next_config,
            "compared_configs": compared_configs,
            "minimum_meaningful_gain": AXIS_EARLY_STOP_MIN_GAIN,
            "decision": "wait_for_three_preceding_settings",
        }
    selected_test = [
        selection["selected"]["test"]["mean"]
        for selection in compared_selections
    ]
    deltas = [
        upper - lower
        for lower, upper in zip(
            selected_test[:-1], selected_test[1:], strict=True
        )
    ]
    stop = all(
        delta < AXIS_EARLY_STOP_MIN_GAIN
        and not math.isclose(
            delta,
            AXIS_EARLY_STOP_MIN_GAIN,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for delta in deltas
    )
    return {
        "status": "stop" if stop else "continue",
        "axis": axis,
        "next_config": next_config,
        "compared_configs": compared_configs,
        "selected_test": selected_test,
        "latest_two_deltas": deltas,
        "minimum_meaningful_gain": AXIS_EARLY_STOP_MIN_GAIN,
        "decision": (
            "stop_axis_before_next_config"
            if stop
            else "continue_axis_to_next_config"
        ),
    }


def _axis_early_stop_gates(selections: dict) -> dict:
    return {
        "width": {
            next_config: assess_axis_early_stop(
                axis="width",
                config_order=WIDTH_CONFIG_ORDER,
                next_config=next_config,
                selections=selections,
            )
            for next_config in (
                "w5120-d4",
                "w6144-d4",
                "w7168-d4",
                "w8192-d4",
            )
        },
        "depth": {
            next_config: assess_axis_early_stop(
                axis="depth",
                config_order=DEPTH_CONFIG_ORDER,
                next_config=next_config,
                selections=selections,
            )
            for next_config in ("r512-w256", "r1024-w256")
        },
    }


def select_best_checkpoint(
    records: list[dict], *, budget_extension_eligible: bool = True
) -> dict:
    """Select one shared checkpoint step from three completed formal seeds.

    Test-map rollout is intentionally part of this experiment's checkpoint
    selection contract. To avoid per-seed peak picking, all seeds use the same
    step selected from the cross-seed mean curve. A sustained overfit suffix is
    excluded before maximizing equal-weight train/test success.
    """
    if len(records) != len(FORMAL_SEEDS):
        raise AssertionError(
            f"checkpoint selection requires {len(FORMAL_SEEDS)} complete seeds"
        )
    if any(record.get("status") != "complete" for record in records):
        raise AssertionError("checkpoint selection requires complete audited runs")

    steps = [row["step"] for row in records[0]["checkpoints"]]
    if not steps:
        raise AssertionError("checkpoint selection requires at least one checkpoint")
    for record in records[1:]:
        record_steps = [row["step"] for row in record["checkpoints"]]
        if record_steps != steps:
            raise AssertionError("checkpoint steps differ across formal seeds")

    metrics = []
    for checkpoint_index, step in enumerate(steps):
        train_values = [
            record["checkpoints"][checkpoint_index]["train"]["success_macro"]
            for record in records
        ]
        test_values = [
            record["checkpoints"][checkpoint_index]["test"]["success_macro"]
            for record in records
        ]
        joint_values = [
            0.5 * (train_value + test_value)
            for train_value, test_value in zip(train_values, test_values, strict=True)
        ]
        generalization_gap_values = [
            train_value - test_value
            for train_value, test_value in zip(
                train_values, test_values, strict=True
            )
        ]
        metrics.append(
            {
                "step": step,
                "train": _summary(train_values),
                "test": _summary(test_values),
                "joint": _summary(joint_values),
                "generalization_gap": _summary(
                    generalization_gap_values
                ),
            }
        )

    overfit_peak_index = None
    endpoint = metrics[-1]
    for peak_index in range(
        0, len(metrics) - OVERFIT_REQUIRED_DECLINING_INTERVALS
    ):
        peak_test = metrics[peak_index]["test"]["mean"]
        if peak_index > 0 and peak_test < metrics[peak_index - 1]["test"]["mean"]:
            continue
        decline_window = metrics[
            peak_index : peak_index + OVERFIT_REQUIRED_DECLINING_INTERVALS + 1
        ]
        if any(
            later["test"]["mean"] >= earlier["test"]["mean"]
            for earlier, later in zip(
                decline_window[:-1], decline_window[1:], strict=True
            )
        ):
            continue
        if (
            peak_test - decline_window[-1]["test"]["mean"]
            < OVERFIT_MIN_TOTAL_TEST_DECLINE
        ):
            continue
        if max(row["test"]["mean"] for row in metrics[peak_index + 1 :]) >= (
            peak_test - OVERFIT_RECOVERY_TOLERANCE
        ):
            continue
        if endpoint["train"]["mean"] < metrics[peak_index]["train"]["mean"]:
            continue
        peak_seed_tests = metrics[peak_index]["test"]["individual_seeds"]
        endpoint_seed_tests = endpoint["test"]["individual_seeds"]
        declining_seed_count = sum(
            endpoint_value < peak_value
            for peak_value, endpoint_value in zip(
                peak_seed_tests, endpoint_seed_tests, strict=True
            )
        )
        if declining_seed_count < 2:
            continue
        overfit_peak_index = peak_index
        break

    if overfit_peak_index is None:
        candidate_metrics = metrics
    else:
        candidate_metrics = metrics[: overfit_peak_index + 1]

    selected = max(
        candidate_metrics,
        key=lambda row: (
            row["joint"]["mean"],
            row["test"]["mean"],
            -row["step"],
        ),
    )

    def checkpoint_artifacts(step: int) -> list[dict]:
        artifacts = []
        for seed, record in zip(FORMAL_SEEDS, records, strict=True):
            checkpoint_row = next(
                row for row in record["checkpoints"] if row["step"] == step
            )
            artifacts.append(
                {
                    "seed": seed,
                    "experiment_id": record.get("experiment_id"),
                    "checkpoint": checkpoint_row.get("checkpoint"),
                    "trainer_state": checkpoint_row.get("trainer_state"),
                }
            )
        return artifacts

    overfit = {
        "detected": overfit_peak_index is not None,
        "peak_step": (
            metrics[overfit_peak_index]["step"]
            if overfit_peak_index is not None
            else None
        ),
        "decline_start_step": (
            metrics[overfit_peak_index + 1]["step"]
            if overfit_peak_index is not None
            else None
        ),
    }
    if overfit_peak_index is not None:
        overfit["test_decline_peak_to_endpoint"] = (
            endpoint["test"]["mean"]
            - metrics[overfit_peak_index]["test"]["mean"]
        )

    return {
        "status": "complete",
        "selection_rule": "max_equal_weight_train_test_before_sustained_overfit",
        "checkpoint_metrics": metrics,
        "candidate_steps": [row["step"] for row in candidate_metrics],
        "selected_step": selected["step"],
        "selected": selected,
        "selected_checkpoints": checkpoint_artifacts(selected["step"]),
        "endpoint_step": endpoint["step"],
        "endpoint": endpoint,
        "endpoint_checkpoints": checkpoint_artifacts(endpoint["step"]),
        "overfit": overfit,
        "selected_to_endpoint": {
            metric_name: endpoint[metric_name]["mean"] - selected[metric_name]["mean"]
            for metric_name in (
                "train",
                "test",
                "joint",
                "generalization_gap",
            )
        },
        "budget_extension": _assess_budget_extension(
            metrics,
            sustained_overfit_detected=overfit["detected"],
            eligible=budget_extension_eligible,
        ),
    }


def assess_next_scaling_range_gate(
    *,
    axis: str,
    family: str,
    previous_config: str,
    current_config: str,
    next_config: str,
    runs: dict,
    selections: dict,
    minimum_test_gain: float | None,
    next_parameter_count: int,
    point_prerequisite_selection: dict | None = None,
    require_plateau: bool = True,
) -> dict:
    """Decide whether a conditional larger network may enter its 50k range.

    This does not approve a formal three-seed run. The next setting must still
    pass its own LR-stability and measured-resource range assessment.
    """

    previous = selections.get(previous_config, {})
    current = selections.get(current_config, {})
    if (
        previous.get("status") != "complete"
        or current.get("status") != "complete"
    ):
        return {
            "status": "incomplete",
            "axis": axis,
            "previous_config": previous_config,
            "current_config": current_config,
            "next_config": next_config,
        }

    previous_selected = previous["selected"]
    current_selected = current["selected"]
    selected_test_delta = (
        current_selected["test"]["mean"]
        - previous_selected["test"]["mean"]
    )
    individual_seed_deltas = [
        current_value - previous_value
        for previous_value, current_value in zip(
            previous_selected["test"]["individual_seeds"],
            current_selected["test"]["individual_seeds"],
            strict=True,
        )
    ]
    positive_seed_count = sum(delta > 0 for delta in individual_seed_deltas)
    performance_gate = (
        True
        if minimum_test_gain is None
        else (
            selected_test_delta >= minimum_test_gain
            and positive_seed_count >= NEXT_DEPTH_REQUIRED_POSITIVE_SEEDS
        )
    )

    tail = current["checkpoint_metrics"][-BUDGET_WINDOW_CHECKPOINTS:]
    tail_train_delta = tail[-1]["train"]["mean"] - tail[0]["train"]["mean"]
    tail_test_delta = tail[-1]["test"]["mean"] - tail[0]["test"]["mean"]
    sustained_overfit = current["overfit"]["detected"]
    plateau_gate = (
        not sustained_overfit
        and (
            not require_plateau
            or (
                abs(tail_test_delta) <= DEPTH_PLATEAU_MAX_ABS_TEST_DELTA
                and abs(tail_train_delta)
                <= DEPTH_PLATEAU_MAX_ABS_TRAIN_DELTA
            )
        )
    )

    complete_runs = [
        record
        for record in runs.get(current_config, [])
        if record.get("status") == "complete"
    ]
    stability_gate = len(complete_runs) == len(FORMAL_SEEDS)
    current_learning_rates = {
        record.get("learning_rate") for record in complete_runs
    }
    current_learning_rate = (
        current_learning_rates.pop()
        if len(current_learning_rates) == 1
        else None
    )
    resource_evidence_gate = stability_gate and all(
        "resources" in record
        and record.get("model", {}).get("trainable_parameter_count", 0) > 0
        for record in complete_runs
    )
    resource_projection = None
    if resource_evidence_gate:
        current_parameter_counts = {
            record["model"]["trainable_parameter_count"]
            for record in complete_runs
        }
        if len(current_parameter_counts) != 1:
            raise AssertionError(
                f"{family}/{current_config}: inconsistent parameter counts"
            )
        current_parameter_count = current_parameter_counts.pop()
        parameter_ratio = next_parameter_count / current_parameter_count
        resource_projection = {
            "current_parameter_count": current_parameter_count,
            "next_parameter_count": next_parameter_count,
            "parameter_ratio": parameter_ratio,
            "linear_projected_peak_gpu_allocated_bytes": (
                max(
                    record["resources"]["peak_gpu_allocated_bytes"]
                    for record in complete_runs
                )
                * parameter_ratio
            ),
            "linear_projected_diagnostic_wall_time_seconds": (
                mean(
                    record["resources"]["diagnostic_wall_time_seconds"]
                    for record in complete_runs
                )
                * parameter_ratio
            ),
            "projection_role": (
                "pre-range planning only; the next 50k range must measure "
                "actual GPU/RSS/wall-clock"
            ),
        }

    cross_family_gate = (
        point_prerequisite_selection is None
        or point_prerequisite_selection.get("status") == "complete"
    )
    eligible = all(
        (
            performance_gate,
            plateau_gate,
            stability_gate,
            resource_evidence_gate,
            cross_family_gate,
        )
    )
    failed_gates = [
        gate_name
        for gate_name, passed in (
            ("selected_test_gain", performance_gate),
            ("common_budget_plateau", plateau_gate),
            ("formal_stability", stability_gate),
            ("resource_evidence", resource_evidence_gate),
            ("point_corresponding_depth", cross_family_gate),
        )
        if not passed
    ]
    return {
        "status": (
            "eligible_for_50k_range" if eligible else "not_eligible"
        ),
        "axis": axis,
        "previous_config": previous_config,
        "current_config": current_config,
        "next_config": next_config,
        "selected_test_delta": selected_test_delta,
        "minimum_selected_test_delta": minimum_test_gain,
        "current_gain_required": minimum_test_gain is not None,
        "individual_seed_test_deltas": individual_seed_deltas,
        "positive_seed_count": positive_seed_count,
        "required_positive_seed_count": NEXT_DEPTH_REQUIRED_POSITIVE_SEEDS,
        "performance_gate": performance_gate,
        "tail_steps": [row["step"] for row in tail],
        "tail_train_delta": tail_train_delta,
        "tail_test_delta": tail_test_delta,
        "maximum_abs_plateau_train_delta": (
            DEPTH_PLATEAU_MAX_ABS_TRAIN_DELTA
        ),
        "maximum_abs_plateau_test_delta": (
            DEPTH_PLATEAU_MAX_ABS_TEST_DELTA
        ),
        "sustained_overfit": sustained_overfit,
        "plateau_gate": plateau_gate,
        "plateau_required": require_plateau,
        "stability_gate": stability_gate,
        "resource_evidence_gate": resource_evidence_gate,
        "resource_projection": resource_projection,
        "current_learning_rate": current_learning_rate,
        "cross_family_gate": cross_family_gate,
        "failed_gates": failed_gates,
        "next_action": (
            f"run_next_{axis}_50k_range"
            if eligible
            else f"do_not_start_next_{axis}"
        ),
    }


def assess_next_depth_range_gate(
    *,
    family: str,
    previous_config: str,
    current_config: str,
    next_config: str,
    runs: dict,
    selections: dict,
    point_prerequisite_selection: dict | None = None,
) -> dict:
    return assess_next_scaling_range_gate(
        axis="depth",
        family=family,
        previous_config=previous_config,
        current_config=current_config,
        next_config=next_config,
        runs=runs,
        selections=selections,
        minimum_test_gain=NEXT_DEPTH_MIN_TEST_GAIN,
        next_parameter_count=DEPTH_PARAMETER_PRECHECK[family][next_config],
        point_prerequisite_selection=point_prerequisite_selection,
    )


def assess_next_width_range_gate(
    *,
    family: str,
    previous_config: str,
    current_config: str,
    next_config: str,
    runs: dict,
    selections: dict,
    continue_until_two_low_gains: bool = False,
) -> dict:
    return assess_next_scaling_range_gate(
        axis="width",
        family=family,
        previous_config=previous_config,
        current_config=current_config,
        next_config=next_config,
        runs=runs,
        selections=selections,
        minimum_test_gain=(
            None
            if continue_until_two_low_gains
            else NEXT_WIDTH_MIN_TEST_GAIN
        ),
        next_parameter_count=WIDTH_PARAMETER_PRECHECK[family][next_config],
        require_plateau=not continue_until_two_low_gains,
    )


def _select_width_gate_evidence(
    *,
    previous_config: str,
    current_config: str,
    formal_runs: dict,
    formal_selections: dict,
    budget_runs: dict,
    budget_selections: dict,
) -> tuple[dict, dict, str]:
    """Prefer a completed shared-budget pair after a budget extension."""

    if all(
        budget_selections.get(config_id, {}).get("status") == "complete"
        for config_id in (previous_config, current_config)
    ):
        return budget_runs, budget_selections, "1m_common_budget"
    return formal_runs, formal_selections, "500k_formal_budget"


def hierarchical_bootstrap_checkpoint_difference(
    lower_records: list[dict],
    upper_records: list[dict],
    *,
    lower_step: int,
    upper_step: int,
    variants: list[str],
    comparison_id: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Paired seed/variant/episode bootstrap for an adjacent scaling delta."""

    if len(lower_records) != len(FORMAL_SEEDS) or len(upper_records) != len(
        FORMAL_SEEDS
    ):
        raise ValueError("hierarchical bootstrap requires three records per setting")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")

    def episode_array(records: list[dict], step: int) -> np.ndarray:
        by_seed = []
        for record in records:
            checkpoint = next(
                row for row in record["checkpoints"] if row["step"] == step
            )
            by_seed.append(
                [
                    checkpoint["test"]["per_variant_episode_success"][variant]
                    for variant in variants
                ]
            )
        values = np.asarray(by_seed, dtype=np.float64)
        expected_shape = (
            len(FORMAL_SEEDS),
            len(variants),
            EPISODES_PER_VARIANT,
        )
        if values.shape != expected_shape:
            raise ValueError(
                f"bootstrap episode tensor {values.shape} != {expected_shape}"
            )
        return values

    lower = episode_array(lower_records, lower_step)
    upper = episode_array(upper_records, upper_step)
    paired_difference = upper - lower
    seed_bytes = hashlib.sha256(comparison_id.encode("utf-8")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
    seed_indices = rng.integers(
        0,
        len(FORMAL_SEEDS),
        size=(replicates, len(FORMAL_SEEDS), 1, 1),
    )
    variant_indices = rng.integers(
        0,
        len(variants),
        size=(replicates, 1, len(variants), 1),
    )
    episode_indices = rng.integers(
        0,
        EPISODES_PER_VARIANT,
        size=(
            replicates,
            len(FORMAL_SEEDS),
            len(variants),
            EPISODES_PER_VARIANT,
        ),
    )
    bootstrap_differences = paired_difference[
        seed_indices,
        variant_indices,
        episode_indices,
    ].mean(axis=(1, 2, 3))
    lower_bound, upper_bound = np.quantile(
        bootstrap_differences, [0.025, 0.975]
    )
    per_variant = paired_difference.mean(axis=(0, 2))
    per_seed = paired_difference.mean(axis=(1, 2))
    return {
        "comparison_id": comparison_id,
        "lower_step": lower_step,
        "upper_step": upper_step,
        "observed_test_delta": float(paired_difference.mean()),
        "bootstrap_replicates": replicates,
        "confidence_level": 0.95,
        "confidence_interval": [
            float(lower_bound),
            float(upper_bound),
        ],
        "confidence_interval_contains_zero": bool(
            lower_bound <= 0.0 <= upper_bound
        ),
        "individual_seed_deltas": [float(value) for value in per_seed],
        "per_variant_deltas": {
            variant: float(value)
            for variant, value in zip(variants, per_variant, strict=True)
        },
        "positive_variant_count": int(np.sum(per_variant > 0.0)),
        "negative_variant_count": int(np.sum(per_variant < 0.0)),
        "zero_variant_count": int(np.sum(per_variant == 0.0)),
    }


def _adjacent_scaling_comparisons(
    *,
    family: str,
    axis: str,
    config_order: tuple[str, ...],
    runs: dict,
    selections: dict,
) -> list[dict]:
    comparisons = []
    variants = FAMILIES[family]["test"]
    for lower_id, upper_id in zip(
        config_order[:-1], config_order[1:], strict=True
    ):
        lower_selection = selections.get(lower_id, {})
        upper_selection = selections.get(upper_id, {})
        if (
            lower_selection.get("status") != "complete"
            or upper_selection.get("status") != "complete"
        ):
            continue
        lower_records = [
            record
            for record in runs[lower_id]
            if record["status"] == "complete"
        ]
        upper_records = [
            record
            for record in runs[upper_id]
            if record["status"] == "complete"
        ]
        comparison_id = f"{family}:{axis}:{lower_id}->{upper_id}"
        record = hierarchical_bootstrap_checkpoint_difference(
            lower_records,
            upper_records,
            lower_step=lower_selection["selected_step"],
            upper_step=upper_selection["selected_step"],
            variants=variants,
            comparison_id=comparison_id,
        )
        record.update(
            {
                "family": family,
                "axis": axis,
                "lower_config": lower_id,
                "upper_config": upper_id,
            }
        )
        comparisons.append(record)
    return comparisons


def _assert_protocol(
    *,
    run_id: str,
    config: dict,
    family: str,
    expected_network: dict,
    updates: int,
) -> None:
    groups = FAMILIES[family]
    if config["algorithm"] != "mlp_bc":
        raise AssertionError(f"{run_id}: algorithm must be mlp_bc")
    if config["n_steps"] != updates or config["n_steps_per_epoch"] != 25_000:
        raise AssertionError(f"{run_id}: formal update budget mismatch")
    if config["save_interval_epochs"] != 4:
        raise AssertionError(f"{run_id}: checkpoint cadence mismatch")
    if config["train_variants"] != groups["train"]:
        raise AssertionError(f"{run_id}: train variant list mismatch")
    if config["eval_variants"] != groups["train"] + groups["test"]:
        raise AssertionError(f"{run_id}: evaluation variant list mismatch")
    if config["observation"] != OBSERVATION:
        raise AssertionError(f"{run_id}: observation configuration mismatch")
    if config.get("training_diagnostics") != TRAINING_DIAGNOSTICS:
        raise AssertionError(f"{run_id}: training diagnostic configuration mismatch")
    if not config["balance_variant_episode_count"] or config["episode_keep_num"] != 300:
        raise AssertionError(f"{run_id}: episode balancing/limit mismatch")
    if config["sampling_seed"] != 0 or config["train_data_ratio"] != 0.9:
        raise AssertionError(f"{run_id}: dataset sampling protocol mismatch")
    evaluation = config["evaluation"]
    if (
        evaluation["num_episodes"] != EPISODES_PER_VARIANT
        or evaluation["rollout_batch_size"] != ROLLOUT_BATCH_SIZE
        or evaluation["seed"] != 0
        or evaluation["every_epochs"] != 4
        or evaluation["env_config"].get("eval_start_goal_mode")
        != groups["eval_start_goal_mode"]
    ):
        raise AssertionError(f"{run_id}: rollout protocol mismatch")
    network = config["network"]
    for key, expected_value in expected_network.items():
        if key in {
            "batch_size",
            "learning_rate",
            "learning_rate_by_family",
            "learning_rate_candidates_by_family",
        }:
            continue
        if network.get(key) != expected_value:
            raise AssertionError(
                f"{run_id}: network.{key}={network.get(key)!r}, expected {expected_value!r}"
            )
    expected_batch = expected_network.get("batch_size", 256)
    if config["algorithm_config"]["batch_size"] != expected_batch:
        raise AssertionError(f"{run_id}: batch size mismatch")
    actual_lr = config["algorithm_config"]["learning_rate"]
    allowed_lrs = allowed_learning_rates(expected_network, family)
    if actual_lr not in allowed_lrs:
        raise AssertionError(
            f"{run_id}: learning rate {actual_lr} not in {allowed_lrs}"
        )


def _audit_rollout(rollout: dict, expected_variants: list[str]) -> None:
    if set(rollout["variants"]) != set(expected_variants):
        raise AssertionError("rollout variants do not match the fixed protocol")
    for variant in expected_variants:
        record = rollout["variants"][variant]
        episodes = record["episodes"]
        if record["num_episodes"] != EPISODES_PER_VARIANT:
            raise AssertionError(f"{variant}: episode count mismatch")
        if [episode["seed"] for episode in episodes] != list(range(EPISODES_PER_VARIANT)):
            raise AssertionError(f"{variant}: reset seeds are not 0..29")
        if record["successful_episode_count"] != sum(
            bool(episode["success"]) for episode in episodes
        ):
            raise AssertionError(f"{variant}: successful episode count mismatch")


def _assert_no_formal_collapse(
    diagnostics: list[dict], *, run_id: str
) -> None:
    if not diagnostics:
        raise AssertionError(f"{run_id}: missing diagnostics for collapse check")
    tail = diagnostics[-min(3, len(diagnostics)) :]
    tail_min_action_std = min(
        value for record in tail for value in record["action_std"]
    )
    if tail_min_action_std < MIN_FORMAL_TAIL_ACTION_STD:
        raise AssertionError(
            f"{run_id}: tail action std {tail_min_action_std:.6g} is below "
            f"{MIN_FORMAL_TAIL_ACTION_STD:.2f}"
        )
    positive_gradient_fraction = sum(
        record["gradient_global_norm_mean"] > 1e-8
        for record in diagnostics
    ) / len(diagnostics)
    if (
        positive_gradient_fraction
        < MIN_FORMAL_POSITIVE_GRADIENT_FRACTION
    ):
        raise AssertionError(
            f"{run_id}: positive-gradient epoch fraction "
            f"{positive_gradient_fraction:.3f} is below "
            f"{MIN_FORMAL_POSITIVE_GRADIENT_FRACTION:.2f}"
        )


def _audit_training_diagnostics(
    diagnostics: list[dict], *, family: str, run_id: str, batch_size: int, updates: int
) -> dict[int, dict]:
    total_epochs = updates // 25_000
    if len(diagnostics) != total_epochs:
        raise AssertionError(
            f"{run_id}: expected {total_epochs} epoch diagnostics"
        )
    by_step = {}
    action_dim = 2 if family == "pointmaze" else 8
    for expected_epoch, record in enumerate(diagnostics, start=1):
        expected_step = expected_epoch * 25_000
        if record.get("epoch") != expected_epoch or record.get("step") != expected_step:
            raise AssertionError(f"{run_id}: diagnostic epoch/step sequence mismatch")
        if record.get("processed_examples") != expected_step * batch_size:
            raise AssertionError(f"{run_id}: diagnostic processed-example count mismatch")
        if record.get("action_probe_count") != 1024:
            raise AssertionError(f"{run_id}: diagnostic action probe size mismatch")
        if record.get("gradient_global_norm_sample_count") != 25:
            raise AssertionError(f"{run_id}: diagnostic gradient sample count mismatch")
        scalar_fields = (
            "gradient_global_norm_mean",
            "gradient_global_norm_max",
            "gradient_global_norm_last",
            "parameter_global_norm",
            "action_abs_mean",
            "epoch_wall_time_seconds",
            "wall_time_seconds",
            "updates_per_second",
            "examples_per_second",
        )
        for field in scalar_fields:
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise AssertionError(f"{run_id}: non-finite diagnostic {field}")
        for field in ("action_mean", "action_std", "action_min", "action_max"):
            values = record.get(field)
            if (
                not isinstance(values, list)
                or len(values) != action_dim
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
            ):
                raise AssertionError(f"{run_id}: invalid diagnostic {field}")
        if record["parameter_global_norm"] <= 0.0:
            raise AssertionError(f"{run_id}: non-positive parameter norm")
        if record["epoch_wall_time_seconds"] <= 0.0 or record["updates_per_second"] <= 0.0:
            raise AssertionError(f"{run_id}: non-positive timing diagnostic")
        if (
            not isinstance(record.get("gpu_peak_allocated_bytes"), int)
            or record["gpu_peak_allocated_bytes"] < 0
        ):
            raise AssertionError(f"{run_id}: invalid GPU memory diagnostic")
        if (
            not isinstance(record.get("process_peak_rss_bytes"), int)
            or record["process_peak_rss_bytes"] <= 0
        ):
            raise AssertionError(f"{run_id}: invalid RSS diagnostic")
        by_step[expected_step] = record
    _assert_no_formal_collapse(diagnostics, run_id=run_id)
    return by_step


def _audit_run(
    family: str,
    config_id: str,
    seed: int,
    *,
    updates: int = 500_000,
    checkpoint_steps: tuple[int, ...] = FORMAL_STEPS,
    run_kind: str = "formal",
    expected_network: dict | None = None,
    required_learning_rate: float | None = None,
) -> dict:
    prefix = _run_prefix(
        family, config_id, seed, updates=updates, run_kind=run_kind
    )
    try:
        run_dir = _resolve_run_dir(
            family,
            config_id,
            seed,
            updates=updates,
            run_kind=run_kind,
            required_learning_rate=required_learning_rate,
        )
        if run_dir is None:
            return {"status": "missing", "experiment_id": f"{prefix}<date>"}
        run_id = run_dir.name
        import yaml

        resolved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        if (
            required_learning_rate is not None
            and resolved["algorithm_config"]["learning_rate"]
            != required_learning_rate
        ):
            raise AssertionError(
                f"{run_id}: learning rate must be {required_learning_rate}"
            )
        _assert_protocol(
            run_id=run_id,
            config=resolved,
            family=family,
            expected_network=(
                CONFIGS[config_id] if expected_network is None else expected_network
            ),
            updates=updates,
        )
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            return {"status": "in_progress", "experiment_id": run_id}
        summary = _load_json(summary_path)
        diagnostics_by_step = _audit_training_diagnostics(
            summary["training_diagnostics"],
            family=family,
            run_id=run_id,
            batch_size=resolved["algorithm_config"]["batch_size"],
            updates=updates,
        )
        evaluations = {int(record["step"]): record for record in summary["evaluation_history"]}
        checkpoint_artifacts = summary.get("checkpoint_artifacts", {})
        rows = []
        expected_variants = FAMILIES[family]["train"] + FAMILIES[family]["test"]
        for step in checkpoint_steps:
            artifact = checkpoint_artifacts.get(str(step), {})
            checkpoint = Path(
                artifact.get(
                    "checkpoint",
                    run_dir / "checkpoints" / f"step_{step}.d3",
                )
            )
            if not checkpoint.is_file():
                raise AssertionError(f"missing checkpoint {checkpoint.name}")
            trainer_state_value = artifact.get("trainer_state")
            trainer_state = (
                Path(trainer_state_value)
                if trainer_state_value is not None
                else run_dir
                / "checkpoints"
                / f"trainer_state_step_{step}.pt"
            )
            record = evaluations.get(step)
            if record is None:
                raise AssertionError(f"missing evaluation at step {step}")
            if record.get("training_diagnostics") != diagnostics_by_step[step]:
                raise AssertionError(f"{run_id}: checkpoint diagnostic mismatch at {step}")
            _audit_rollout(record["rollout"], expected_variants)
            rows.append(
                {
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "trainer_state": (
                        str(trainer_state) if trainer_state.is_file() else None
                    ),
                    "train": _group_metrics(record["rollout"], FAMILIES[family]["train"]),
                    "test": _group_metrics(record["rollout"], FAMILIES[family]["test"]),
                    "validation": record["validation"],
                }
            )
        metadata = _load_json(run_dir / "model_metadata.json")
        final_diagnostic = diagnostics_by_step[updates]
        wall_time_seconds = final_diagnostic["wall_time_seconds"]
        return {
            "status": "complete",
            "experiment_id": run_id,
            "learning_rate": resolved["algorithm_config"]["learning_rate"],
            "model": metadata,
            "resources": {
                "peak_gpu_allocated_bytes": max(
                    row["gpu_peak_allocated_bytes"]
                    for row in diagnostics_by_step.values()
                ),
                "peak_process_rss_bytes": max(
                    row["process_peak_rss_bytes"]
                    for row in diagnostics_by_step.values()
                ),
                "diagnostic_wall_time_seconds": wall_time_seconds,
                "effective_updates_per_second": updates / wall_time_seconds,
                "processed_examples": final_diagnostic["processed_examples"],
                "wall_time_scope": (
                    "through_final_epoch_callback_before_final_rollout"
                ),
            },
            "checkpoints": rows,
        }
    except Exception as exc:  # preserve a complete matrix even when one run fails audit
        return {"status": "invalid", "experiment_id": prefix, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="缺少或不合规 run 时返回失败")
    args = parser.parse_args()
    output = {
        "protocol": {
            "updates": 500_000,
            "checkpoint_steps": list(FORMAL_STEPS),
            "episodes_per_variant": EPISODES_PER_VARIANT,
            "rollout_batch_size": ROLLOUT_BATCH_SIZE,
            "observation": OBSERVATION,
            "training_diagnostics": TRAINING_DIAGNOSTICS,
            "train_test_semantics": "训练期间在固定测试地图 rollout；测试地图不参与离线训练或 scaler 拟合。",
        },
        "checkpoint_selection_protocol": {
            "seed_count": len(FORMAL_SEEDS),
            "shared_step_across_seeds": True,
            "joint_success": "0.5 * train_success_macro + 0.5 * test_success_macro",
            "required_declining_intervals": OVERFIT_REQUIRED_DECLINING_INTERVALS,
            "minimum_total_test_decline": OVERFIT_MIN_TOTAL_TEST_DECLINE,
            "recovery_tolerance": OVERFIT_RECOVERY_TOLERANCE,
            "endpoint_train_must_not_be_lower": True,
            "required_declining_seed_count": 2,
            "test_maps_participate_in_selection": True,
        },
        "budget_extension_protocol": {
            "tail_checkpoint_count": BUDGET_WINDOW_CHECKPOINTS,
            "minimum_test_gain": BUDGET_MIN_TEST_GAIN,
            "minimum_train_gain": BUDGET_MIN_TRAIN_GAIN,
            "maximum_test_drop_for_train_trigger": (
                BUDGET_MAX_TEST_DROP_FOR_TRAIN_TRIGGER
            ),
            "required_positive_seed_count": 2,
            "blocked_by_sustained_overfit": True,
            "pairing_rule": (
                "extend_with_smaller_neighbor_or_larger_neighbor_at_lower_boundary"
            ),
        },
        "families": FAMILIES,
        "required_formal_configs": {
            family: list(config_ids)
            for family, config_ids in REQUIRED_FORMAL_CONFIGS.items()
        },
        "conditional_formal_configs": {
            family: list(config_ids)
            for family, config_ids in CONDITIONAL_FORMAL_CONFIGS.items()
        },
        "required_budget_configs": {
            family: list(config_ids)
            for family, config_ids in REQUIRED_BUDGET_CONFIGS.items()
        },
        "runs": {},
        "selections": {},
        "budget_runs": {},
        "budget_selections": {},
        "depth_recipe_audits": {"pointmaze": {}, "antmaze": {}},
        "width_recipe_audits": {"pointmaze": {}, "antmaze": {}},
        "extension_runs": {},
        "extension_selections": {},
    }
    incomplete = []
    formal_configs = {
        family: list(_activated_formal_config_ids(family))
        for family in FAMILIES
    }
    output["formal_configs"] = {
        family: list(config_ids) for family, config_ids in formal_configs.items()
    }
    for family in FAMILIES:
        output["runs"][family] = {}
        output["selections"][family] = {}
        for config_id in CONFIGS:
            core_width_range = None
            required_learning_rate = None
            if (family, config_id) in CORE_WIDTH_RESCUES:
                try:
                    core_width_range = _audit_core_width_rescue_ladder(
                        family, config_id
                    )
                except Exception as exc:
                    core_width_range = {
                        "status": "invalid",
                        "error": str(exc),
                        "range_evidence": [],
                    }
                required_learning_rate = (
                    core_width_range["selected_learning_rate"]
                    if core_width_range.get("status") == "stable"
                    else -1.0
                )
            elif (family, config_id) in CORE_WIDTH_FOLLOWUPS:
                prerequisite_family, prerequisite_config = (
                    CORE_WIDTH_FOLLOWUPS[(family, config_id)]
                )
                prerequisite_recipe = output["width_recipe_audits"][
                    prerequisite_family
                ].get(prerequisite_config, {"status": "incomplete"})
                try:
                    core_width_range = _audit_followup_width_ladder(
                        family,
                        config_id,
                        prerequisite_recipe,
                    )
                except Exception as exc:
                    core_width_range = {
                        "status": "invalid",
                        "error": str(exc),
                        "range_evidence": [],
                    }
                required_learning_rate = (
                    core_width_range["selected_learning_rate"]
                    if core_width_range.get("status") == "stable"
                    else -1.0
                )
            rows = []
            for seed in FORMAL_SEEDS:
                record = _audit_run(
                    family,
                    config_id,
                    seed,
                    required_learning_rate=required_learning_rate,
                )
                rows.append(record)
            output["runs"][family][config_id] = rows
            complete = [record for record in rows if record["status"] == "complete"]
            recipe_valid = True
            if core_width_range is not None:
                recipe_audit = _audit_core_width_recipe(
                    family,
                    config_id,
                    rows,
                    core_width_range,
                    selection_rule=(
                        "wait for the preceding width's valid shared formal "
                        "recipe; start range from that LR, lower only on "
                        "instability, then share the first stable LR across "
                        "all three formal seeds"
                        if (family, config_id) in CORE_WIDTH_FOLLOWUPS
                        else None
                    ),
                )
                output["width_recipe_audits"][family][
                    config_id
                ] = recipe_audit
                recipe_valid = recipe_audit["status"] == "valid"
            if family == "pointmaze" and config_id == "r128-w256":
                try:
                    recipe_audit = _audit_depth_recipe(
                        family, config_id, rows
                    )
                except Exception as exc:
                    recipe_valid = False
                    recipe_audit = {
                        "status": "invalid",
                        "error": str(exc),
                    }
                    incomplete.append(
                        f"{family}/{config_id}: invalid depth recipe"
                    )
                output["depth_recipe_audits"][family][
                    config_id
                ] = recipe_audit
            if len(complete) == len(FORMAL_SEEDS) and recipe_valid:
                output["selections"][family][config_id] = select_best_checkpoint(
                    complete,
                    budget_extension_eligible=(
                        family == "pointmaze"
                        and config_id in BUDGET_EXTENSION_ELIGIBLE_CONFIGS
                    ),
                )
            else:
                output["selections"][family][config_id] = {
                    "status": (
                        "incomplete" if recipe_valid else "invalid_recipe"
                    ),
                    "complete_seed_count": len(complete),
                    "required_seed_count": len(FORMAL_SEEDS),
                }
    budget_configs = {
        family: list(
            _activated_budget_config_ids(
                family, output["selections"][family]
            )
        )
        for family in REQUIRED_BUDGET_CONFIGS
    }
    output["budget_configs"] = {
        family: list(config_ids) for family, config_ids in budget_configs.items()
    }
    for family, config_ids in budget_configs.items():
        output["budget_runs"][family] = {}
        output["budget_selections"][family] = {}
        for config_id in config_ids:
            rows = []
            for seed in FORMAL_SEEDS:
                record = _audit_run(
                    family,
                    config_id,
                    seed,
                    updates=1_000_000,
                    checkpoint_steps=MILLION_STEPS,
                    run_kind="budget",
                )
                rows.append(record)
            output["budget_runs"][family][config_id] = rows
            complete = [record for record in rows if record["status"] == "complete"]
            recipe_valid = True
            if family == "antmaze" and config_id in {
                "r64-w256",
                "r128-w256",
                "r256-w256",
            }:
                try:
                    recipe_audit = _audit_ant_depth_recipe(config_id, rows)
                except Exception as exc:
                    recipe_valid = False
                    recipe_audit = {
                        "status": "invalid",
                        "error": str(exc),
                    }
                    incomplete.append(f"antmaze/{config_id}: invalid depth recipe")
                output["depth_recipe_audits"]["antmaze"][
                    config_id
                ] = recipe_audit
            resolved_stability_failure = (
                family == "antmaze"
                and config_id == "r256-w256"
                and output["depth_recipe_audits"]["antmaze"]
                .get(config_id, {})
                .get("status")
                == "confirmed_cross_seed_instability"
            )
            if resolved_stability_failure:
                failure_evidence = output["depth_recipe_audits"][
                    "antmaze"
                ][config_id]
                for seed, record in zip(FORMAL_SEEDS, rows, strict=True):
                    record["status"] = (
                        "failed_action_collapse"
                        if seed == failure_evidence["failed_formal_seed"]
                        else "stopped_after_cross_seed_failure"
                    )
                    record["terminal_failure"] = failure_evidence
            if len(complete) == len(FORMAL_SEEDS) and recipe_valid:
                output["budget_selections"][family][config_id] = (
                    select_best_checkpoint(
                        complete,
                        budget_extension_eligible=(
                            family == "antmaze"
                            and config_id in BUDGET_EXTENSION_ELIGIBLE_CONFIGS
                        ),
                    )
                )
            elif resolved_stability_failure:
                output["budget_selections"][family][config_id] = {
                    "status": "failed_unstable",
                    "complete_seed_count": len(complete),
                    "required_seed_count": len(FORMAL_SEEDS),
                    "failure_evidence": output["depth_recipe_audits"][
                        "antmaze"
                    ][config_id],
                }
            else:
                output["budget_selections"][family][config_id] = {
                    "status": "incomplete" if recipe_valid else "invalid_recipe",
                    "complete_seed_count": len(complete),
                    "required_seed_count": len(FORMAL_SEEDS),
                }
    output["axis_early_stop_protocol"] = {
        "metric": (
            "cross_seed_mean_selected_checkpoint_test_success_macro"
        ),
        "consecutive_extension_count": 2,
        "minimum_meaningful_gain": AXIS_EARLY_STOP_MIN_GAIN,
        "comparison": "each of the latest two adjacent gains < threshold",
        "in_flight_policy": (
            "finish settings already started; do not launch later settings"
        ),
    }
    output["axis_early_stop_gates"] = {
        "pointmaze": _axis_early_stop_gates(
            output["selections"]["pointmaze"]
        ),
        "antmaze": _axis_early_stop_gates(
            output["budget_selections"]["antmaze"]
        ),
    }
    for family, config_ids in formal_configs.items():
        for config_id in config_ids:
            for record in output["runs"][family][config_id]:
                if record["status"] != "complete":
                    incomplete.append(record["experiment_id"])
    for family, config_ids in budget_configs.items():
        for config_id in config_ids:
            resolved_stability_failure = (
                family == "antmaze"
                and config_id == "r256-w256"
                and output["depth_recipe_audits"]["antmaze"]
                .get(config_id, {})
                .get("status")
                == "confirmed_cross_seed_instability"
            )
            for record in output["budget_runs"][family][config_id]:
                if (
                    record["status"] != "complete"
                    and not resolved_stability_failure
                ):
                    incomplete.append(record["experiment_id"])
    output["formal_configs"] = {
        family: list(config_ids) for family, config_ids in formal_configs.items()
    }
    output["budget_configs"] = {
        family: list(config_ids) for family, config_ids in budget_configs.items()
    }
    output["next_depth_range_gates"] = {
        "pointmaze": {
            "r512-w256": assess_next_depth_range_gate(
                family="pointmaze",
                previous_config="r128-w256",
                current_config="r256-w256",
                next_config="r512-w256",
                runs=output["runs"]["pointmaze"],
                selections=output["selections"]["pointmaze"],
            ),
            "r1024-w256": assess_next_depth_range_gate(
                family="pointmaze",
                previous_config="r256-w256",
                current_config="r512-w256",
                next_config="r1024-w256",
                runs=output["runs"]["pointmaze"],
                selections=output["selections"]["pointmaze"],
            ),
        },
        "antmaze": {
            "r512-w256": assess_next_depth_range_gate(
                family="antmaze",
                previous_config="r128-w256",
                current_config="r256-w256",
                next_config="r512-w256",
                runs=output["budget_runs"]["antmaze"],
                selections=output["budget_selections"]["antmaze"],
                point_prerequisite_selection=output["selections"][
                    "pointmaze"
                ].get("r512-w256"),
            ),
            "r1024-w256": assess_next_depth_range_gate(
                family="antmaze",
                previous_config="r256-w256",
                current_config="r512-w256",
                next_config="r1024-w256",
                runs=output["budget_runs"]["antmaze"],
                selections=output["budget_selections"]["antmaze"],
                point_prerequisite_selection=output["selections"][
                    "pointmaze"
                ].get("r1024-w256"),
            ),
        },
    }
    (
        point_width_gate_runs,
        point_width_gate_selections,
        point_width_gate_evidence_stage,
    ) = _select_width_gate_evidence(
        previous_config="w3072-d4",
        current_config="w4096-d4",
        formal_runs=output["runs"]["pointmaze"],
        formal_selections=output["selections"]["pointmaze"],
        budget_runs=output["budget_runs"]["pointmaze"],
        budget_selections=output["budget_selections"]["pointmaze"],
    )
    point_width_gate = assess_next_width_range_gate(
        family="pointmaze",
        previous_config="w3072-d4",
        current_config="w4096-d4",
        next_config="w5120-d4",
        runs=point_width_gate_runs,
        selections=point_width_gate_selections,
    )
    point_width_gate["evidence_stage"] = point_width_gate_evidence_stage
    point_w6144_gate = assess_next_width_range_gate(
        family="pointmaze",
        previous_config="w4096-d4",
        current_config="w5120-d4",
        next_config="w6144-d4",
        runs=output["runs"]["pointmaze"],
        selections=output["selections"]["pointmaze"],
    )
    point_w6144_gate["evidence_stage"] = "500k_formal_budget"
    point_w7168_gate = assess_next_width_range_gate(
        family="pointmaze",
        previous_config="w5120-d4",
        current_config="w6144-d4",
        next_config="w7168-d4",
        runs=output["runs"]["pointmaze"],
        selections=output["selections"]["pointmaze"],
        continue_until_two_low_gains=True,
    )
    point_w7168_gate["evidence_stage"] = "500k_formal_budget"
    point_w7168_gate["continuation_rule"] = (
        "run one more +1024 width while the latest two adjacent selected-test "
        "gains have not both fallen below 2 pp"
    )
    point_w8192_gate = assess_next_width_range_gate(
        family="pointmaze",
        previous_config="w6144-d4",
        current_config="w7168-d4",
        next_config="w8192-d4",
        runs=output["runs"]["pointmaze"],
        selections=output["selections"]["pointmaze"],
        continue_until_two_low_gains=True,
    )
    point_w8192_gate["evidence_stage"] = "500k_formal_budget"
    point_w8192_gate["continuation_rule"] = (
        "stop when the latest two adjacent selected-test gains are both "
        "strictly below 2 pp; otherwise run the next +1024 width range"
    )
    point_w8192_gate["protocol_gate_status"] = point_w8192_gate["status"]
    point_w8192_gate["status"] = "stopped_by_user_decision"
    point_w8192_gate["decision_date"] = "2026-07-30"
    point_w8192_gate["decision_reason"] = (
        "stop PointMaze width scaling at W7168: W5120 is the observed peak "
        "and W6144/W7168 are fluctuations without a sustained gain"
    )
    point_w8192_gate["partial_range_run_id"] = (
        "bc-scaling-range-pointmaze-w8192-d4-s0-lr3e5-50k-20260730"
    )
    output["next_width_range_gates"] = {
        "pointmaze": {
            "w5120-d4": point_width_gate,
            "w6144-d4": point_w6144_gate,
            "w7168-d4": point_w7168_gate,
            "w8192-d4": point_w8192_gate,
        },
        "antmaze": {
            "w5120-d4": assess_next_width_range_gate(
                family="antmaze",
                previous_config="w3072-d4",
                current_config="w4096-d4",
                next_config="w5120-d4",
                runs=output["budget_runs"]["antmaze"],
                selections=output["budget_selections"]["antmaze"],
            ),
        },
    }
    conditional_specs = (
        ("pointmaze", "r512-w256", "depth", "formal"),
        ("pointmaze", "r1024-w256", "depth", "formal"),
        ("pointmaze", "w5120-d4", "width", "formal"),
        ("pointmaze", "w6144-d4", "width", "formal"),
        ("pointmaze", "w7168-d4", "width", "formal"),
        ("pointmaze", "w8192-d4", "width", "formal"),
        ("antmaze", "r512-w256", "depth", "budget"),
        ("antmaze", "r1024-w256", "depth", "budget"),
        ("antmaze", "w5120-d4", "width", "budget"),
    )
    output["conditional_range_requirements"] = {
        family: {} for family in FAMILIES
    }
    for family, config_id, axis, run_kind in conditional_specs:
        gate = (
            output["next_depth_range_gates"][family][config_id]
            if axis == "depth"
            else output["next_width_range_gates"][family][config_id]
        )
        early_stop_gate = output["axis_early_stop_gates"][family][
            axis
        ][config_id]
        prerequisite_failures = _conditional_prerequisite_failures(
            family=family,
            config_id=config_id,
            requirements=output["conditional_range_requirements"],
        )
        if (
            gate.get("status") == "eligible_for_50k_range"
            and early_stop_gate.get("status") != "stop"
            and prerequisite_failures
        ):
            requirement = {
                "status": "blocked_by_conditional_prerequisite",
                "family": family,
                "config_id": config_id,
                "axis": axis,
                "gate_status": gate["status"],
                "failed_prerequisites": prerequisite_failures,
            }
        else:
            requirement = _audit_conditional_range_requirement(
                family=family,
                config_id=config_id,
                axis=axis,
                gate=gate,
                early_stop_gate=early_stop_gate,
            )
        output["conditional_range_requirements"][family][
            config_id
        ] = requirement
        active_configs = (
            formal_configs[family]
            if run_kind == "formal"
            else budget_configs[family]
        )
        was_active = config_id in active_configs
        status = requirement["status"]
        if status in {"missing", "invalid"}:
            incomplete.append(
                f"conditional-range/{family}/{config_id}: {status}"
            )
        if status != "stable":
            if was_active:
                requirement["formal_recipe"] = {
                    "status": "invalid",
                    "error": (
                        "conditional matrix was activated before its gate and "
                        "50k range produced a stable recipe"
                    ),
                }
                incomplete.append(
                    f"conditional-matrix/{family}/{config_id}: "
                    "activated_without_stable_range"
                )
            continue

        if not was_active:
            active_configs.append(config_id)
        if run_kind == "formal":
            rows = output["runs"][family][config_id]
            selection_bucket = output["selections"][family]
        else:
            if config_id not in output["budget_runs"][family]:
                rows = [
                    _audit_run(
                        family,
                        config_id,
                        seed,
                        updates=1_000_000,
                        checkpoint_steps=MILLION_STEPS,
                        run_kind="budget",
                    )
                    for seed in FORMAL_SEEDS
                ]
                output["budget_runs"][family][config_id] = rows
                complete = [
                    record
                    for record in rows
                    if record["status"] == "complete"
                ]
                output["budget_selections"][family][config_id] = (
                    select_best_checkpoint(
                        complete,
                        budget_extension_eligible=True,
                    )
                    if len(complete) == len(FORMAL_SEEDS)
                    else {
                        "status": "incomplete",
                        "complete_seed_count": len(complete),
                        "required_seed_count": len(FORMAL_SEEDS),
                    }
                )
            rows = output["budget_runs"][family][config_id]
            selection_bucket = output["budget_selections"][family]
        for record in rows:
            if record["status"] != "complete":
                incomplete.append(record["experiment_id"])
        recipe = _validate_conditional_formal_recipe(requirement, rows)
        requirement["formal_recipe"] = recipe
        if recipe["status"] == "invalid":
            incomplete.append(
                f"conditional-matrix/{family}/{config_id}: invalid recipe"
            )
            selection_bucket[config_id]["protocol_status"] = "invalid_recipe"

    output["formal_configs"] = {
        family: list(config_ids) for family, config_ids in formal_configs.items()
    }
    output["budget_configs"] = {
        family: list(config_ids) for family, config_ids in budget_configs.items()
    }
    output["budget_extension_recommendations"] = {
        "pointmaze_500k_to_1m": _group_budget_extension_recommendations(
            output["selections"]["pointmaze"]
        ),
        "antmaze_1m_to_2m": _group_budget_extension_recommendations(
            output["budget_selections"]["antmaze"]
        ),
    }
    output["scaling_comparisons"] = {
        "pointmaze_500k": {
            "width": _adjacent_scaling_comparisons(
                family="pointmaze",
                axis="width",
                config_order=WIDTH_CONFIG_ORDER,
                runs=output["runs"]["pointmaze"],
                selections=output["selections"]["pointmaze"],
            ),
            "depth": _adjacent_scaling_comparisons(
                family="pointmaze",
                axis="depth",
                config_order=DEPTH_CONFIG_ORDER,
                runs=output["runs"]["pointmaze"],
                selections=output["selections"]["pointmaze"],
            ),
        },
        "antmaze_1m": {
            "width": _adjacent_scaling_comparisons(
                family="antmaze",
                axis="width",
                config_order=WIDTH_CONFIG_ORDER,
                runs=output["budget_runs"]["antmaze"],
                selections=output["budget_selections"]["antmaze"],
            ),
            "depth": _adjacent_scaling_comparisons(
                family="antmaze",
                axis="depth",
                config_order=DEPTH_CONFIG_ORDER,
                runs=output["budget_runs"]["antmaze"],
                selections=output["budget_selections"]["antmaze"],
            ),
        },
    }
    extension_configs = {
        family: _activated_extension_config_ids(
            family, output["budget_selections"][family]
        )
        for family in FAMILIES
    }
    output["extension_configs"] = {
        family: list(config_ids)
        for family, config_ids in extension_configs.items()
    }
    for family, config_ids in extension_configs.items():
        output["extension_runs"][family] = {}
        output["extension_selections"][family] = {}
        for config_id in config_ids:
            rows = []
            for seed in FORMAL_SEEDS:
                record = _audit_run(
                    family,
                    config_id,
                    seed,
                    updates=2_000_000,
                    checkpoint_steps=TWO_MILLION_STEPS,
                    run_kind="extension",
                )
                rows.append(record)
                if record["status"] != "complete":
                    incomplete.append(record["experiment_id"])
            output["extension_runs"][family][config_id] = rows
            complete = [record for record in rows if record["status"] == "complete"]
            if len(complete) == len(FORMAL_SEEDS):
                output["extension_selections"][family][config_id] = (
                    select_best_checkpoint(
                        complete, budget_extension_eligible=False
                    )
                )
            else:
                output["extension_selections"][family][config_id] = {
                    "status": "incomplete",
                    "complete_seed_count": len(complete),
                    "required_seed_count": len(FORMAL_SEEDS),
                }
    output["controls"] = {}
    for config_id, spec in CONTROL_CONFIGS.items():
        record = _audit_run(
            "pointmaze",
            config_id,
            0,
            updates=spec["updates"],
            checkpoint_steps=spec["checkpoint_steps"],
            run_kind="control",
            expected_network=spec["network"],
        )
        output["controls"][config_id] = record
        if record["status"] != "complete":
            incomplete.append(record["experiment_id"])
    incomplete = list(dict.fromkeys(incomplete))
    output["incomplete_requirements"] = incomplete
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    complete_count = sum(
        record["status"] == "complete"
        for family in output["runs"].values()
        for config_rows in family.values()
        for record in config_rows
    )
    budget_complete_count = sum(
        record["status"] == "complete"
        for family in output["budget_runs"].values()
        for config_rows in family.values()
        for record in config_rows
    )
    extension_complete_count = sum(
        record["status"] == "complete"
        for family in output["extension_runs"].values()
        for config_rows in family.values()
        for record in config_rows
    )
    print(
        f"已审计 formal run：{complete_count}；"
        f"共同预算扩展 run：{budget_complete_count}；"
        f"2M 条件延长 run：{extension_complete_count}；"
        f"冻结计划所需但未完成或不合规：{len(incomplete)}"
    )
    print(f"审计文件：{OUTPUT}")
    if args.strict and incomplete:
        raise SystemExit("冻结 scaling 矩阵尚未完整或存在不合规 run")


if __name__ == "__main__":
    main()
