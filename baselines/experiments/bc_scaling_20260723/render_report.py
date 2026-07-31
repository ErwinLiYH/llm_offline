"""从审计 JSON 生成中文 BC scaling 实验报告。"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "reports/bc_scaling_20260723_checkpoint_audit.json"
OUTPUT = Path(__file__).with_name("report.md")
STEPS = (100_000, 200_000, 300_000, 400_000, 500_000)
MILLION_STEPS = tuple(range(100_000, 1_000_001, 100_000))
TWO_MILLION_STEPS = tuple(range(100_000, 2_000_001, 100_000))
FORMAL_SEEDS = (0, 1, 2)


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return _percent(values[0])
    return f"{_percent(mean(values))} ± {100 * stdev(values):.2f} pp"


def _mean_std_scalar(values: list[float], *, digits: int = 2) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return (
        f"{mean(values):.{digits}f} ± "
        f"{stdev(values):.{digits}f}"
    )


def _summary_cell(summary: dict) -> str:
    return f"{_percent(summary['mean'])} ± {100 * summary['std']:.2f} pp"


def _signed_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def _checkpoint(record: dict, step: int) -> dict:
    return next(row for row in record["checkpoints"] if row["step"] == step)


def _append_completion_table(
    lines: list[str],
    title: str,
    runs: dict,
    *,
    included_configs: dict[str, list[str]] | None = None,
) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "| Family | 配置 | 完成 seeds | 状态 |",
        "| --- | --- | ---: | --- |",
    ])
    for family, configs in runs.items():
        for config_id, records in configs.items():
            if (
                included_configs is not None
                and config_id not in included_configs.get(family, [])
            ):
                continue
            complete = [record for record in records if record["status"] == "complete"]
            states = ", ".join(record["status"] for record in records)
            lines.append(f"| {family} | `{config_id}` | {len(complete)}/3 | {states} |")


def _append_macro_table(
    lines: list[str],
    title: str,
    runs: dict,
    steps: tuple[int, ...],
) -> None:
    step_headers = " | ".join(f"{step // 1000}k" for step in steps)
    lines.extend([
        "",
        f"## {title}",
        "",
        "每个 cell 先对每张地图 success rate 等权宏平均，再对训练 seed 汇总。"
        "格式为 `训练图 / 测试图`；只有完成的 run 才进入数值。",
        "",
        f"| Family / 配置 | {step_headers} | 参数量 |",
        "| --- | " + " | ".join("---" for _ in steps) + " | ---: |",
    ])
    for family, configs in runs.items():
        for config_id, records in configs.items():
            complete = [record for record in records if record["status"] == "complete"]
            if not complete:
                continue
            cells = []
            for step in steps:
                train = [
                    _checkpoint(record, step)["train"]["success_macro"]
                    for record in complete
                ]
                test = [
                    _checkpoint(record, step)["test"]["success_macro"]
                    for record in complete
                ]
                cells.append(f"{_mean_std(train)} / {_mean_std(test)}")
            parameters = [record["model"]["trainable_parameter_count"] for record in complete]
            lines.append(
                f"| {family} / `{config_id}` | "
                + " | ".join(cells)
                + f" | {mean(parameters):,.0f} |"
            )


def _append_resource_table(lines: list[str], title: str, runs: dict) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "内存列取三个 seed 中的实测最大值；wall-clock 与 updates/s 对完成 seed "
        "汇总 mean ± sample std。wall-clock 截止末 epoch callback，包含此前 "
        "checkpoint rollout，但不包含其后的最终 rollout。",
        "",
        "| Family / 配置 | 参数量 | GPU peak | RSS peak | "
        "Wall-clock | Updates/s | Processed examples/seed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for family, configs in runs.items():
        for config_id, records in configs.items():
            complete = [
                record
                for record in records
                if record["status"] == "complete" and "resources" in record
            ]
            if not complete:
                continue
            parameter_counts = [
                record["model"]["trainable_parameter_count"]
                for record in complete
            ]
            gpu_peak_mib = max(
                record["resources"]["peak_gpu_allocated_bytes"]
                for record in complete
            ) / (1024 ** 2)
            rss_peak_gib = max(
                record["resources"]["peak_process_rss_bytes"]
                for record in complete
            ) / (1024 ** 3)
            wall_hours = [
                record["resources"]["diagnostic_wall_time_seconds"] / 3600
                for record in complete
            ]
            update_rates = [
                record["resources"]["effective_updates_per_second"]
                for record in complete
            ]
            processed_examples = {
                record["resources"]["processed_examples"]
                for record in complete
            }
            processed_cell = (
                f"{processed_examples.pop():,}"
                if len(processed_examples) == 1
                else "不一致"
            )
            lines.append(
                f"| {family} / `{config_id}` | "
                f"{mean(parameter_counts):,.0f} | "
                f"{gpu_peak_mib:,.1f} MiB | "
                f"{rss_peak_gib:,.2f} GiB | "
                f"{_mean_std_scalar(wall_hours)} h | "
                f"{_mean_std_scalar(update_rates)} | "
                f"{processed_cell} |"
            )


def _append_depth_recipe_table(lines: list[str], audits: dict) -> None:
    lines.extend([
        "",
        "## 深度学习率冻结证据",
        "",
        "D64/D128/D256 分别从论文起始值 `3e-4` 开始。若完整 range 不稳定，先用"
        "相同配置重跑一次，再依次测试 `1e-4 → 3e-5 → 1e-5`；没有观测到任何"
        "训练 update 的 `invalid` run 只允许同学习率重跑，不能触发降学习率。"
        "三个正式 seed 必须共享首个通过完整 range 的学习率。Ant D256 的最低"
        "LR 若仅剩一个动作维慢启动、其余稳定条件全部通过，可在机器可读 gate "
        "批准后从同一 50k checkpoint 精确续到 100k；原 50k 仍记为 unstable，"
        "100k 仍使用完全相同的全维稳定阈值。若 100k 仍只有 7/8 维活跃，"
        "则沿约三倍下降间隔从头测试 `3e-6` 的完整 50k range；两项都明确"
        "标为 post-hoc rescue，且不放宽稳定阈值。",
        "",
        "| Family | 配置 | 状态 | 正式学习率 | Range 证据 |",
        "| --- | --- | --- | ---: | --- |",
    ])
    for family, config_ids in (
        ("pointmaze", ("r128-w256",)),
        ("antmaze", ("r64-w256", "r128-w256", "r256-w256")),
    ):
        for config_id in config_ids:
            record = audits.get(family, {}).get(
                config_id, {"status": "incomplete"}
            )
            learning_rate = record.get("selected_learning_rate")
            learning_rate_text = (
                f"`{learning_rate:g}`" if learning_rate is not None else "—"
            )
            evidence = record.get("range_evidence", [])
            evidence_text = "<br>".join(
                f"`{row['experiment_id']}`: {row['status']}"
                for row in evidence
            )
            if not evidence_text:
                evidence_text = record.get("error", "尚未完成")
            status_text = record["status"]
            if record.get("range_status") is not None:
                status_text += f" (range: {record['range_status']})"
            lines.append(
                f"| {family} | `{config_id}` | {status_text} | "
                f"{learning_rate_text} | {evidence_text} |"
            )


def _append_width_recipe_table(lines: list[str], audits: dict) -> None:
    lines.extend([
        "",
        "## 核心宽网络学习率 rescue 证据",
        "",
        "若论文起始值 `3e-4` 在正式 pilot 的任一 seed 出现动作坍缩，该设置不再"
        "混用 per-seed 学习率。使用坍缩 seed 依次测试"
        "`1e-4 → 3e-5 → 1e-5`，并用首个通过完整 50k range 的学习率"
        "重新训练全部三个正式 seed；旧 `3e-4` run 仅作为 pilot 证据。",
        "",
        "| Family | 配置 | 状态 | 正式学习率 | Pilot / range 证据 |",
        "| --- | --- | --- | ---: | --- |",
    ])
    for family, config_id in (
        ("pointmaze", "w3072-d4"),
        ("pointmaze", "w4096-d4"),
    ):
        record = audits.get(family, {}).get(
            config_id, {"status": "incomplete"}
        )
        learning_rate = record.get("selected_learning_rate")
        learning_rate_text = (
            f"`{learning_rate:g}`" if learning_rate is not None else "—"
        )
        evidence_texts = []
        for row in record.get("range_evidence", []):
            experiment_id = row.get("experiment_id")
            if experiment_id is None:
                experiment_id = ", ".join(row.get("experiment_ids", []))
            label = row.get("candidate", "candidate")
            evidence_texts.append(
                f"{label} `{experiment_id or 'missing'}`: {row['status']}"
            )
        evidence_text = "<br>".join(evidence_texts)
        if not evidence_text:
            evidence_text = record.get("error", "尚未完成")
        lines.append(
            f"| {family} | `{config_id}` | {record['status']} | "
            f"{learning_rate_text} | {evidence_text} |"
        )


def _append_axis_early_stop_table(lines: list[str], gates: dict) -> None:
    lines.extend([
        "",
        "## Width/depth 连续无增益早停",
        "",
        "核心网格完成后，若最近两次相邻扩展在三 seed 共享 selected checkpoint "
        "上的 test 宏平均增量都严格小于 2 pp（含下降），该 family 的该轴停止，"
        "不再启动下一条件扩展。已启动的核心设置继续完成。",
        "",
        "| Family | Axis | 下一设置 | 状态 | 最近三设置 | 最近两个 Δ |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for family in ("pointmaze", "antmaze"):
        for axis in ("width", "depth"):
            for next_config, record in gates.get(family, {}).get(
                axis, {}
            ).items():
                deltas = record.get("latest_two_deltas")
                delta_text = (
                    " / ".join(_signed_pp(value) for value in deltas)
                    if deltas is not None
                    else "—"
                )
                lines.append(
                    f"| {family} | {axis} | `{next_config}` | "
                    f"{record['status']} | "
                    f"{' → '.join(f'`{value}`' for value in record['compared_configs'])} | "
                    f"{delta_text} |"
                )


def _append_conditional_range_table(
    lines: list[str], requirements: dict
) -> None:
    lines.extend([
        "",
        "## 条件结构 50k range 与正式矩阵状态",
        "",
        "| Family | 配置 | Axis | 状态 | 下一候选/正式 LR | Range 证据 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ])
    for family in ("pointmaze", "antmaze"):
        for config_id, record in requirements.get(family, {}).items():
            candidate = record.get("selected_learning_rate")
            candidate_text = (
                f"`{candidate:g}`"
                if candidate is not None
                else record.get("next_required_candidate", "—")
            )
            evidence = record.get("range_evidence", [])
            evidence_text = "<br>".join(
                f"`{row['experiment_id']}`: {row['status']}"
                for row in evidence
            )
            if not evidence_text:
                failed_prerequisites = record.get(
                    "failed_prerequisites", []
                )
                evidence_text = (
                    "未通过前置：" + ", ".join(failed_prerequisites)
                    if failed_prerequisites
                    else record.get("error", "—")
                )
            lines.append(
                f"| {family} | `{config_id}` | {record['axis']} | "
                f"{record['status']} | {candidate_text} | {evidence_text} |"
            )


def _append_next_depth_gate_table(lines: list[str], gates: dict) -> None:
    lines.extend([
        "",
        "## 条件深度进入 50k range 的门槛",
        "",
        "只有当前深度相对前一深度的 selected test 提升至少 3 pp、至少 2/3 seed"
        " 同向、正式 run 无塌缩，并且共同预算最后三个 checkpoint 已进入平台，"
        "才允许下一深度开始 50k range。平台固定定义为：没有持续过拟合，且窗口"
        "首尾 train 变化绝对值不超过 3 pp、test 不超过 2 pp。50k range 仍须"
        "独立验证学习率、显存、RSS、wall-clock 和动作分布，不能由本表直接批准"
        "三 seed 正式训练。",
        "",
        "| Family | 下一深度 | 状态 | Selected test Δ | 正向 seeds | "
        "尾段 train/test Δ | 失败门槛 |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ])
    for family in ("pointmaze", "antmaze"):
        for next_config in ("r512-w256", "r1024-w256"):
            record = gates.get(family, {}).get(
                next_config, {"status": "incomplete"}
            )
            if record["status"] == "incomplete":
                lines.append(
                    f"| {family} | `{next_config}` | incomplete | — | — | — | "
                    "当前/前一深度尚未完成 |"
                )
                continue
            lines.append(
                f"| {family} | `{next_config}` | {record['status']} | "
                f"{_signed_pp(record['selected_test_delta'])} | "
                f"{record['positive_seed_count']}/3 | "
                f"{_signed_pp(record['tail_train_delta'])} / "
                f"{_signed_pp(record['tail_test_delta'])} | "
                f"{', '.join(record['failed_gates']) or '无'} |"
            )


def _append_next_width_gate_table(lines: list[str], gates: dict) -> None:
    lines.extend([
        "",
        "## 条件宽度进入 50k range 的门槛",
        "",
        "W5120/W6144 只在当前宽度相对前一宽度的 selected test 提升至少"
        " 2 pp、至少 2/3 seed 同向、正式 run 无塌缩，且当前宽度的共同预算"
        "末三个 checkpoint 已按同一 3 pp train / 2 pp test 规则进入平台时"
        "启动。其后按冻结的连续两次低增益规则逐级 +1024：若最近两次增益"
        "尚未同时严格低于 2 pp，允许再运行一个宽度的 50k range；这不直接"
        "批准正式三 seed。",
        "",
        "| Family | 证据预算 | 状态 | Selected test Δ | 正向 seeds | "
        "尾段 train/test Δ | 失败门槛 |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ])
    known_configs = tuple(
        dict.fromkeys(
            config_id
            for family_gates in gates.values()
            for config_id in family_gates
        )
    )
    for family in ("pointmaze", "antmaze"):
        family_gates = gates.get(family, {})
        config_ids = tuple(family_gates) or known_configs
        for next_config in config_ids:
            record = family_gates.get(
                next_config,
                {
                    "status": "incomplete",
                    "previous_config": "前一宽度",
                    "current_config": "当前宽度",
                },
            )
            if record["status"] == "incomplete":
                lines.append(
                    f"| {family}/{next_config} | — | incomplete | — | — | — | "
                    f"{record.get('previous_config', '前一宽度')}/"
                    f"{record.get('current_config', '当前宽度')} 尚未完成 |"
                )
                continue
            lines.append(
                f"| {family}/{next_config} | "
                f"`{record.get('evidence_stage', 'common_budget')}` | "
                f"{record['status']} | "
                f"{_signed_pp(record['selected_test_delta'])} | "
                f"{record['positive_seed_count']}/3 | "
                f"{_signed_pp(record['tail_train_delta'])} / "
                f"{_signed_pp(record['tail_test_delta'])} | "
                f"{', '.join(record['failed_gates']) or '无'} |"
            )


def _append_pareto_table(
    lines: list[str],
    title: str,
    runs: dict,
    selections: dict,
    *,
    included_configs: dict[str, list[str]] | None = None,
) -> None:
    rows = []
    for family, configs in runs.items():
        for config_id, records in configs.items():
            if (
                included_configs is not None
                and config_id not in included_configs.get(family, [])
            ):
                continue
            selection = selections.get(family, {}).get(config_id, {})
            complete = [
                record
                for record in records
                if record["status"] == "complete" and "resources" in record
            ]
            if selection.get("status") != "complete" or len(complete) != 3:
                continue
            rows.append(
                {
                    "family": family,
                    "config_id": config_id,
                    "test": selection["selected"]["test"]["mean"],
                    "wall_hours": mean(
                        record["resources"]["diagnostic_wall_time_seconds"]
                        for record in complete
                    )
                    / 3600,
                    "gpu_mib": max(
                        record["resources"]["peak_gpu_allocated_bytes"]
                        for record in complete
                    )
                    / (1024 ** 2),
                    "parameter_count": mean(
                        record["model"]["trainable_parameter_count"]
                        for record in complete
                    ),
                }
            )

    def is_frontier(row: dict, cost_key: str) -> bool:
        return not any(
            other["family"] == row["family"]
            and other is not row
            and other[cost_key] <= row[cost_key]
            and other["test"] >= row["test"]
            and (
                other[cost_key] < row[cost_key]
                or other["test"] > row["test"]
            )
            for other in rows
        )

    lines.extend([
        "",
        f"## {title}",
        "",
        "同一 family、同一预算内，以 selected test success 为收益。若不存在另一个"
        "配置成本不更高且 test 不更低，并且至少一项严格更优，则该点位于对应"
        " Pareto frontier。wall-clock 使用三 seed 均值，GPU 使用三 seed 最大值。",
        "",
        "| Family / 配置 | Selected test | Test pp / M params | Wall-clock | "
        "GPU peak | Wall Pareto | GPU Pareto |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in rows:
        lines.append(
            f"| {row['family']} / `{row['config_id']}` | "
            f"{_percent(row['test'])} | "
            f"{100 * row['test'] / (row['parameter_count'] / 1_000_000):,.2f} | "
            f"{row['wall_hours']:.2f} h | "
            f"{row['gpu_mib']:,.1f} MiB | "
            f"{'是' if is_frontier(row, 'wall_hours') else '否'} | "
            f"{'是' if is_frontier(row, 'gpu_mib') else '否'} |"
        )


def _append_scaling_comparison_table(
    lines: list[str], title: str, comparisons: dict
) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "差值为 upper − lower 的 selected test success。95% interval 使用固定"
        "10,000 次配对 hierarchical bootstrap：依次重采样 training seed、测试"
        "地图，并在地图内对上下配置使用相同 episode 索引重采样。",
        "",
        "| Axis | 相邻配置 | Selected steps | Test Δ | 95% interval | "
        "逐 seed Δ | 正/负/零测试地图 |",
        "| --- | --- | --- | ---: | --- | --- | ---: |",
    ])
    for axis in ("width", "depth"):
        for record in comparisons.get(axis, []):
            interval = record["confidence_interval"]
            seed_deltas = ", ".join(
                _signed_pp(value)
                for value in record["individual_seed_deltas"]
            )
            lines.append(
                f"| {axis} | `{record['lower_config']}` → "
                f"`{record['upper_config']}` | "
                f"{record['lower_step'] // 1000}k → "
                f"{record['upper_step'] // 1000}k | "
                f"{_signed_pp(record['observed_test_delta'])} | "
                f"[{_signed_pp(interval[0])}, {_signed_pp(interval[1])}]"
                f"{'，含 0' if record['confidence_interval_contains_zero'] else ''} | "
                f"{seed_deltas} | "
                f"{record['positive_variant_count']}/"
                f"{record['negative_variant_count']}/"
                f"{record['zero_variant_count']} |"
            )


def _append_selection_table(lines: list[str], title: str, selections: dict) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "每个设置先在三 seed 均值曲线上排除持续过拟合后缀，再最大化"
        " `0.5 × train_success_macro + 0.5 × test_success_macro`。"
        "三个 seed 共用同一个 selected step，禁止逐 seed 挑峰值。",
        "`Trainer state` 显示 selected/endpoint 两个 step 各有多少个 seed 保存了"
        "可用于精确续训的 sidecar；性能 checkpoint 缺少 sidecar 仍可参与比较，"
        "但预算延长必须从头训练。",
        "",
        "| Family / 配置 | Selected step | Selected train / test / joint | "
        "Endpoint | Endpoint train / test | Gap selected / endpoint | "
        "Trainer state | 过拟合 | 预算延长 | "
        "Selected → endpoint |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ])
    for family, configs in selections.items():
        for config_id, selection in configs.items():
            if selection["status"] != "complete":
                continue
            selected = selection["selected"]
            endpoint = selection["endpoint"]
            overfit = selection["overfit"]
            if overfit["detected"]:
                overfit_cell = (
                    f"是：峰值 {overfit['peak_step'] // 1000}k，"
                    f"{overfit['decline_start_step'] // 1000}k 起下降"
                )
            else:
                overfit_cell = "否"
            extension = selection["budget_extension"]
            if extension["status"] == "complete":
                extension_cell = (
                    ("建议" if extension["recommended"] else "不建议")
                    + f"：末三点 train {_signed_pp(extension['train_delta'])} / "
                    + f"test {_signed_pp(extension['test_delta'])}"
                )
            else:
                extension_cell = (
                    "不适用（非共同预算末端）"
                    if extension["status"] == "not_applicable"
                    else "checkpoint 不足"
                )
            selected_trainer_states = sum(
                artifact.get("trainer_state") is not None
                for artifact in selection["selected_checkpoints"]
            )
            endpoint_trainer_states = sum(
                artifact.get("trainer_state") is not None
                for artifact in selection["endpoint_checkpoints"]
            )
            trainer_state_cell = (
                f"{selected_trainer_states}/3 / {endpoint_trainer_states}/3"
            )
            delta = selection["selected_to_endpoint"]
            lines.append(
                f"| {family} / `{config_id}` | "
                f"{selection['selected_step'] // 1000}k | "
                f"{_summary_cell(selected['train'])} / "
                f"{_summary_cell(selected['test'])} / "
                f"{_summary_cell(selected['joint'])} | "
                f"{selection['endpoint_step'] // 1000}k | "
                f"{_summary_cell(endpoint['train'])} / "
                f"{_summary_cell(endpoint['test'])} | "
                f"{_summary_cell(selected['generalization_gap'])} / "
                f"{_summary_cell(endpoint['generalization_gap'])} | "
                f"{trainer_state_cell} | "
                f"{overfit_cell} | "
                f"{extension_cell} | "
                f"train {_signed_pp(delta['train'])} / "
                f"test {_signed_pp(delta['test'])} / "
                f"gap {_signed_pp(delta['generalization_gap'])} |"
            )


def _append_individual_seed_table(
    lines: list[str], title: str, selections: dict
) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "每个设置的三个 seed 使用同一个 selected step。若 selected 与共同预算 "
        "endpoint 相同，则合并为 `selected+endpoint`，避免重复行。",
        "",
        "| Family / 配置 | Role | Step | Seed | Train | Test | Joint | Gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for family, configs in selections.items():
        for config_id, selection in configs.items():
            if selection.get("status") != "complete":
                continue
            checkpoints = [
                (
                    "selected",
                    selection["selected_step"],
                    selection["selected"],
                )
            ]
            if selection["endpoint_step"] == selection["selected_step"]:
                checkpoints[0] = (
                    "selected+endpoint",
                    selection["selected_step"],
                    selection["selected"],
                )
            else:
                checkpoints.append(
                    (
                        "endpoint",
                        selection["endpoint_step"],
                        selection["endpoint"],
                    )
                )
            for role, step, metrics in checkpoints:
                for seed in FORMAL_SEEDS:
                    train = metrics["train"]["individual_seeds"][seed]
                    test = metrics["test"]["individual_seeds"][seed]
                    joint = metrics["joint"]["individual_seeds"][seed]
                    gap = metrics["generalization_gap"][
                        "individual_seeds"
                    ][seed]
                    lines.append(
                        f"| {family} / `{config_id}` | {role} | "
                        f"{step // 1000}k | {seed} | {_percent(train)} | "
                        f"{_percent(test)} | {_percent(joint)} | "
                        f"{_percent(gap)} |"
                    )


def _append_budget_extension_recommendations(
    lines: list[str], recommendations: dict
) -> None:
    lines.extend([
        "",
        "## 条件预算延长建议",
        "",
        "这里只根据共同预算末三个 checkpoint 的冻结规则生成建议。触发点不会"
        "单独延长，而是与相邻 width/depth 配置成组；尚未完成共同预算三 seed"
        "的配置不会提前进入建议。",
        "",
        "| 阶段 | 触发配置 | 相邻成组 | 最终配置并集 |",
        "| --- | --- | --- | --- |",
    ])
    labels = {
        "pointmaze_500k_to_1m": "PointMaze 500k → 1M",
        "antmaze_1m_to_2m": "AntMaze 1M → 2M",
    }
    for key, label in labels.items():
        record = recommendations[key]
        triggers = ", ".join(f"`{value}`" for value in record["triggered_configs"])
        pairs = "; ".join(
            f"{pair['axis']}: "
            + "/".join(f"`{value}`" for value in pair["configs"])
            for pair in record["pairs"]
        )
        config_union = ", ".join(
            f"`{value}`" for value in record["config_union"]
        )
        lines.append(
            f"| {label} | {triggers or '无'} | {pairs or '无'} | "
            f"{config_union or '无'} |"
        )


def _append_per_variant_tables(
    lines: list[str],
    title: str,
    runs: dict,
    families: dict,
    steps: tuple[int, ...],
) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        "以下表格保留每个 checkpoint 的逐环境成功率。每个 cell 是三个训练 seed"
        " 在该环境、该 checkpoint 上各 30 episodes success rate 的 mean ± sample std。",
    ])
    step_headers = " | ".join(f"{step // 1000}k" for step in steps)
    for family, configs in runs.items():
        for config_id, records in configs.items():
            complete = [record for record in records if record["status"] == "complete"]
            if len(complete) != 3:
                continue
            lines.extend([
                "",
                f"### {family} / `{config_id}`",
                "",
                f"| Split | 环境 | {step_headers} |",
                "| --- | --- | " + " | ".join("---:" for _ in steps) + " |",
            ])
            for split in ("train", "test"):
                for variant in families[family][split]:
                    cells = []
                    for step in steps:
                        values = [
                            _checkpoint(record, step)[split]["per_variant_success"][
                                variant
                            ]
                            for record in complete
                        ]
                        cells.append(_mean_std(values))
                    lines.append(
                        f"| {split} | `{variant}` | " + " | ".join(cells) + " |"
                    )


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    lines = [
        "# CrossMaze BC Scaling 实验报告",
        "",
        "本报告只汇总通过固定协议审计的正式 run。训练图与测试图沿用既有多环境设置；"
        "测试图不进入离线数据采样或 observation scaler，但按既有协议在训练 checkpoint 时"
        "固定 rollout，并按预先冻结的 train/test 联合规则参与 checkpoint 选择。"
        "因此 selected-checkpoint 结果不是独立 held-out 推断；共同预算 endpoint 仍单独报告。",
        "",
        "## 固定设置",
        "",
        "- 训练：每变体最多 300 episode、变体平衡、90/10 episode train/validation split。",
        "- 输入：map + location sensing + wall sensing（v3，boundary threshold 0.10）。",
        "- 预算：PointMaze 共同预算 500k；AntMaze 后续正式共同预算 1M。",
        "- checkpoint：每 100k updates；每张训练图和测试图各 30 个 reset seed。",
        "- 网络：论文式 `Dense → LayerNorm → Swish`；残差 block 为 4 个 unit 后加 identity。",
    ]
    _append_completion_table(
        lines,
        "500k 正式矩阵完成状态",
        audit["runs"],
        included_configs=audit["formal_configs"],
    )
    _append_completion_table(
        lines,
        "1M 预算实验完成状态（Point 条件延长 / Ant 正式主矩阵）",
        audit["budget_runs"],
    )
    _append_macro_table(lines, "500k checkpoint 宏平均成功率", audit["runs"], STEPS)
    _append_resource_table(lines, "500k 训练资源与吞吐", audit["runs"])
    _append_selection_table(
        lines, "500k 最佳 checkpoint 与 endpoint", audit["selections"]
    )
    _append_individual_seed_table(
        lines, "500k selected/endpoint 逐 seed 结果", audit["selections"]
    )
    _append_pareto_table(
        lines,
        "PointMaze 500k 性能–成本 Pareto",
        audit["runs"],
        audit["selections"],
        included_configs={
            "pointmaze": audit["formal_configs"]["pointmaze"],
            "antmaze": [],
        },
    )
    if "scaling_comparisons" in audit:
        _append_scaling_comparison_table(
            lines,
            "PointMaze 500k 相邻 scaling 差值",
            audit["scaling_comparisons"]["pointmaze_500k"],
        )
    _append_macro_table(
        lines,
        "1M checkpoint 宏平均成功率",
        audit["budget_runs"],
        MILLION_STEPS,
    )
    _append_depth_recipe_table(
        lines,
        audit.get(
            "depth_recipe_audits",
            {"pointmaze": {}, "antmaze": {}},
        ),
    )
    _append_width_recipe_table(
        lines,
        audit.get(
            "width_recipe_audits",
            {"pointmaze": {}, "antmaze": {}},
        ),
    )
    _append_next_depth_gate_table(
        lines, audit.get("next_depth_range_gates", {})
    )
    _append_next_width_gate_table(
        lines, audit.get("next_width_range_gates", {})
    )
    _append_axis_early_stop_table(
        lines, audit.get("axis_early_stop_gates", {})
    )
    _append_conditional_range_table(
        lines, audit.get("conditional_range_requirements", {})
    )
    _append_resource_table(
        lines, "1M 训练资源与吞吐", audit["budget_runs"]
    )
    _append_selection_table(
        lines,
        "1M 最佳 checkpoint 与 endpoint",
        audit["budget_selections"],
    )
    _append_individual_seed_table(
        lines,
        "1M selected/endpoint 逐 seed 结果",
        audit["budget_selections"],
    )
    _append_pareto_table(
        lines,
        "1M 性能–成本 Pareto",
        audit["budget_runs"],
        audit["budget_selections"],
    )
    if "scaling_comparisons" in audit:
        _append_scaling_comparison_table(
            lines,
            "AntMaze 1M 相邻 scaling 差值",
            audit["scaling_comparisons"]["antmaze_1m"],
        )
    _append_budget_extension_recommendations(
        lines, audit["budget_extension_recommendations"]
    )
    _append_per_variant_tables(
        lines,
        "500k 逐环境 checkpoint 结果",
        audit["runs"],
        audit["families"],
        STEPS,
    )
    _append_per_variant_tables(
        lines,
        "1M 逐环境 checkpoint 结果",
        audit["budget_runs"],
        audit["families"],
        MILLION_STEPS,
    )
    if any(audit["extension_runs"].values()):
        _append_completion_table(
            lines, "2M 条件延长完成状态", audit["extension_runs"]
        )
        _append_macro_table(
            lines,
            "2M 条件延长 checkpoint 宏平均成功率",
            audit["extension_runs"],
            TWO_MILLION_STEPS,
        )
        _append_resource_table(
            lines, "2M 条件延长训练资源与吞吐", audit["extension_runs"]
        )
        _append_selection_table(
            lines,
            "2M 条件延长最佳 checkpoint 与 endpoint",
            audit["extension_selections"],
        )
        _append_individual_seed_table(
            lines,
            "2M 条件延长 selected/endpoint 逐 seed 结果",
            audit["extension_selections"],
        )
        _append_pareto_table(
            lines,
            "2M 条件延长性能–成本 Pareto",
            audit["extension_runs"],
            audit["extension_selections"],
        )
        _append_per_variant_tables(
            lines,
            "2M 条件延长逐环境 checkpoint 结果",
            audit["extension_runs"],
            audit["families"],
            TWO_MILLION_STEPS,
        )
    lines.extend([
        "",
        "## 审计与原始产物",
        "",
        "机器可读审计位于 `reports/bc_scaling_20260723_checkpoint_audit.json`。每个 raw run"
        " 位于 `baseline_runs/`，其中包含 resolved config、dataset manifest、"
        "model metadata、所有 checkpoint、evaluation JSONL 和最终模型。",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成中文报告：{OUTPUT}")


if __name__ == "__main__":
    main()
