from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "reports/gcrl_multienv_full_sensing_20260723_checkpoint_audit.json"
PREVIOUS_AUDIT = ROOT / "reports/gcrl_multienv_paper_20260722_checkpoint_audit.json"
OUTPUT = Path(__file__).with_name("report.md")
STEPS = (250_000, 500_000, 750_000, 1_000_000)
SPLITS = {
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


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _aggregate_row(family: str, algorithm: str, rows: list[dict]) -> str:
    rendered = [
        f"{_percent(row['train']['success_rate'])} / {_percent(row['held_out_test']['success_rate'])}"
        for row in rows
    ]
    best = max(rows, key=lambda row: row["held_out_test"]["success_rate"])
    return (
        f"| {family.title()} {algorithm.upper()} | "
        + " | ".join(rendered)
        + f" | **{_percent(best['held_out_test']['success_rate'])} ({best['step'] // 1000}k)** |"
    )


def _per_variant_table(family: str, algorithm: str, rows: list[dict]) -> list[str]:
    output = [
        f"### {family.title()} / {algorithm.upper()}",
        "",
        "| Split | Variant | 250k | 500k | 750k | 1m |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    rates_by_step = {row["step"]: row["per_variant"] for row in rows}
    for split, variants in SPLITS[family].items():
        for variant in variants:
            rates = [
                _percent(rates_by_step[step][variant]["success_rate"])
                for step in STEPS
            ]
            output.append(
                f"| {split} | `{variant}` | " + " | ".join(rates) + " |"
            )
    output.append("")
    return output


def _best_held_out(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: row["held_out_test"]["success_rate"])


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    lines = [
        "# Full-sensing multi-environment CRL / HIQL report",
        "",
        "This controlled round enables all numeric CrossMaze context features: "
        "the padded maze map, current/goal grid cells, and four-way wall/risk "
        "state. It retains the paper-protocol seed, update budget, network, "
        "sampler, train/test maps, and evaluation contract from the prior round.",
        "",
        "## Protocol",
        "",
        "| Family | Train maps | Held-out rollout maps | Checkpoint contract |",
        "| --- | --- | --- | --- |",
        "| PointMaze | 16: official `open/umaze/medium/large` + `local-layoutV2-01..12` | 6: `test-layoutV2-01..06` | each map: 30 reset seeds `0..29` (660 episodes/checkpoint) |",
        "| AntMaze | 16: official `umaze/medium-diverse/large-diverse/ultra` + `local-layout-01..12` | 4: `test-layout-01..04` | each map: 30 reset seeds `0..29` (600 episodes/checkpoint) |",
        "",
        "The held-out maps are excluded from the offline sampler and normalizer. "
        "Maps are stored once per variant in the offline GCRL dataset and are "
        "restored in their original state-vector position per sampled batch; this "
        "preserves the online/offline input schema while avoiding repeated map "
        "storage for AntMaze's 65M transitions.",
        "",
        "## Aggregate checkpoint results",
        "",
        "Each cell is `train-map success / held-out-map success`; this report does "
        "not use held-out results to change hyperparameters.",
        "",
        "| Family / algorithm | 250k | 500k | 750k | 1m | Best held-out |",
        "| --- | --- | --- | --- | --- |",
    ]
    for family in ("pointmaze", "antmaze"):
        for algorithm in ("crl", "hiql"):
            rows = audit["runs"][family][algorithm]["checkpoint_evaluations"]
            lines.append(_aggregate_row(family, algorithm, rows))

    if PREVIOUS_AUDIT.is_file():
        previous = json.loads(PREVIOUS_AUDIT.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## Controlled comparison with the prior no-context round",
                "",
                "This is a single-seed descriptive comparison only; the held-out maps "
                "are not used to select a hyperparameter change. `Delta` compares the "
                "best held-out checkpoint success in each full 1M trajectory.",
                "",
                "| Family / algorithm | Previous best held-out | Full-sensing best held-out | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for family in ("pointmaze", "antmaze"):
            for algorithm in ("crl", "hiql"):
                prior_rows = previous["runs"][family][algorithm]["checkpoint_evaluations"]
                current_rows = audit["runs"][family][algorithm]["checkpoint_evaluations"]
                prior = _best_held_out(prior_rows)
                current = _best_held_out(current_rows)
                delta = current["held_out_test"]["success_rate"] - prior["held_out_test"]["success_rate"]
                lines.append(
                    f"| {family.title()} {algorithm.upper()} | "
                    f"{_percent(prior['held_out_test']['success_rate'])} ({prior['step'] // 1000}k) | "
                    f"{_percent(current['held_out_test']['success_rate'])} ({current['step'] // 1000}k) | "
                    f"{delta * 100.0:+.2f} pp |"
                )
        lines.extend(
            [
                "",
                "The feature set is therefore retained as a documented AntMaze-CRL "
                "candidate (the one setting with a positive held-out change), but is "
                "not promoted as a universal default from one seed: PointMaze and "
                "AntMaze HIQL held-out curves do not improve. A future selection should "
                "use replicated seeds or disjoint validation maps before changing a "
                "production/default configuration.",
            ]
        )

    lines.extend(["", "## Per-variant checkpoint success", ""])
    for family in ("pointmaze", "antmaze"):
        for algorithm in ("crl", "hiql"):
            rows = audit["runs"][family][algorithm]["checkpoint_evaluations"]
            lines.extend(_per_variant_table(family, algorithm, rows))

    lines.extend(
        [
            "## Artifacts",
            "",
            "The machine-readable audit is "
            "`reports/gcrl_multienv_full_sensing_20260723_checkpoint_audit.json`. "
            "Raw run directories under `baseline_runs/` retain each resolved config, "
            "dataset manifest, normalizer, four checkpoints, evaluation JSONL, and "
            "final model.",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {OUTPUT}")


if __name__ == "__main__":
    main()
