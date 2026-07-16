from __future__ import annotations

import importlib.metadata
import random
from pathlib import Path

import d3rlpy
import numpy as np
import torch
from d3rlpy.logging import (
    CombineAdapterFactory,
    FileAdapterFactory,
    WanDBAdapterFactory,
)

from baselines.algorithms import create_algorithm
from baselines.artifacts import create_run_dir, write_json, write_yaml
from baselines.data.loader import prepare_datasets
from baselines.evaluation import BaselineEpochCallback
from baselines.registry import resolve_baseline_selections


EXPECTED_RUNTIME_VERSIONS = {
    "d3rlpy": "2.8.1",
    "torch": "2.10.0",
    "numpy": "2.2.6",
    "gymnasium": "1.2.3",
    "gymnasium-robotics": "1.4.2",
    "minari": "0.5.3",
}


def runtime_versions() -> dict[str, str]:
    versions = {}
    mismatches = []
    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "not-installed"
        versions[package] = actual
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, found {actual}")
    if mismatches:
        raise RuntimeError(
            "Baseline dependency versions do not match baselines/environment.yaml:\n- "
            + "\n- ".join(mismatches)
            + "\nRun: bash baselines/setup_env.sh"
        )
    return versions


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    d3rlpy.seed(seed)


def _logger_factory(config: dict, run_dir: Path):
    factories = [FileAdapterFactory(root_dir=str(run_dir / "d3rlpy_logs"))]
    wandb_config = config["logging"]["wandb"]
    if wandb_config["enabled"]:
        factories.append(WanDBAdapterFactory(project=wandb_config["project"]))
    return factories[0] if len(factories) == 1 else CombineAdapterFactory(factories)


def train_baseline(config: dict) -> Path:
    versions = runtime_versions()
    selections = resolve_baseline_selections(config)
    experiment_id, run_dir = create_run_dir(
        config, selections.train.selection_tag
    )
    resolved_config = dict(config)
    resolved_config["experiment_id"] = experiment_id
    resolved_config["resolved_train_variants"] = selections.train.selected_variants
    resolved_config["resolved_eval_variants"] = selections.eval.selected_variants
    resolved_config["resolved_train_reward_types"] = selections.train_reward_types
    resolved_config["resolved_eval_reward_types"] = selections.eval_reward_types
    resolved_config["runtime_versions"] = versions
    write_yaml(run_dir / "config.yaml", resolved_config)

    print(
        f"[baseline] loading {config['env_family']} datasets for "
        f"{selections.train.selected_variants}"
    )
    prepared = prepare_datasets(
        config,
        selections.train.selected_variants,
        selections.train_reward_types,
    )
    write_json(run_dir / "dataset_manifest.json", prepared.manifest)
    for warning in prepared.manifest["warnings"]:
        print(f"[baseline warning] {warning}")

    _seed_everything(config["seed"])
    algo = create_algorithm(config)
    total_epochs = config["n_steps"] // config["n_steps_per_epoch"]
    epoch_callback = BaselineEpochCallback(
        config=config,
        selections=selections,
        validation_buffer=prepared.validation_buffer,
        run_dir=run_dir,
        total_epochs=total_epochs,
    )
    print(
        f"[baseline] training {config['algorithm']} for {config['n_steps']} updates "
        f"({total_epochs} epochs x {config['n_steps_per_epoch']} updates)"
    )
    training_history = algo.fit(
        prepared.train_buffer,
        n_steps=config["n_steps"],
        n_steps_per_epoch=config["n_steps_per_epoch"],
        experiment_name="training",
        with_timestamp=False,
        logger_adapter=_logger_factory(config, run_dir),
        show_progress=config["show_progress"],
        save_interval=total_epochs + 1,
        epoch_callback=epoch_callback,
    )
    algo.save(run_dir / "model.d3")
    summary = {
        "experiment_id": experiment_id,
        "algorithm": config["algorithm"],
        "env_family": config["env_family"],
        "train_variants": selections.train.selected_variants,
        "eval_variants": selections.eval.selected_variants,
        "n_steps": config["n_steps"],
        "n_steps_per_epoch": config["n_steps_per_epoch"],
        "epochs": total_epochs,
        "dataset": {
            "train_episode_count": prepared.manifest["train_episode_count"],
            "validation_episode_count": prepared.manifest[
                "validation_episode_count"
            ],
            "train_transition_count": prepared.manifest[
                "train_transition_count"
            ],
            "validation_transition_count": prepared.manifest[
                "validation_transition_count"
            ],
        },
        "training_history": [
            {"epoch": int(epoch), "metrics": metrics}
            for epoch, metrics in training_history
        ],
        "evaluation_history": epoch_callback.history,
        "final_evaluation": (
            epoch_callback.history[-1] if epoch_callback.history else None
        ),
        "model_path": str(run_dir / "model.d3"),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"[baseline] run complete: {run_dir}")
    return run_dir
