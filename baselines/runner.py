from __future__ import annotations

import importlib.metadata
import json
import random
import re
from pathlib import Path

import d3rlpy
import numpy as np
import torch
import yaml
from d3rlpy.logging import (
    CombineAdapterFactory,
    FileAdapterFactory,
    WanDBAdapterFactory,
)

from baselines.algorithms import create_algorithm, resume_algorithm
from baselines.algorithms.scaling_mlp import ScalingMLPEncoder
from baselines.artifacts import create_run_dir, write_json, write_yaml
from baselines.data.loader import prepare_datasets
from baselines.evaluation import BaselineEpochCallback
from baselines.registry import resolve_baseline_selections
from baselines.trainer_state import load_trainer_state, restore_trainer_state


EXPECTED_RUNTIME_VERSIONS = {
    "d3rlpy": "2.8.1",
    "torch": "2.10.0",
    "numpy": "2.2.6",
    "gymnasium": "1.2.3",
    "gymnasium-robotics": "1.4.2",
    "minari": "0.5.3",
}

_RESUME_COMPATIBILITY_FIELDS = (
    "algorithm",
    "env_family",
    "train_mode",
    "train_variants",
    "eval_mode",
    "eval_variants",
    "reward_type",
    "allow_mixed_reward_types",
    "local_dataset_root",
    "episode_keep_num",
    "episode_keep_per_variant",
    "balance_variant_episode_count",
    "sampling_seed",
    "train_data_ratio",
    "seed",
    "observation",
    "network",
    "algorithm_config",
    "evaluation",
    "training_diagnostics",
    "n_steps_per_epoch",
    "save_interval_epochs",
)


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


def model_metadata(algo, config: dict) -> dict | None:
    """Return instantiated-model facts that cannot be inferred from YAML alone."""

    if config["algorithm"] != "mlp_bc" or algo.impl is None:
        return None
    policy = algo.impl.policy
    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    metadata = {
        "policy_class": type(policy).__name__,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in trainable_parameters)
        ),
        "trainable_parameter_tensor_count": len(trainable_parameters),
        "network_config": config["network"],
    }
    encoder = policy._encoder
    if isinstance(encoder, ScalingMLPEncoder):
        structure = encoder.structure()
        structure["policy_action_head_dense_count"] = 1
        structure["total_dense_count"] = structure["encoder_dense_count"] + 1
        metadata["scaling_encoder"] = structure
    return metadata


def _build_action_probe(episodes, *, size: int) -> np.ndarray:
    """Build a fixed, compact probe from training transitions only."""

    chunks = []
    for episode in episodes:
        observations = np.asarray(episode.observations[:-1], dtype=np.float32)
        if not len(observations):
            continue
        indices = np.linspace(
            0, len(observations) - 1, num=min(8, len(observations)), dtype=int
        )
        chunks.append(observations[indices])
    if not chunks:
        raise ValueError("Cannot build an action probe from zero transitions")
    probe = np.concatenate(chunks, axis=0)
    if len(probe) > size:
        probe = probe[np.linspace(0, len(probe) - 1, num=size, dtype=int)]
    return np.ascontiguousarray(probe, dtype=np.float32)


def _load_resume_context(config: dict) -> dict | None:
    resume = config["resume"]
    if resume["checkpoint"] is None:
        return None
    checkpoint = Path(resume["checkpoint"]).expanduser().resolve()
    trainer_state_path = Path(resume["trainer_state"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    if not trainer_state_path.is_file():
        raise FileNotFoundError(
            f"Resume trainer state does not exist: {trainer_state_path}"
        )
    match = re.fullmatch(r"step_(\d+)\.d3", checkpoint.name)
    if match is None:
        raise ValueError(
            "Resume checkpoint filename must be step_<updates>.d3, got "
            f"{checkpoint.name!r}"
        )
    checkpoint_step = int(match.group(1))
    trainer_state = load_trainer_state(trainer_state_path)
    if trainer_state["step"] != checkpoint_step:
        raise ValueError(
            "Resume checkpoint/trainer-state step mismatch: "
            f"{checkpoint_step} != {trainer_state['step']}"
        )
    if config["n_steps"] <= checkpoint_step:
        raise ValueError(
            f"Target n_steps={config['n_steps']} must exceed resume step "
            f"{checkpoint_step}"
        )
    segment_steps = config["n_steps"] - checkpoint_step
    if segment_steps % config["n_steps_per_epoch"] != 0:
        raise ValueError(
            "The updates after resume must be divisible by n_steps_per_epoch"
        )
    if checkpoint_step % config["n_steps_per_epoch"] != 0:
        raise ValueError("Resume step must align with n_steps_per_epoch")

    source_run_dir = checkpoint.parent.parent
    source_config_path = source_run_dir / "config.yaml"
    source_summary_path = source_run_dir / "summary.json"
    if not source_config_path.is_file() or not source_summary_path.is_file():
        raise FileNotFoundError(
            "Resume checkpoint must belong to a completed baseline run with "
            "config.yaml and summary.json"
        )
    source_config = yaml.safe_load(
        source_config_path.read_text(encoding="utf-8")
    )
    mismatches = []
    for field in _RESUME_COMPATIBILITY_FIELDS:
        source_value = source_config.get(field)
        current_value = config.get(field)
        if field == "observation":
            source_value = dict(source_value or {})
            current_value = dict(current_value or {})
            # Dynamic maps were added as an independent opt-in feature. Older
            # d3rlpy runs that omit the key are schema-equivalent to false.
            source_value.setdefault("include_dynamic_map", False)
            current_value.setdefault("include_dynamic_map", False)
        if source_value != current_value:
            mismatches.append(field)
    if mismatches:
        raise ValueError(
            "Resume configuration differs from the source run for fields: "
            + ", ".join(mismatches)
        )
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary["n_steps"] < checkpoint_step:
        raise ValueError("Resume step exceeds source summary training budget")
    return {
        "checkpoint": checkpoint,
        "trainer_state_path": trainer_state_path,
        "trainer_state": trainer_state,
        "step": checkpoint_step,
        "epoch": checkpoint_step // config["n_steps_per_epoch"],
        "segment_steps": segment_steps,
        "source_run_dir": source_run_dir,
        "source_summary": source_summary,
    }


def _prefix_records(records: list[dict], *, step: int) -> list[dict]:
    return [record for record in records if int(record["step"]) <= step]


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
    resume_context = _load_resume_context(config)
    if resume_context is None:
        algo = create_algorithm(config)
        step_offset = 0
        epoch_offset = 0
        segment_steps = config["n_steps"]
        prefix_training_history = []
        prefix_diagnostics = []
        prefix_evaluations = []
    else:
        step_offset = resume_context["step"]
        epoch_offset = resume_context["epoch"]
        segment_steps = resume_context["segment_steps"]
        algo = resume_algorithm(
            config,
            prepared.train_buffer,
            str(resume_context["checkpoint"]),
            resume_step=step_offset,
        )
        restore_trainer_state(resume_context["trainer_state"])
        source_summary = resume_context["source_summary"]
        prefix_training_history = [
            record
            for record in source_summary["training_history"]
            if int(record["epoch"]) <= epoch_offset
        ]
        prefix_diagnostics = _prefix_records(
            source_summary["training_diagnostics"], step=step_offset
        )
        prefix_evaluations = _prefix_records(
            source_summary["evaluation_history"], step=step_offset
        )
    segment_epochs = segment_steps // config["n_steps_per_epoch"]
    target_epochs = config["n_steps"] // config["n_steps_per_epoch"]
    epoch_callback = BaselineEpochCallback(
        config=config,
        selections=selections,
        validation_buffer=prepared.validation_buffer,
        action_probe=(
            _build_action_probe(
                prepared.train_episodes,
                size=config["training_diagnostics"]["action_probe_size"],
            )
            if config["training_diagnostics"]["enabled"]
            else None
        ),
        run_dir=run_dir,
        total_epochs=segment_epochs,
        epoch_offset=epoch_offset,
        step_offset=step_offset,
    )
    print(
        f"[baseline] training {config['algorithm']} from step {step_offset} "
        f"to {config['n_steps']} ({segment_epochs} segment epochs x "
        f"{config['n_steps_per_epoch']} updates)"
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    training_history = algo.fit(
        prepared.train_buffer,
        n_steps=segment_steps,
        n_steps_per_epoch=config["n_steps_per_epoch"],
        experiment_name="training",
        with_timestamp=False,
        logger_adapter=_logger_factory(config, run_dir),
        show_progress=config["show_progress"],
        save_interval=segment_epochs + 1,
        epoch_callback=epoch_callback,
    )
    algo.save(run_dir / "model.d3")
    instantiated_model = model_metadata(algo, config)
    if instantiated_model is not None:
        write_json(run_dir / "model_metadata.json", instantiated_model)
    segment_training_history = [
        {"epoch": int(epoch) + epoch_offset, "metrics": metrics}
        for epoch, metrics in training_history
    ]
    combined_diagnostics = prefix_diagnostics + epoch_callback.diagnostics_history
    combined_evaluations = prefix_evaluations + epoch_callback.history
    checkpoint_artifacts = {}
    for evaluation_record in combined_evaluations:
        step = int(evaluation_record["step"])
        artifact_run_dir = (
            resume_context["source_run_dir"]
            if resume_context is not None and step <= step_offset
            else run_dir
        )
        checkpoint_path = (
            artifact_run_dir / "checkpoints" / f"step_{step}.d3"
        )
        trainer_state_path = (
            artifact_run_dir
            / "checkpoints"
            / f"trainer_state_step_{step}.pt"
        )
        checkpoint_artifacts[str(step)] = {
            "checkpoint": str(checkpoint_path),
            "trainer_state": (
                str(trainer_state_path)
                if trainer_state_path.is_file()
                else None
            ),
        }
    summary = {
        "experiment_id": experiment_id,
        "algorithm": config["algorithm"],
        "env_family": config["env_family"],
        "train_variants": selections.train.selected_variants,
        "eval_variants": selections.eval.selected_variants,
        "n_steps": config["n_steps"],
        "segment_n_steps": segment_steps,
        "n_steps_per_epoch": config["n_steps_per_epoch"],
        "epochs": target_epochs,
        "segment_epochs": segment_epochs,
        "resume": (
            {
                "checkpoint": str(resume_context["checkpoint"]),
                "trainer_state": str(resume_context["trainer_state_path"]),
                "source_run_dir": str(resume_context["source_run_dir"]),
                "step": step_offset,
                "epoch": epoch_offset,
                "rng_state_restored": True,
            }
            if resume_context is not None
            else None
        ),
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
        "training_history": prefix_training_history + segment_training_history,
        "training_diagnostics": combined_diagnostics,
        "evaluation_history": combined_evaluations,
        "checkpoint_artifacts": checkpoint_artifacts,
        "final_evaluation": (
            combined_evaluations[-1] if combined_evaluations else None
        ),
        "model_path": str(run_dir / "model.d3"),
        "model": instantiated_model,
    }
    write_json(run_dir / "summary.json", summary)
    print(f"[baseline] run complete: {run_dir}")
    return run_dir
