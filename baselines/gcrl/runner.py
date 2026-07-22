from __future__ import annotations

import importlib.metadata
import random
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from baselines.artifacts import append_jsonl, create_run_dir, write_json, write_yaml
from baselines.evaluation import evaluate_rollouts
from baselines.gcrl.agents import create_agent
from baselines.gcrl.data import (
    GCRLNormalizer,
    PreparedGCRLDatasets,
    prepare_gcrl_datasets,
)
from baselines.registry import resolve_baseline_selections


EXPECTED_RUNTIME_VERSIONS = {
    "jax": "0.6.2",
    "jaxlib": "0.6.2",
    "flax": "0.10.7",
    "optax": "0.2.5",
    "numpy": "2.2.6",
    "gymnasium": "1.2.3",
    "gymnasium-robotics": "1.4.2",
    "minari": "0.5.3",
    "mujoco": "3.6.0",
    "h5py": "3.16.0",
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
            "GCRL dependency versions do not match baselines/environment.gcrl.yaml:\n- "
            + "\n- ".join(mismatches)
            + "\nRun: bash baselines/setup_gcrl_env.sh"
        )
    return versions


def _select_device(configured):
    if configured is False or configured == "cpu":
        platform = "cpu"
        index = 0
    elif configured is True:
        platform = "gpu"
        index = 0
    elif isinstance(configured, int):
        platform = "gpu"
        index = configured
    elif isinstance(configured, str):
        normalized = configured.strip().lower()
        if normalized.startswith(("cuda:", "gpu:")):
            platform = "gpu"
            index = int(normalized.split(":", 1)[1])
        elif normalized in {"cuda", "gpu"}:
            platform = "gpu"
            index = 0
        else:
            raise ValueError(
                "CRL/HIQL device must be cpu, cuda, cuda:<index>, gpu:<index>, "
                "a GPU index, or a bool"
            )
    else:
        raise ValueError(f"Unsupported GCRL device: {configured!r}")
    devices = jax.devices(platform)
    if index < 0 or index >= len(devices):
        raise ValueError(
            f"Requested {platform} device {index}, but available devices are {devices}"
        )
    return devices[index]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _agent_config(config: dict) -> dict:
    return {
        **config["algorithm_config"],
        "hidden_dims": tuple(config["network"]["hidden_units"]),
        "activation": config["network"]["activation"],
        "layer_norm": config["network"]["use_layer_norm"],
    }


def _jax_batch(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value) for key, value in batch.items()}


def _python_metrics(metrics: dict) -> dict[str, float]:
    converted = {
        key: float(np.asarray(jax.device_get(value)))
        for key, value in metrics.items()
    }
    non_finite = [key for key, value in converted.items() if not np.isfinite(value)]
    if non_finite:
        raise FloatingPointError(f"Non-finite GCRL metrics: {non_finite}")
    return converted


def _mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = records[0].keys()
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def _save_agent(path: Path, agent) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.to_bytes(agent))


class GCRLPolicy:
    def __init__(self, agent, normalizer: GCRLNormalizer):
        self.agent = agent
        self.normalizer = normalizer

    def predict(self, observations):
        if not isinstance(observations, dict) or set(observations) != {"state", "goal"}:
            raise ValueError("GCRL policy expects batched state and goal observations")
        states = self.normalizer.normalize_states(observations["state"])
        goals = self.normalizer.normalize_goals(observations["goal"])
        actions = self.agent.predict_actions(jnp.asarray(states), jnp.asarray(goals))
        return np.asarray(jax.device_get(actions), dtype=np.float32)


def _validation_metrics(
    agent,
    prepared: PreparedGCRLDatasets,
    normalizer: GCRLNormalizer,
    *,
    algorithm: str,
    agent_config: dict,
) -> dict[str, float]:
    raw_batch = prepared.validation.sample(
        agent_config["batch_size"], algorithm=algorithm, config=agent_config
    )
    batch = _jax_batch(normalizer.normalize_batch(raw_batch))
    if algorithm == "crl":
        loss, info = agent.total_loss(batch, agent.network.params, agent.rng)
    else:
        loss, info = agent.total_loss(batch, agent.network.params)
    return _python_metrics({"total_loss": loss, **info})


def train_gcrl_baseline(config: dict) -> Path:
    if config["logging"]["wandb"]["enabled"]:
        raise ValueError("CRL/HIQL smoke backend currently requires logging.wandb.enabled=false")
    versions = runtime_versions()
    selections = resolve_baseline_selections(config)
    experiment_id, run_dir = create_run_dir(
        config, selections.train.selection_tag
    )
    device = _select_device(config["device"])
    resolved_config = dict(config)
    resolved_config["experiment_id"] = experiment_id
    resolved_config["resolved_train_variants"] = selections.train.selected_variants
    resolved_config["resolved_eval_variants"] = selections.eval.selected_variants
    resolved_config["resolved_train_reward_types"] = selections.train_reward_types
    resolved_config["resolved_eval_reward_types"] = selections.eval_reward_types
    resolved_config["runtime_versions"] = versions
    resolved_config["jax_device"] = str(device)
    write_yaml(run_dir / "config.yaml", resolved_config)

    print(
        f"[baseline:gcrl] loading {config['env_family']} episodes for "
        f"{selections.train.selected_variants}"
    )
    prepared = prepare_gcrl_datasets(
        config,
        selections.train.selected_variants,
        selections.train_reward_types,
    )
    write_json(run_dir / "dataset_manifest.json", prepared.manifest)
    for warning in prepared.manifest["warnings"]:
        print(f"[baseline warning] {warning}")

    algorithm_config = _agent_config(config)
    if algorithm_config["normalize_observations"]:
        normalizer = GCRLNormalizer.fit(prepared.train)
    else:
        normalizer = GCRLNormalizer.identity(
            prepared.train.state_dim, prepared.train.goal_dim
        )
    write_json(run_dir / "normalizer.json", normalizer.to_dict())

    _seed_everything(config["seed"])
    example_batch = normalizer.normalize_batch(
        prepared.train.sample(
            2,
            algorithm=config["algorithm"],
            config=algorithm_config,
        )
    )
    total_epochs = config["n_steps"] // config["n_steps_per_epoch"]
    training_history = []
    evaluation_history = []
    with jax.default_device(device):
        agent = create_agent(
            config["algorithm"],
            seed=config["seed"],
            example_observations=jnp.asarray(example_batch["observations"]),
            example_goals=jnp.asarray(example_batch["value_goals"]),
            example_actions=jnp.asarray(example_batch["actions"]),
            config=algorithm_config,
        )
        print(
            f"[baseline:gcrl] training {config['algorithm']} for {config['n_steps']} "
            f"updates ({total_epochs} epochs x {config['n_steps_per_epoch']} updates) "
            f"on {device}"
        )
        total_step = 0
        for epoch in range(1, total_epochs + 1):
            epoch_metrics = []
            for _ in range(config["n_steps_per_epoch"]):
                raw_batch = prepared.train.sample(
                    algorithm_config["batch_size"],
                    algorithm=config["algorithm"],
                    config=algorithm_config,
                )
                batch = _jax_batch(normalizer.normalize_batch(raw_batch))
                agent, info = agent.update(batch)
                epoch_metrics.append(_python_metrics(info))
                total_step += 1
            metrics = _mean_metrics(epoch_metrics)
            training_record = {
                "epoch": epoch,
                "step": total_step,
                "metrics": metrics,
            }
            training_history.append(training_record)
            append_jsonl(run_dir / "training.jsonl", training_record)
            if config["show_progress"]:
                print(
                    f"[baseline:gcrl] epoch={epoch}/{total_epochs} "
                    f"step={total_step} loss metrics recorded"
                )

            final_epoch = epoch == total_epochs
            if epoch % config["save_interval_epochs"] == 0 or final_epoch:
                _save_agent(
                    run_dir / "checkpoints" / f"step_{total_step}.msgpack", agent
                )

            evaluation = config["evaluation"]
            should_evaluate = evaluation["enabled"] and (
                epoch % evaluation["every_epochs"] == 0 or final_epoch
            )
            if should_evaluate:
                validation = _validation_metrics(
                    agent,
                    prepared,
                    normalizer,
                    algorithm=config["algorithm"],
                    agent_config=algorithm_config,
                )
                rollout = evaluate_rollouts(
                    GCRLPolicy(agent, normalizer),
                    env_family=config["env_family"],
                    variants=selections.eval.selected_variants,
                    reward_types=selections.eval_reward_types,
                    evaluation_config=evaluation,
                    observation_config=config["observation"],
                    goal_conditioned=True,
                )
                evaluation_record = {
                    "epoch": epoch,
                    "step": total_step,
                    "validation": validation,
                    "rollout": rollout,
                }
                evaluation_history.append(evaluation_record)
                append_jsonl(run_dir / "evaluation.jsonl", evaluation_record)
                aggregate = rollout["aggregate"]
                print(
                    "[baseline:gcrl eval] "
                    f"epoch={epoch} step={total_step} "
                    f"success={aggregate['success_rate']:.4f} "
                    f"return={aggregate['return_mean']:.4f} "
                    f"length={aggregate['length_mean']:.1f}"
                )

        _save_agent(run_dir / "model.msgpack", agent)

    summary = {
        "experiment_id": experiment_id,
        "algorithm": config["algorithm"],
        "backend": "jax",
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
        "training_history": training_history,
        "evaluation_history": evaluation_history,
        "final_evaluation": evaluation_history[-1] if evaluation_history else None,
        "model_path": str(run_dir / "model.msgpack"),
        "normalizer_path": str(run_dir / "normalizer.json"),
    }
    write_json(run_dir / "summary.json", summary)
    print(f"[baseline:gcrl] run complete: {run_dir}")
    return run_dir
