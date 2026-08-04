from __future__ import annotations

import resource
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

import crossmaze
from crossmaze.eval_position import (
    build_seed_map_eval_position_config,
    eval_position_selection_policy,
    resolve_eval_position_mode,
    resolve_seed_map_eval_position_mode,
    select_seed_map_eval_position,
)
from crossmaze.seed_map import generate_seed_map, seed_map_hash
from baselines.artifacts import append_jsonl
from baselines.data.observation import (
    BaselineObservationWrapper,
    GoalConditionedObservationWrapper,
)
from utils.seed_map_eval import SeedMapEvalSelection, seed_map_eval_target


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _mean_std_or_none(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def _actual_start_goal_record(
    env: BaselineObservationWrapper,
    *,
    env_family: str,
    variant: str,
    env_config: dict,
    base_seed: int,
    seed_map_target: dict | None = None,
) -> dict:
    state = env.last_crossmaze_state
    if not isinstance(state, Mapping):
        raise ValueError(
            f"CrossMaze reset observation for {variant!r} is missing structured state"
        )
    required = {"position_cell", "goal_cell", "position_xy", "goal_xy"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(
            f"CrossMaze reset observation for {variant!r} is missing: {missing}"
        )
    if seed_map_target is None:
        sampling_mode = resolve_eval_position_mode(env_family, env_config)
        selection_policy = eval_position_selection_policy(
            env_family, variant, config=env_config, seed=base_seed
        )
    else:
        sampling_mode = resolve_seed_map_eval_position_mode(env_config)
        selection_policy = (
            "seeded_weighted_hard_sample_permutation_cycle"
            if sampling_mode == "hard-sample"
            else "env_default_random"
        )
    return {
        "sampling_mode": sampling_mode,
        "selection_policy": selection_policy,
        "start_cell": [int(value) for value in state["position_cell"]],
        "goal_cell": [int(value) for value in state["goal_cell"]],
        "start_xy": [float(value) for value in state["position_xy"]],
        "goal_xy": [float(value) for value in state["goal_xy"]],
    }


def _make_evaluation_env(
    *,
    env_family: str,
    variant: str,
    reward_type: str,
    env_config: dict,
    observation_config: dict,
    goal_conditioned: bool,
    seed_map_target: dict | None = None,
):
    env_config = dict(env_config)
    if goal_conditioned:
        env_kwargs = dict(env_config.get("env_kwargs") or {})
        for field in ("continuing_task", "reset_target"):
            if field in env_config and env_config[field] is not False:
                raise ValueError(
                    f"CRL/HIQL single-goal evaluation requires {field}=false"
                )
            if field in env_kwargs and env_kwargs[field] is not False:
                raise ValueError(
                    "CRL/HIQL single-goal evaluation requires "
                    f"env_kwargs.{field}=false"
                )
            env_config.pop(field, None)
            env_kwargs[field] = False
        env_config["env_kwargs"] = env_kwargs
    wrapper_class = (
        GoalConditionedObservationWrapper
        if goal_conditioned
        else BaselineObservationWrapper
    )
    if seed_map_target is None:
        env = crossmaze.make(
            env_family, variant, mode="eval", config=env_config
        )
    else:
        seed_map_config = dict(env_config)
        seed_map_config["reward_type"] = seed_map_target["reward_type"]
        seed_map_config["max_episode_steps"] = seed_map_target[
            "max_episode_steps"
        ]
        env = crossmaze.make_seed_map(
            env_family,
            seed_map_target["map_seed"],
            seed_map_spec=seed_map_target["seed_map_spec"],
            config=seed_map_config,
        )
    return wrapper_class(
        env,
        env_family=env_family,
        observation_config=observation_config,
    )


def _batch_policy_observation(observations: list):
    """Stack independent rollout observations without changing their ordering."""

    first = observations[0]
    if isinstance(first, Mapping):
        keys = set(first)
        if any(not isinstance(item, Mapping) or set(item) != keys for item in observations):
            raise ValueError("Goal-conditioned rollout observations have inconsistent keys")
        return {
            key: np.stack([np.asarray(item[key]) for item in observations], axis=0)
            for key in first
        }
    return np.stack([np.asarray(item) for item in observations], axis=0)


def _episode_record(state: dict) -> dict:
    first_success_step = state["first_success_step"]
    return {
        "episode_index": int(state["episode_index"]),
        "seed": int(state["seed"]),
        "start_goal": state["start_goal"],
        "success": bool(state["success"]),
        "first_success_step": (
            int(first_success_step) if first_success_step is not None else None
        ),
        "return": float(state["return"]),
        "length": int(state["length"]),
        "terminated": bool(state["terminated"]),
        "truncated": bool(state["truncated"]),
    }


def _evaluate_variant_batched(
    algo,
    *,
    env_family: str,
    variant: str,
    reward_type: str,
    evaluation_config: dict,
    observation_config: dict,
    goal_conditioned: bool,
    seed_map_target: dict | None = None,
) -> list[dict]:
    """Roll out one variant in independent env slots with batched policy calls.

    Only observations reach the policy in a batch.  Resets, environment steps,
    seed assignment, action validation, and per-episode output records remain
    exactly episode-local, which makes this a throughput change rather than an
    evaluation-protocol change.
    """

    base_seed = evaluation_config["seed"]
    batch_size = min(
        evaluation_config.get("rollout_batch_size", 1),
        evaluation_config["num_episodes"],
    )
    env_config = dict(evaluation_config["env_config"])
    env_config["reward_type"] = reward_type
    env_config["seed"] = base_seed
    env_config["wall_sensing_version"] = observation_config["wall_sensing_version"]
    env_config["map_sensing_boundary_risk_threshold"] = observation_config[
        "map_sensing_boundary_risk_threshold"
    ]
    seed_map_position_config = None
    if seed_map_target is not None:
        maze_map = generate_seed_map(
            seed_map_target["map_seed"],
            seed_map_target["seed_map_spec"],
        )
        seed_map_position_config = build_seed_map_eval_position_config(
            env_family,
            seed_map_target["map_seed"],
            maze_map,
            seed=base_seed,
            config=env_config,
        )
    episodes: list[dict | None] = [None] * evaluation_config["num_episodes"]

    for batch_start in range(0, evaluation_config["num_episodes"], batch_size):
        active = []
        try:
            batch_stop = min(batch_start + batch_size, evaluation_config["num_episodes"])
            for episode_index in range(batch_start, batch_stop):
                env = _make_evaluation_env(
                    env_family=env_family,
                    variant=variant,
                    reward_type=reward_type,
                    env_config=env_config,
                    observation_config=observation_config,
                    goal_conditioned=goal_conditioned,
                    seed_map_target=seed_map_target,
                )
                episode_seed = base_seed + episode_index
                reset_options = None
                if seed_map_target is not None:
                    eval_position = select_seed_map_eval_position(
                        env_family,
                        seed_map_target["map_seed"],
                        episode_index=episode_index,
                        seed=base_seed,
                        position_config=seed_map_position_config,
                        config=env_config,
                    )
                    if eval_position is not None:
                        reset_options = {
                            "reset_cell": np.asarray(
                                eval_position["start_cell"], dtype=np.int64
                            ),
                            "goal_cell": np.asarray(
                                eval_position["goal_cell"], dtype=np.int64
                            ),
                        }
                observation, _ = env.reset(
                    seed=episode_seed,
                    options=reset_options,
                )
                active.append(
                    {
                        "env": env,
                        "episode_index": episode_index,
                        "seed": episode_seed,
                        "observation": observation,
                        "start_goal": _actual_start_goal_record(
                            env,
                            env_family=env_family,
                            variant=variant,
                            env_config=env_config,
                            base_seed=base_seed,
                            seed_map_target=seed_map_target,
                        ),
                        "return": 0.0,
                        "length": 0,
                        "success": False,
                        "first_success_step": None,
                        "terminated": False,
                        "truncated": False,
                    }
                )
            while active:
                policy_observation = _batch_policy_observation(
                    [state["observation"] for state in active]
                )
                actions = np.asarray(algo.predict(policy_observation))
                if actions.shape[0] != len(active):
                    raise ValueError(
                        f"Policy returned batch size {actions.shape[0]}, expected {len(active)}"
                    )
                next_active = []
                for state, action in zip(active, actions, strict=True):
                    env = state["env"]
                    action = np.asarray(action)
                    if action.shape != env.action_space.shape:
                        raise ValueError(
                            f"Policy action shape mismatch for {variant!r}: "
                            f"expected {env.action_space.shape}, got {action.shape}"
                        )
                    if not np.all(np.isfinite(action)):
                        raise ValueError(f"Policy produced a non-finite action for {variant!r}")
                    action = np.clip(
                        action, env.action_space.low, env.action_space.high
                    ).astype(env.action_space.dtype, copy=False)
                    observation, reward, terminated, truncated, info = env.step(action)
                    state["observation"] = observation
                    state["return"] += float(reward)
                    state["length"] += 1
                    step_success = bool(info.get("success", False))
                    if step_success and state["first_success_step"] is None:
                        state["first_success_step"] = state["length"]
                    state["success"] = state["success"] or step_success
                    state["terminated"] = bool(terminated)
                    state["truncated"] = bool(truncated)
                    if terminated or truncated:
                        episodes[state["episode_index"]] = _episode_record(state)
                        env.close()
                    else:
                        next_active.append(state)
                active = next_active
        finally:
            for state in active:
                state["env"].close()

    if any(record is None for record in episodes):
        raise RuntimeError(f"Batched rollout did not finish every episode for {variant!r}")
    return [record for record in episodes if record is not None]


def evaluate_rollouts(
    algo,
    *,
    env_family: str,
    variants: list[str],
    reward_types: dict[str, str],
    evaluation_config: dict,
    observation_config: dict,
    goal_conditioned: bool = False,
    seed_map_eval: SeedMapEvalSelection | Mapping | None = None,
) -> dict:
    variant_metrics = {}
    all_successes = []
    all_returns = []
    all_lengths = []
    all_success_steps = []
    base_seed = evaluation_config["seed"]
    if isinstance(seed_map_eval, SeedMapEvalSelection):
        resolved_seed_map_eval = seed_map_eval.to_dict()
    elif isinstance(seed_map_eval, Mapping):
        resolved_seed_map_eval = dict(seed_map_eval)
    else:
        resolved_seed_map_eval = None
    for variant in variants:
        target = (
            seed_map_eval_target(
                {"resolved_seed_map_eval": resolved_seed_map_eval},
                variant,
            )
            if resolved_seed_map_eval is not None
            else None
        )
        episodes = _evaluate_variant_batched(
            algo,
            env_family=env_family,
            variant=variant,
            reward_type=reward_types[variant],
            evaluation_config=evaluation_config,
            observation_config=observation_config,
            goal_conditioned=goal_conditioned,
            seed_map_target=target,
        )
        successes = [float(episode["success"]) for episode in episodes]
        returns = [episode["return"] for episode in episodes]
        lengths = [float(episode["length"]) for episode in episodes]
        success_steps = [
            float(episode["first_success_step"])
            for episode in episodes
            if episode["first_success_step"] is not None
        ]
        success_step_mean, success_step_std = _mean_std_or_none(success_steps)
        unique_start_goals = {
            (
                tuple(episode["start_goal"]["start_cell"]),
                tuple(episode["start_goal"]["goal_cell"]),
            )
            for episode in episodes
        }
        variant_metrics[variant] = {
            "reward_type": reward_types[variant],
            "num_episodes": len(successes),
            "successful_episode_count": int(sum(successes)),
            "success_rate": _mean(successes),
            "first_success_step_mean": success_step_mean,
            "first_success_step_std": success_step_std,
            "return_mean": _mean(returns),
            "return_std": float(np.std(returns)),
            "length_mean": _mean(lengths),
            "unique_start_goal_count": len(unique_start_goals),
            "episodes": episodes,
        }
        if target is not None:
            maze_map = generate_seed_map(
                target["map_seed"],
                target["seed_map_spec"],
            )
            variant_metrics[variant]["seed_map"] = {
                "map_seed": target["map_seed"],
                "seed_map_spec": target["seed_map_spec"],
                "map_hash": seed_map_hash(maze_map),
                "reward_type": target["reward_type"],
                "max_episode_steps": target["max_episode_steps"],
            }
        all_successes.extend(successes)
        all_returns.extend(returns)
        all_lengths.extend(lengths)
        all_success_steps.extend(success_steps)
    success_step_mean, success_step_std = _mean_std_or_none(all_success_steps)
    return {
        "aggregate": {
            "num_episodes": len(all_successes),
            "successful_episode_count": int(sum(all_successes)),
            "success_rate": _mean(all_successes),
            "first_success_step_mean": success_step_mean,
            "first_success_step_std": success_step_std,
            "return_mean": _mean(all_returns),
            "return_std": float(np.std(all_returns)),
            "length_mean": _mean(all_lengths),
        },
        "variants": variant_metrics,
    }


def evaluate_validation(algo, validation_buffer, *, algorithm: str) -> dict:
    from d3rlpy.metrics import ContinuousActionDiffEvaluator, TDErrorEvaluator

    metrics = {
        "action_mse_sum": ContinuousActionDiffEvaluator()(algo, validation_buffer)
    }
    if algorithm in {"td3_bc", "iql", "rebrac"}:
        metrics["td_error"] = TDErrorEvaluator()(algo, validation_buffer)
    return metrics


def _policy_parameter_global_norm(algo) -> float:
    import torch

    assert algo.impl is not None
    squared_norms = [
        parameter.detach().float().pow(2).sum()
        for parameter in algo.impl.policy.parameters()
        if parameter.requires_grad
    ]
    if not squared_norms:
        raise RuntimeError("BC policy has no trainable parameters")
    return float(torch.stack(squared_norms).sum().sqrt().item())


def _process_peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB. The baseline environments are Linux-only.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


class BaselineEpochCallback:
    def __init__(
        self,
        *,
        config: dict,
        selections,
        validation_buffer,
        action_probe: np.ndarray | None,
        run_dir: Path,
        total_epochs: int,
        epoch_offset: int = 0,
        step_offset: int = 0,
    ):
        self._config = config
        self._selections = selections
        self._validation_buffer = validation_buffer
        self._action_probe = action_probe
        self._run_dir = run_dir
        self._total_epochs = total_epochs
        self._epoch_offset = int(epoch_offset)
        self._step_offset = int(step_offset)
        self.history: list[dict] = []
        self.diagnostics_history: list[dict] = []
        self._start_time = time.perf_counter()
        self._last_epoch_time = self._start_time

    def _record_training_diagnostics(self, algo, epoch: int, total_step: int) -> dict | None:
        if self._action_probe is None:
            return None
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        actions = np.asarray(algo.predict(self._action_probe), dtype=np.float64)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        if not np.all(np.isfinite(actions)):
            raise FloatingPointError("BC diagnostic action probe contains NaN or Inf")
        consume = getattr(algo, "consume_training_diagnostics", None)
        if consume is None:
            raise RuntimeError("Enabled training diagnostics require InstrumentedBC")
        gradient_metrics = consume()
        batch_size = self._config["algorithm_config"]["batch_size"]
        epoch_seconds = now - self._last_epoch_time
        elapsed_seconds = now - self._start_time
        record = {
            "epoch": int(epoch),
            "step": int(total_step),
            "processed_examples": int(total_step * batch_size),
            "epoch_wall_time_seconds": float(epoch_seconds),
            "wall_time_seconds": float(elapsed_seconds),
            "updates_per_second": float(
                self._config["n_steps_per_epoch"] / epoch_seconds
            ),
            "examples_per_second": float(
                self._config["n_steps_per_epoch"] * batch_size / epoch_seconds
            ),
            "parameter_global_norm": _policy_parameter_global_norm(algo),
            "action_probe_count": int(len(actions)),
            "action_mean": [float(value) for value in actions.mean(axis=0)],
            "action_std": [float(value) for value in actions.std(axis=0)],
            "action_min": [float(value) for value in actions.min(axis=0)],
            "action_max": [float(value) for value in actions.max(axis=0)],
            "action_abs_mean": float(np.abs(actions).mean()),
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
            **gradient_metrics,
        }
        self.diagnostics_history.append(record)
        append_jsonl(self._run_dir / "training_diagnostics.jsonl", record)
        return record

    def __call__(self, algo, epoch: int, total_step: int) -> None:
        from baselines.trainer_state import save_trainer_state

        local_epoch = epoch
        epoch += self._epoch_offset
        total_step += self._step_offset
        diagnostic_record = self._record_training_diagnostics(algo, epoch, total_step)
        final_epoch = local_epoch == self._total_epochs
        checkpoint_saved = (
            epoch % self._config["save_interval_epochs"] == 0 or final_epoch
        )
        if checkpoint_saved:
            algo.save(self._run_dir / "checkpoints" / f"step_{total_step}.d3")

        evaluation = self._config["evaluation"]
        if not evaluation["enabled"]:
            if checkpoint_saved:
                save_trainer_state(
                    self._run_dir
                    / "checkpoints"
                    / f"trainer_state_step_{total_step}.pt",
                    epoch=epoch,
                    step=total_step,
                )
            self._last_epoch_time = time.perf_counter()
            return
        if epoch % evaluation["every_epochs"] != 0 and not final_epoch:
            if checkpoint_saved:
                save_trainer_state(
                    self._run_dir
                    / "checkpoints"
                    / f"trainer_state_step_{total_step}.pt",
                    epoch=epoch,
                    step=total_step,
                )
            self._last_epoch_time = time.perf_counter()
            return

        validation = evaluate_validation(
            algo,
            self._validation_buffer,
            algorithm=self._config["algorithm"],
        )
        rollout = evaluate_rollouts(
            algo,
            env_family=self._config["env_family"],
            variants=self._selections.eval.selected_variants,
            reward_types=self._selections.eval_reward_types,
            evaluation_config=evaluation,
            observation_config=self._config["observation"],
            seed_map_eval=self._selections.seed_map_eval,
        )
        result = {
            "epoch": epoch,
            "step": total_step,
            "validation": validation,
            "rollout": rollout,
        }
        if diagnostic_record is not None:
            result["training_diagnostics"] = diagnostic_record
        self.history.append(result)
        append_jsonl(self._run_dir / "evaluation.jsonl", result)
        aggregate = rollout["aggregate"]
        print(
            "[baseline eval] "
            f"epoch={epoch} step={total_step} "
            f"success={aggregate['success_rate']:.4f} "
            f"return={aggregate['return_mean']:.4f} "
            f"length={aggregate['length_mean']:.1f}"
        )
        if self._config["logging"]["wandb"]["enabled"]:
            import wandb

            wandb.log(
                {
                    "baseline_eval/success_rate": aggregate["success_rate"],
                    "baseline_eval/return_mean": aggregate["return_mean"],
                    "baseline_eval/length_mean": aggregate["length_mean"],
                    **{
                        f"baseline_validation/{key}": value
                        for key, value in validation.items()
                    },
                },
                step=total_step,
            )
        if checkpoint_saved:
            # Capture after validation/rollout because evaluators can consume
            # RNG state even though they do not update policy parameters.
            save_trainer_state(
                self._run_dir
                / "checkpoints"
                / f"trainer_state_step_{total_step}.pt",
                epoch=epoch,
                step=total_step,
            )
        self._last_epoch_time = time.perf_counter()
