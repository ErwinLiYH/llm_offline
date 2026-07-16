from __future__ import annotations

from pathlib import Path

import numpy as np
from d3rlpy.metrics import ContinuousActionDiffEvaluator, TDErrorEvaluator

import crossmaze
from baselines.artifacts import append_jsonl
from baselines.data.observation import BaselineObservationWrapper


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def evaluate_rollouts(
    algo,
    *,
    env_family: str,
    variants: list[str],
    reward_types: dict[str, str],
    evaluation_config: dict,
) -> dict:
    variant_metrics = {}
    all_successes = []
    all_returns = []
    all_lengths = []
    base_seed = evaluation_config["seed"]
    for variant in variants:
        env_config = dict(evaluation_config["env_config"])
        env_config["reward_type"] = reward_types[variant]
        env_config["seed"] = base_seed
        env = BaselineObservationWrapper(
            crossmaze.make(
                env_family,
                variant,
                mode="eval",
                config=env_config,
            ),
            env_family=env_family,
        )
        successes = []
        returns = []
        lengths = []
        try:
            for episode_index in range(evaluation_config["num_episodes"]):
                episode_seed = base_seed + episode_index
                observation, _ = env.reset(seed=episode_seed)
                episode_return = 0.0
                episode_length = 0
                episode_success = False
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    action = np.asarray(algo.predict(observation[None])[0])
                    if action.shape != env.action_space.shape:
                        raise ValueError(
                            f"Policy action shape mismatch for {variant!r}: "
                            f"expected {env.action_space.shape}, got {action.shape}"
                        )
                    if not np.all(np.isfinite(action)):
                        raise ValueError(
                            f"Policy produced a non-finite action for {variant!r}"
                        )
                    action = np.clip(
                        action, env.action_space.low, env.action_space.high
                    ).astype(env.action_space.dtype, copy=False)
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    episode_length += 1
                    episode_success = episode_success or bool(info.get("success", False))
                successes.append(float(episode_success))
                returns.append(episode_return)
                lengths.append(float(episode_length))
        finally:
            env.close()
        variant_metrics[variant] = {
            "reward_type": reward_types[variant],
            "num_episodes": len(successes),
            "success_rate": _mean(successes),
            "return_mean": _mean(returns),
            "return_std": float(np.std(returns)),
            "length_mean": _mean(lengths),
        }
        all_successes.extend(successes)
        all_returns.extend(returns)
        all_lengths.extend(lengths)
    return {
        "aggregate": {
            "num_episodes": len(all_successes),
            "success_rate": _mean(all_successes),
            "return_mean": _mean(all_returns),
            "return_std": float(np.std(all_returns)),
            "length_mean": _mean(all_lengths),
        },
        "variants": variant_metrics,
    }


def evaluate_validation(algo, validation_buffer, *, algorithm: str) -> dict:
    metrics = {
        "action_mse_sum": ContinuousActionDiffEvaluator()(algo, validation_buffer)
    }
    if algorithm in {"td3_bc", "iql"}:
        metrics["td_error"] = TDErrorEvaluator()(algo, validation_buffer)
    return metrics


class BaselineEpochCallback:
    def __init__(
        self,
        *,
        config: dict,
        selections,
        validation_buffer,
        run_dir: Path,
        total_epochs: int,
    ):
        self._config = config
        self._selections = selections
        self._validation_buffer = validation_buffer
        self._run_dir = run_dir
        self._total_epochs = total_epochs
        self.history: list[dict] = []

    def __call__(self, algo, epoch: int, total_step: int) -> None:
        final_epoch = epoch == self._total_epochs
        if epoch % self._config["save_interval_epochs"] == 0 or final_epoch:
            algo.save(self._run_dir / "checkpoints" / f"step_{total_step}.d3")

        evaluation = self._config["evaluation"]
        if not evaluation["enabled"]:
            return
        if epoch % evaluation["every_epochs"] != 0 and not final_epoch:
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
        )
        result = {
            "epoch": epoch,
            "step": total_step,
            "validation": validation,
            "rollout": rollout,
        }
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
