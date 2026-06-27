from __future__ import annotations

from dataclasses import asdict

import numpy as np

from data.registry import get_formatter
from utils.action_bins import uses_action_bins
from utils.eval_parallel import (
    apply_rollout_config_defaults,
    resolve_rollout_worker_lifetime,
    resolve_rollout_worker_num,
)
from utils.rollout.policy import RolloutPolicy
from utils.rollout.protocol import EpisodeResult
from utils.rollout.supervisor import run_episode_supervisor


def _ordered_episode_values(episodes: list[EpisodeResult], attr: str):
    return [getattr(episode, attr) for episode in episodes]


def _mean(values, default: float = 0.0) -> float:
    return float(np.mean(values)) if values else float(default)


def _std(values, default: float = 0.0) -> float:
    return float(np.std(values)) if values else float(default)


def run_evaluate_variant(
    *,
    config: dict,
    variant: str,
    model,
    tokenizer,
    device,
    template: str,
    variant_results_dir: str | None = None,
) -> dict:
    config = apply_rollout_config_defaults(config)
    formatter = get_formatter(config["env_family"])
    collect_bin_probabilities = bool(config.get("record_step_logs", True)) and uses_action_bins(config)
    policy = RolloutPolicy(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        formatter=formatter,
        collect_bin_probabilities=collect_bin_probabilities,
    )
    supervisor_result = run_episode_supervisor(
        config=config,
        variant=variant,
        mode="eval",
        template=template,
        policy=policy,
        variant_results_dir=variant_results_dir,
    )
    episodes = supervisor_result.episode_results
    returns = _ordered_episode_values(episodes, "episode_return")
    successes = _ordered_episode_values(episodes, "success")
    steps = _ordered_episode_values(episodes, "steps")
    video_paths = [
        episode.video_path
        for episode in episodes
        if episode.video_path is not None
    ]
    global_video_paths = [
        episode.global_video_path
        for episode in episodes
        if episode.global_video_path is not None
    ]
    episode_artifact_dirs = [
        episode.episode_artifact_dir
        for episode in episodes
        if episode.episode_artifact_dir is not None
    ]
    total_action_time = sum(float(episode.action_time_seconds) for episode in episodes)
    total_actions = sum(int(episode.action_count) for episode in episodes)
    mean_action_time_ms = (total_action_time / total_actions * 1000) if total_actions else 0.0
    worker_failures = [failure.to_dict() for failure in supervisor_result.worker_failures]

    return {
        "variant": variant,
        "num_episodes": int(config["num_episodes"]),
        "seed": int(config.get("seed", 1)),
        "episode_seeds": [int(config.get("seed", 1)) + idx for idx in range(int(config["num_episodes"]))],
        "mean_return": _mean(returns),
        "std_return": _std(returns),
        "success_rate": _mean(successes),
        "mean_episode_steps": _mean(steps),
        "std_episode_steps": _std(steps),
        "total_parse_failures": int(sum(episode.parse_failures for episode in episodes)),
        "total_fallbacks": int(sum(episode.fallbacks for episode in episodes)),
        "mean_action_time_ms": round(mean_action_time_ms, 2),
        "video_path": video_paths[0] if len(video_paths) == 1 else None,
        "video_paths": video_paths,
        "global_video_path": (
            global_video_paths[0] if len(global_video_paths) == 1 else None
        ),
        "global_video_paths": global_video_paths,
        "all_video_paths": video_paths + global_video_paths,
        "episode_artifact_dirs": episode_artifact_dirs,
        "episode_artifacts_dir": variant_results_dir,
        "rollout_isolation": "process",
        "rollout_worker_num": resolve_rollout_worker_num(config),
        "rollout_worker_lifetime": resolve_rollout_worker_lifetime(config),
        "rollout_workers_used": supervisor_result.workers_used,
        "worker_failures": worker_failures,
        "completed_episodes": int(sum(1 for episode in episodes if not episode.worker_failed)),
        "failed_episodes": int(sum(1 for episode in episodes if episode.worker_failed)),
        "episode_results": [asdict(episode) for episode in episodes],
        "video_save_workers": int(config.get("video_save_workers", 1)),
        "video_save_max_pending": config.get("video_save_max_pending", 2),
    }

