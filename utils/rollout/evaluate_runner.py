from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from crossmaze.eval_position import (
    eval_position_count,
    eval_position_selection_policy,
    get_eval_position_pool_payload,
    get_map_difficulty_config,
    resolve_eval_position_mode,
)
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


def _eval_position_source(episodes: list[EpisodeResult]) -> str | None:
    sources = sorted(
        {
            str(episode.start_goal_source)
            for episode in episodes
            if not episode.worker_failed and episode.start_goal_source is not None
        }
    )
    if not sources:
        return None
    if len(sources) == 1:
        return sources[0]
    return "mixed"


def _mean_difficulty_component(
    episodes: list[EpisodeResult],
    key: str,
) -> float | None:
    values = [
        float(episode.start_goal_difficulty_components[key])
        for episode in episodes
        if not episode.worker_failed
        and episode.start_goal_difficulty_components is not None
        and episode.start_goal_difficulty_components.get(key) is not None
    ]
    return _mean(values) if values else None


def _write_eval_position_pool(
    *,
    variant_results_dir: str | None,
    payload: dict | None,
) -> str | None:
    if variant_results_dir is None or payload is None:
        return None
    os.makedirs(variant_results_dir, exist_ok=True)
    path = os.path.join(variant_results_dir, "eval_position_pool.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


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
    start_goal_difficulties = [
        float(episode.start_goal_difficulty)
        for episode in episodes
        if not episode.worker_failed and episode.start_goal_difficulty is not None
    ]
    eval_seed = int(config.get("seed", 1))
    eval_position_mode = resolve_eval_position_mode(config["env_family"], config)
    map_difficulty_config = get_map_difficulty_config(
        config["env_family"],
        variant,
    )
    position_pool_payload = get_eval_position_pool_payload(
        config["env_family"],
        variant,
        seed=eval_seed,
        config=config,
    )
    position_pool_path = _write_eval_position_pool(
        variant_results_dir=variant_results_dir,
        payload=position_pool_payload,
    )

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
        "mean_start_goal_difficulty": (
            _mean(start_goal_difficulties)
            if start_goal_difficulties
            else None
        ),
        "mean_start_goal_length_score": _mean_difficulty_component(
            episodes,
            "length_score",
        ),
        "mean_start_goal_branch_score": _mean_difficulty_component(
            episodes,
            "branch_score",
        ),
        "mean_start_goal_detour_score": _mean_difficulty_component(
            episodes,
            "detour_score",
        ),
        "difficulty_version": (
            map_difficulty_config.get("difficulty_version")
            if map_difficulty_config is not None
            else None
        ),
        "difficulty_config": (
            map_difficulty_config.get("difficulty_config")
            if map_difficulty_config is not None
            else None
        ),
        "map_difficulty": (
            map_difficulty_config.get("map_difficulty")
            if map_difficulty_config is not None
            else None
        ),
        "map_difficulty_top_fraction": (
            map_difficulty_config.get("map_difficulty_top_fraction")
            if map_difficulty_config is not None
            else None
        ),
        "map_difficulty_path_count": (
            map_difficulty_config.get("map_difficulty_path_count")
            if map_difficulty_config is not None
            else None
        ),
        "map_reachable_pair_count": (
            map_difficulty_config.get("map_reachable_pair_count")
            if map_difficulty_config is not None
            else None
        ),
        "map_diameter": (
            map_difficulty_config.get("map_diameter")
            if map_difficulty_config is not None
            else None
        ),
        "eval_position_pool_path": position_pool_path,
        "eval_position_source": _eval_position_source(episodes),
        "eval_position_mode": eval_position_mode,
        "eval_position_selection_policy": eval_position_selection_policy(
            config["env_family"],
            variant,
            seed=eval_seed,
            config=config,
        ),
        "eval_position_count": eval_position_count(
            config["env_family"],
            variant,
            seed=eval_seed,
            config=config,
        ),
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
