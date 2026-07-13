from __future__ import annotations

from dataclasses import asdict

import numpy as np

from data.pointmaze.variants import POINTMAZE_VARIANTS, get_pointmaze_variant_type
from data.registry import get_formatter
from utils.eval_parallel import (
    apply_rollout_config_defaults,
    resolve_rollout_worker_lifetime,
    resolve_rollout_worker_num,
)
from utils.pointmaze_score import (
    build_pointmaze_score_env_spec,
    get_remote_pointmaze_reference,
    load_and_validate_local_reference,
    normalize_score,
    normalize_score_std,
)
from utils.rollout.policy import RolloutPolicy
from utils.rollout.protocol import EpisodeResult
from utils.rollout.supervisor import run_episode_supervisor

_EVAL_POSITION_EPISODE_KEYS = {
    "start_cell",
    "goal_cell",
    "start_goal_difficulty",
    "start_goal_difficulty_components",
    "start_goal_source",
    "start_goal_index",
}


def _reference_for_variant(config: dict, variant: str, score_env_spec) -> dict:
    meta = POINTMAZE_VARIANTS[variant]
    if get_pointmaze_variant_type(meta) == "remote":
        return get_remote_pointmaze_reference(variant)

    reference, reference_path = load_and_validate_local_reference(
        config=config,
        variant=variant,
        score_env_spec=score_env_spec,
    )
    return {
        "ref_min_score": float(reference["ref_min_score"]),
        "ref_max_score": float(reference["ref_max_score"]),
        "reference_source": str(reference_path),
        "num_episodes_average_score": reference.get("num_reference_episodes"),
    }


def _mean(values, default: float = 0.0) -> float:
    return float(np.mean(values)) if values else float(default)


def _std(values, default: float = 0.0) -> float:
    return float(np.std(values)) if values else float(default)


def _score_episode_dict(episode: EpisodeResult) -> dict:
    payload = asdict(episode)
    for key in _EVAL_POSITION_EPISODE_KEYS:
        payload.pop(key, None)
    return payload


def run_score_variant(
    *,
    config: dict,
    variant: str,
    model,
    tokenizer,
    device,
    template: str,
    prompt_template_name: str,
    variant_results_dir: str | None = None,
) -> dict:
    config = apply_rollout_config_defaults(config)
    score_env_spec = build_pointmaze_score_env_spec(variant, config)
    reference = _reference_for_variant(config, variant, score_env_spec)
    formatter = get_formatter(config["env_family"])
    policy = RolloutPolicy(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        formatter=formatter,
        collect_bin_probabilities=False,
    )
    supervisor_result = run_episode_supervisor(
        config=config,
        variant=variant,
        mode="score",
        template=template,
        policy=policy,
        variant_results_dir=variant_results_dir,
    )
    episodes: list[EpisodeResult] = supervisor_result.episode_results
    episode_returns = [float(episode.episode_return) for episode in episodes]
    episode_steps = [int(episode.steps) for episode in episodes]
    mean_return = _mean(episode_returns)
    std_return = _std(episode_returns)
    ref_min_score = float(reference["ref_min_score"])
    ref_max_score = float(reference["ref_max_score"])
    total_action_time = sum(float(episode.action_time_seconds) for episode in episodes)
    total_actions = sum(int(episode.action_count) for episode in episodes)
    mean_action_time_ms = (total_action_time / total_actions * 1000) if total_actions else 0.0
    video_paths = [
        episode.video_path
        for episode in episodes
        if episode.video_path is not None
    ]
    episode_artifact_dirs = [
        episode.episode_artifact_dir
        for episode in episodes
        if episode.episode_artifact_dir is not None
    ]
    video_episode_indices = [
        episode.episode_index
        for episode in episodes
        if episode.video_path is not None
    ]

    return {
        "variant": variant,
        "env_family": config["env_family"],
        "mode": "score",
        "mean_return": mean_return,
        "std_return": std_return,
        "num_episodes": int(config["num_episodes"]),
        "episode_returns": episode_returns,
        "episode_steps": episode_steps,
        "normalized_score": normalize_score(mean_return, ref_min_score, ref_max_score),
        "std_normalized_score": normalize_score_std(std_return, ref_min_score, ref_max_score),
        "ref_min_score": ref_min_score,
        "ref_max_score": ref_max_score,
        "reference_source": reference["reference_source"],
        "num_episodes_average_score": reference.get("num_episodes_average_score"),
        "score_env_spec": score_env_spec.to_result_dict(),
        "seed": int(config["seed"]),
        "prompt_template_name": prompt_template_name,
        "total_parse_failures": int(sum(episode.parse_failures for episode in episodes)),
        "total_fallbacks": int(sum(episode.fallbacks for episode in episodes)),
        "mean_action_time_ms": round(mean_action_time_ms, 2),
        "video_path": video_paths[0] if len(video_paths) == 1 else None,
        "video_paths": video_paths,
        "video_episode_indices": video_episode_indices,
        "episode_artifact_dirs": episode_artifact_dirs,
        "rollout_isolation": "process",
        "rollout_worker_num": resolve_rollout_worker_num(config),
        "rollout_worker_lifetime": resolve_rollout_worker_lifetime(config),
        "rollout_workers_used": supervisor_result.workers_used,
        "worker_failures": [failure.to_dict() for failure in supervisor_result.worker_failures],
        "completed_episodes": int(sum(1 for episode in episodes if not episode.worker_failed)),
        "failed_episodes": int(sum(1 for episode in episodes if episode.worker_failed)),
        "episode_results": [_score_episode_dict(episode) for episode in episodes],
        "video_save_workers": int(config.get("video_save_workers", 1)),
        "video_save_max_pending": config.get("video_save_max_pending", 2),
    }
