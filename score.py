"""Official-style PointMaze normalized score entry point.

Usage:
    python score.py --config score.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import uuid
from pathlib import Path

import gymnasium_robotics  # noqa: F401  registers PointMaze envs
import numpy as np
import torch
import yaml

from data.pointmaze.variants import POINTMAZE_VARIANTS, get_pointmaze_variant_type
from data.registry import get_formatter
from utils.pointmaze_score import (
    build_pointmaze_score_env_spec,
    get_remote_pointmaze_reference,
    load_and_validate_local_reference,
    local_reference_path,
    make_pointmaze_score_env,
    normalize_score,
    normalize_score_std,
)
from utils.prompt_loader import load_named_templates, load_template_names
from utils.variant_selection import get_available_variants, resolve_selection


OFFICIAL_POINTMAZE_DIR = (
    Path(__file__).resolve().parent
    / "third_party"
    / "minari-dataset-generation-scripts"
    / "scripts"
    / "pointmaze"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="score.yaml")
    return parser.parse_args()


def load_config(args) -> dict:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config.setdefault("env_family", "pointmaze")
    config.setdefault("mode", "score")
    config.setdefault("eval_mode", "single")
    config.setdefault("variants", [])
    config.setdefault("num_episodes", 100)
    config.setdefault("seed", 123)
    config.setdefault("parse_retry_limit", 3)
    config.setdefault("history_num", 0)
    config.setdefault("history_stride", 1)
    config.setdefault("action_sampling", False)
    config.setdefault("assume_yes", False)
    config.setdefault("local_reference_root", "local_references/pointmaze")
    config["score_config_source"] = args.config
    return config


def resolve_score_selection(config: dict):
    if config["env_family"] != "pointmaze":
        raise ValueError("score.py currently supports env_family='pointmaze' only")
    return resolve_selection(
        mode=config.get("eval_mode", "single"),
        variants=config.get("variants"),
        available_variants=get_available_variants(config["env_family"]),
        field_name="variants",
    )


def get_run_results_dir(config: dict, mode: str, score_id: str) -> str:
    result_root = config.get("result_root", "score_results")
    return os.path.join(result_root, f"{mode}_{score_id}")


def get_variant_results_dir(parent_results_dir: str, env_family: str, variant: str) -> str:
    return os.path.join(parent_results_dir, f"score={env_family}-{variant}")


def save_json(path: str | Path, payload: dict):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_waypoint_controller_class():
    if not OFFICIAL_POINTMAZE_DIR.exists():
        raise RuntimeError(
            "Official Farama PointMaze generator scripts are missing. "
            "Run: git submodule update --init --recursive"
        )
    official_path = str(OFFICIAL_POINTMAZE_DIR)
    if official_path not in sys.path:
        sys.path.insert(0, official_path)
    return importlib.import_module("controller").WaypointController


class RandomPolicy:
    def __init__(self, env, seed: int):
        self.action_space = env.action_space
        self.action_space.seed(seed)

    def reset(self):
        pass

    def __call__(self, obs):
        return self.action_space.sample()


class WaypointControllerPolicy:
    def __init__(self, env):
        self.env = env
        self.controller_cls = _load_waypoint_controller_class()
        self.controller = None
        self.reset()

    def reset(self):
        self.controller = self.controller_cls(
            maze=self.env.unwrapped.maze,
            maze_solver="QIteration",
        )

    def __call__(self, obs):
        return self.controller.compute_action(obs)


def run_reference_policy_returns(
    score_env_spec,
    *,
    policy_kind: str,
    num_episodes: int,
    seed: int,
) -> list[float]:
    env = make_pointmaze_score_env(score_env_spec)
    try:
        np.random.seed(seed)
        if policy_kind == "random":
            policy = RandomPolicy(env, seed=seed)
        elif policy_kind == "waypoint":
            policy = WaypointControllerPolicy(env)
        else:
            raise ValueError(f"Unknown reference policy kind: {policy_kind!r}")

        episode_returns = []
        for ep_idx in range(num_episodes):
            reset_seed = seed if ep_idx == 0 else None
            obs, _ = env.reset(seed=reset_seed)
            if hasattr(policy, "reset"):
                policy.reset()
            ep_return = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = policy(obs)
                obs, reward, terminated, truncated, _info = env.step(action)
                ep_return += float(reward)
            episode_returns.append(ep_return)
        return episode_returns
    finally:
        env.close()


def run_reference_mode(config: dict, selection, run_results_dir: str) -> list[dict]:
    num_reference_episodes = int(
        config.get("num_reference_episodes", config.get("num_episodes", 100))
    )
    if num_reference_episodes < 1:
        raise ValueError("num_reference_episodes must be >= 1")

    results = []
    for variant in selection.selected_variants:
        meta = POINTMAZE_VARIANTS[variant]
        if get_pointmaze_variant_type(meta) != "local":
            raise ValueError(f"Reference mode only supports local variants, got {variant!r}")

        print(f"[score] Generating local reference: {variant}")
        score_env_spec = build_pointmaze_score_env_spec(variant, config)
        random_returns = run_reference_policy_returns(
            score_env_spec,
            policy_kind="random",
            num_episodes=num_reference_episodes,
            seed=int(config["seed"]),
        )
        expert_returns = run_reference_policy_returns(
            score_env_spec,
            policy_kind="waypoint",
            num_episodes=num_reference_episodes,
            seed=int(config["seed"]),
        )
        ref_min_score = float(np.mean(random_returns))
        ref_max_score = float(np.mean(expert_returns))

        reference_path = local_reference_path(config, variant)
        payload = {
            "variant": variant,
            "env_family": "pointmaze",
            "mode": "reference",
            "ref_min_score": ref_min_score,
            "ref_max_score": ref_max_score,
            "num_reference_episodes": num_reference_episodes,
            "seed": int(config["seed"]),
            "horizon": score_env_spec.max_episode_steps,
            "goal_cell": score_env_spec.goal_cell,
            "env_fingerprint": score_env_spec.env_fingerprint,
            "reward_type": score_env_spec.reward_type,
            "score_env_spec": score_env_spec.to_result_dict(),
            "random_policy_episode_returns": random_returns,
            "waypoint_controller_episode_returns": expert_returns,
            "method": {
                "ref_min_score": "seeded_random_policy",
                "ref_max_score": "Farama WaypointController(maze_solver='QIteration')",
                "action_noise": 0.0,
            },
            "reference_path": str(reference_path),
        }
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        with open(reference_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        variant_dir = get_variant_results_dir(run_results_dir, "pointmaze", variant)
        result_path = os.path.join(variant_dir, "result.json")
        payload["result_path"] = result_path
        save_json(result_path, payload)
        print(
            f"[score] {variant}: ref_min={ref_min_score:.4f}, "
            f"ref_max={ref_max_score:.4f}, saved={reference_path}"
        )
        results.append(payload)
    return results


def _resolve_score_prompt(config: dict, *, assume_yes: bool) -> tuple[dict, str, str]:
    from evaluate import apply_checkpoint_action_config, apply_checkpoint_prompt_config

    config = dict(config)
    for prompt_key in ("prompt_templete_index", "prompt_template_index"):
        if prompt_key in config and config[prompt_key] is None:
            config.pop(prompt_key)
    config = apply_checkpoint_action_config(config)
    config = apply_checkpoint_prompt_config(config, assume_yes=assume_yes)
    prompt_name = config.get("resolved_eval_prompt_name")
    if prompt_name is None:
        prompt_name = load_template_names(config["env_family"])[0]
        config["resolved_eval_prompt_name"] = prompt_name
    template = load_named_templates(config["env_family"], [prompt_name])[0]
    return config, prompt_name, template


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


def score_model_variant(
    *,
    config: dict,
    variant: str,
    model,
    tokenizer,
    device: torch.device,
    template: str,
    prompt_template_name: str,
) -> dict:
    from utils.eval_rollout import (
        build_action_rollout_context,
        generate_valid_action,
        render_policy_prompt,
        validate_history_config,
    )

    formatter = get_formatter(config["env_family"])
    meta = POINTMAZE_VARIANTS[variant]
    prompt_vars = meta["prompt_vars"]
    score_env_spec = build_pointmaze_score_env_spec(variant, config)
    reference = _reference_for_variant(config, variant, score_env_spec)

    env = make_pointmaze_score_env(score_env_spec)
    try:
        action_dim = int(env.action_space.shape[0])
        history_num, history_stride = validate_history_config(config)
        action_context = build_action_rollout_context(
            config=config,
            tokenizer=tokenizer,
            action_dim=action_dim,
            collect_bin_probabilities=False,
        )

        episode_returns = []
        episode_steps = []
        total_parse_failures = 0
        total_fallbacks = 0
        total_action_time = 0.0
        total_actions = 0

        num_episodes = int(config["num_episodes"])
        for ep_idx in range(num_episodes):
            reset_seed = int(config["seed"]) if ep_idx == 0 else None
            obs, _ = env.reset(seed=reset_seed)
            history_buffer = []
            ep_return = 0.0
            ep_steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                prompt = render_policy_prompt(
                    formatter=formatter,
                    template=template,
                    prompt_vars=prompt_vars,
                    obs=obs,
                    history_buffer=history_buffer,
                    history_num=history_num,
                    history_stride=history_stride,
                )
                current_obs_vec = obs["observation"].astype(np.float32)
                action_result = generate_valid_action(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    formatter=formatter,
                    prompt=prompt,
                    config=config,
                    action_context=action_context,
                    action_shape=env.action_space.shape,
                    action_dim=action_dim,
                    parse_retry_limit=int(config["parse_retry_limit"]),
                )

                obs, reward, terminated, truncated, _info = env.step(action_result.action)
                history_buffer.append(
                    {
                        "observation": current_obs_vec,
                        "action_text": action_result.executed_action_text,
                    }
                )

                ep_return += float(reward)
                ep_steps += 1
                total_parse_failures += action_result.parse_failures
                total_fallbacks += action_result.fallback_count
                total_action_time += action_result.action_time_seconds
                total_actions += action_result.generation_count

            episode_returns.append(ep_return)
            episode_steps.append(ep_steps)
            print(
                f"  [{variant}] episode {ep_idx + 1}/{num_episodes} | "
                f"return={ep_return:.4f} | steps={ep_steps}"
            )

        mean_return = float(np.mean(episode_returns))
        std_return = float(np.std(episode_returns))
        ref_min_score = float(reference["ref_min_score"])
        ref_max_score = float(reference["ref_max_score"])
        mean_action_time_ms = (total_action_time / total_actions * 1000) if total_actions > 0 else 0.0
        return {
            "variant": variant,
            "env_family": config["env_family"],
            "mode": "score",
            "mean_return": mean_return,
            "std_return": std_return,
            "num_episodes": num_episodes,
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
            "total_parse_failures": total_parse_failures,
            "total_fallbacks": total_fallbacks,
            "mean_action_time_ms": round(mean_action_time_ms, 2),
        }
    finally:
        env.close()


def run_score_mode(config: dict, selection, run_results_dir: str, *, assume_yes: bool) -> tuple[dict, list[dict]]:
    config, prompt_name, template = _resolve_score_prompt(config, assume_yes=assume_yes)

    from model.policy import load_from_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[score] Using device: {device}")
    print(f"[score] Loading model from: {config['model_path']}")
    model, tokenizer = load_from_checkpoint(
        config["model_path"],
        load_in_4bit=config.get("load_in_4bit"),
    )
    model.to(device)
    model.eval()

    results = []
    for variant in selection.selected_variants:
        print(f"\n[score] Scoring variant: {variant}")
        variant_dir = get_variant_results_dir(run_results_dir, config["env_family"], variant)
        result_path = os.path.join(variant_dir, "result.json")
        result = score_model_variant(
            config=config,
            variant=variant,
            model=model,
            tokenizer=tokenizer,
            device=device,
            template=template,
            prompt_template_name=prompt_name,
        )
        result["result_path"] = result_path
        save_json(result_path, result)
        print(
            f"[score] {variant}: mean_return={result['mean_return']:.4f}, "
            f"normalized_score={result['normalized_score']:.4f}, "
            f"parse_failures={result['total_parse_failures']}, "
            f"fallbacks={result['total_fallbacks']}"
        )
        print(f"[score] Results saved to: {result_path}")
        results.append(result)
    return config, results


def write_run_summary(
    *,
    config: dict,
    selection,
    mode: str,
    score_id: str,
    run_results_dir: str,
    results: list[dict],
) -> str:
    summary = {
        "score_id": score_id,
        "mode": mode,
        "env_family": config["env_family"],
        "selected_variants": selection.selected_variants,
        "selection_tag": selection.selection_tag,
        "result_count": len(results),
        "results": results,
    }
    if mode == "score" and results:
        summary["mean_normalized_score"] = float(
            np.mean([result["normalized_score"] for result in results])
        )
    summary_path = os.path.join(run_results_dir, "summary.json")
    save_json(summary_path, summary)
    return summary_path


def main():
    args = parse_args()
    config = load_config(args)
    selection = resolve_score_selection(config)
    mode = config["mode"]
    score_id = uuid.uuid4().hex[:8]
    run_results_dir = get_run_results_dir(config, mode, score_id)
    os.makedirs(run_results_dir, exist_ok=True)

    config["score_id"] = score_id
    config["score_results_dir"] = run_results_dir
    config["resolved_score_variants"] = selection.selected_variants
    print(f"[score] Mode: {mode}")
    print(f"[score] Score ID: {score_id}")
    print(f"[score] Resolved variants: {selection.selected_variants}")

    if mode == "reference":
        results = run_reference_mode(config, selection, run_results_dir)
    elif mode == "score":
        config, results = run_score_mode(
            config,
            selection,
            run_results_dir,
            assume_yes=bool(config.get("assume_yes", False)),
        )
    else:
        raise ValueError(f"Unknown score mode: {mode!r}")

    config_path = os.path.join(run_results_dir, "score_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    summary_path = write_run_summary(
        config=config,
        selection=selection,
        mode=mode,
        score_id=score_id,
        run_results_dir=run_results_dir,
        results=results,
    )
    print(f"[score] Config saved to: {config_path}")
    print(f"[score] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
