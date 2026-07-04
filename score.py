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
from utils.config_loader import load_merged_config
from utils.pointmaze_score import (
    build_pointmaze_score_env_spec,
    local_reference_path,
    make_pointmaze_score_env,
)
from utils.prompt_loader import load_named_templates, load_template_names
from utils.eval_parallel import apply_rollout_config_defaults
from utils.rollout.score_runner import run_score_variant
from utils.sensing_config import normalize_sensing_config
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
    parser.add_argument("--config", nargs="+", default=["score.yaml"])
    return parser.parse_args()


def load_config(args) -> dict:
    config = load_merged_config(args.config)

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
    config.setdefault("record_video", False)
    config.setdefault("record_all", False)
    config.setdefault("video_episode_index", 0)
    config.setdefault("video_fps", 20)
    config.setdefault("video_format", "gif")
    config.setdefault("video_save_workers", 1)
    config.setdefault("video_save_max_pending", 2)
    config.setdefault("rollout_worker_num", 1)
    config.setdefault("rollout_worker_lifetime", "slot")
    config.setdefault("rollout_worker_retries", 1)
    config.setdefault("rollout_worker_start_timeout_seconds", 120)
    config.setdefault("rollout_action_timeout_seconds", 300)
    config.setdefault("policy_batch_timeout_ms", 10)
    config = apply_rollout_config_defaults(config)
    config["score_config_source"] = (
        args.config[0] if len(args.config) == 1 else list(args.config)
    )
    config["config_sources"] = list(args.config)
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


def configure_mujoco_gl(config: dict):
    mujoco_gl = config.get("mujoco_gl")
    if mujoco_gl:
        os.environ["MUJOCO_GL"] = str(mujoco_gl)
        return

    if config.get("record_video", False):
        os.environ.setdefault("MUJOCO_GL", "egl")


def print_sensing_config(prefix: str, config: dict) -> None:
    print(
        f"{prefix} Wall sensing: "
        f"version={config['wall_sensing_version']}, "
        f"boundary_risk_threshold={config['map_sensing_boundary_risk_threshold']}"
    )


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
    from evaluate import (
        apply_checkpoint_action_config,
        apply_checkpoint_prompt_config,
        apply_checkpoint_sensing_config,
    )

    config = dict(config)
    for prompt_key in ("prompt_templete_index", "prompt_template_index"):
        if prompt_key in config and config[prompt_key] is None:
            config.pop(prompt_key)
    config = apply_checkpoint_action_config(config)
    config = apply_checkpoint_sensing_config(config)
    config = apply_checkpoint_prompt_config(config, assume_yes=assume_yes)
    prompt_name = config.get("resolved_eval_prompt_name")
    if prompt_name is None:
        prompt_name = load_template_names(config["env_family"])[0]
        config["resolved_eval_prompt_name"] = prompt_name
    template = load_named_templates(config["env_family"], [prompt_name])[0]
    return config, prompt_name, template


def score_model_variant(
    *,
    config: dict,
    variant: str,
    model,
    tokenizer,
    device: torch.device,
    template: str,
    prompt_template_name: str,
    variant_results_dir: str | None = None,
) -> dict:
    return run_score_variant(
        config=config,
        variant=variant,
        model=model,
        tokenizer=tokenizer,
        device=device,
        template=template,
        prompt_template_name=prompt_template_name,
        variant_results_dir=variant_results_dir,
    )


def run_score_mode(config: dict, selection, run_results_dir: str, *, assume_yes: bool) -> tuple[dict, list[dict]]:
    config, prompt_name, template = _resolve_score_prompt(config, assume_yes=assume_yes)

    from model.policy import load_from_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[score] Using device: {device}")
    print(f"[score] Loading model from: {config['model_path']}")
    print_sensing_config("[score]", config)
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
            variant_results_dir=variant_dir,
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
        "selection_tag_full": selection.full_selection_tag,
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
        normalize_sensing_config(config)
        print_sensing_config("[score]", config)
        results = run_reference_mode(config, selection, run_results_dir)
    elif mode == "score":
        configure_mujoco_gl(config)
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
