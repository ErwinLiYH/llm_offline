"""Evaluation entry point: rollout the fine-tuned policy in gymnasium environments.

Usage:
    python evaluate.py --config eval.yaml
"""

import argparse
import json
import os
import time

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers PointMaze envs
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from data.registry import get_formatter
from data.pointmaze.variants import POINTMAZE_VARIANTS
from model.policy import load_from_checkpoint
from utils.prompt_loader import load_templates, render_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="eval.yaml")
    return parser.parse_args()


def get_results_dir(config: dict) -> str:
    """Build results directory path encoding model, training context, and eval context."""
    from model.policy import get_model_slug

    model_path = config["model_path"]
    eval_tag = f"eval={config['env_family']}-{config['variant']}"

    if model_path.startswith("checkpoints/") or model_path.startswith("checkpoints\\"):
        parts = model_path.replace("\\", "/").rstrip("/").split("/")
        model_slug = parts[2] if len(parts) > 2 else model_path
        train_tag = f"train={parts[1]}-{parts[4]}-{parts[3]}"
        exp_tag = f"exp={parts[5]}" if len(parts) > 5 else "exp=unknown"
        return os.path.join("results", model_slug, train_tag, exp_tag, eval_tag)

    model_slug = get_model_slug(model_path)
    train_tag = "train=pretrained"
    return os.path.join("results", model_slug, train_tag, eval_tag)


def _normalize_render_frame(frame) -> np.ndarray:
    if isinstance(frame, (list, tuple)):
        if not frame:
            raise ValueError("Environment render() returned an empty frame list")
        frame = frame[-1]

    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported frame shape from render(): {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if float(arr.max(initial=0.0)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _capture_render_frame(env, frames: list[np.ndarray]):
    frame = env.render()
    if frame is None:
        raise ValueError("render() returned None; use env_kwargs.render_mode='rgb_array' when recording")
    frames.append(_normalize_render_frame(frame))


def _save_video(frames: list[np.ndarray], output_path: str, fps: int):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        duration_sec = 1.0 / max(fps, 1)
        imageio.mimsave(output_path, frames, format="GIF", duration=duration_sec)
        return

    try:
        imageio.mimsave(output_path, frames, fps=fps)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save video to {output_path}. mp4 output requires a working ffmpeg backend; "
            "try video_format='gif' if ffmpeg is unavailable."
        ) from exc


def generate_action(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 20,
) -> str:
    """Run inference and return the generated text (action portion)."""
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_variant(
    config: dict,
    variant: str,
    model,
    tokenizer,
    device: torch.device,
    template: str,
    video_path: str | None = None,
) -> dict:
    formatter = get_formatter(config["env_family"])
    meta = POINTMAZE_VARIANTS[variant]
    env_id = meta["env_id"]
    num_episodes = config["num_episodes"]
    parse_retry_limit = config["parse_retry_limit"]

    env_kwargs = dict(config.get("env_kwargs") or {})
    record_video = bool(config.get("record_video", False))
    video_episode_index = int(config.get("video_episode_index", 0))
    video_fps = int(config.get("video_fps", 20))

    if record_video:
        if not (0 <= video_episode_index < num_episodes):
            raise ValueError(
                f"video_episode_index must satisfy 0 <= index < num_episodes; got index={video_episode_index}, num_episodes={num_episodes}"
            )
        render_mode = env_kwargs.get("render_mode")
        if render_mode is None:
            env_kwargs["render_mode"] = "rgb_array"
        elif render_mode != "rgb_array":
            raise ValueError(
                "record_video requires env_kwargs.render_mode='rgb_array' (or omit render_mode and it will be set automatically)"
            )

    env = gym.make(env_id, **env_kwargs)

    episode_returns = []
    episode_successes = []
    episode_steps = []
    total_parse_failures = 0
    total_fallbacks = 0
    total_action_time = 0.0
    total_actions = 0
    saved_video_path = None

    for ep_idx in range(num_episodes):
        obs_dict, info = env.reset()
        record_this_episode = record_video and ep_idx == video_episode_index
        episode_frames = [] if record_this_episode else None

        obs = obs_dict["observation"].astype(np.float32)
        goal = obs_dict["desired_goal"].astype(np.float32)

        if episode_frames is not None:
            _capture_render_frame(env, episode_frames)

        ep_return = 0.0
        ep_success = False
        ep_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            obs_text = formatter.format_obs(obs, goal)
            prompt = render_template(template, meta["prompt_vars"], obs_text=obs_text)

            action = None
            for _attempt in range(parse_retry_limit + 1):
                t0 = time.perf_counter()
                generated = generate_action(model, tokenizer, prompt, device)
                total_action_time += time.perf_counter() - t0
                total_actions += 1
                parsed_action, success = formatter.parse_action(generated)
                if success and formatter.validate_action(parsed_action):
                    action = np.clip(parsed_action, -1.0, 1.0)
                    break
                total_parse_failures += 1

            if action is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                total_fallbacks += 1

            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["observation"].astype(np.float32)
            goal = obs_dict["desired_goal"].astype(np.float32)

            if episode_frames is not None:
                _capture_render_frame(env, episode_frames)

            ep_return += float(reward)
            ep_steps += 1

            if terminated:
                ep_success = True

        if episode_frames is not None and video_path is not None:
            _save_video(episode_frames, video_path, video_fps)
            saved_video_path = video_path
            print(f"  [{variant}] saved video: {video_path}")

        episode_returns.append(ep_return)
        episode_successes.append(ep_success)
        episode_steps.append(ep_steps)

        if (ep_idx + 1) % 5 == 0 or record_this_episode:
            print(
                f"  [{variant}] episode {ep_idx+1}/{num_episodes} | "
                f"return={ep_return:.2f} | steps={ep_steps} | success={ep_success}"
            )

    env.close()

    mean_action_time_ms = (total_action_time / total_actions * 1000) if total_actions > 0 else 0.0
    return {
        "variant": variant,
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(episode_successes)),
        "mean_episode_steps": float(np.mean(episode_steps)),
        "std_episode_steps": float(np.std(episode_steps)),
        "total_parse_failures": total_parse_failures,
        "total_fallbacks": total_fallbacks,
        "mean_action_time_ms": round(mean_action_time_ms, 2),
        "video_path": saved_video_path,
    }


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")
    print(f"[eval] Loading model from: {config['model_path']}")

    model, tokenizer = load_from_checkpoint(
        config["model_path"],
        load_in_4bit=config.get("load_in_4bit"),
    )
    model.to(device)
    model.eval()

    env_family = config["env_family"]
    variant_arg = config["variant"]

    if variant_arg == "all":
        variants_to_eval = list(POINTMAZE_VARIANTS.keys())
    else:
        variants_to_eval = [variant_arg]

    results_dir = get_results_dir(config)
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for variant in variants_to_eval:
        print(f"\n[eval] Evaluating variant: {variant}")
        templates = load_templates(env_family)
        template = templates[0]

        video_path = None
        if config.get("record_video", False):
            video_ext = str(config.get("video_format", "gif")).lstrip(".")
            episode_num = int(config.get("video_episode_index", 0)) + 1
            video_path = os.path.join(results_dir, f"{variant}-episode{episode_num}.{video_ext}")

        result = evaluate_variant(
            config,
            variant,
            model,
            tokenizer,
            device,
            template,
            video_path=video_path,
        )
        all_results.append(result)
        print(
            f"[eval] {variant}: mean_return={result['mean_return']:.4f}, "
            f"success_rate={result['success_rate']:.2%}, "
            f"parse_failures={result['total_parse_failures']}, "
            f"fallbacks={result['total_fallbacks']}"
        )

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
