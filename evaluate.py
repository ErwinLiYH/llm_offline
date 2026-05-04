"""Evaluation entry point: rollout the fine-tuned policy in gymnasium environments.

Usage:
    python evaluate.py --config eval.yaml
"""

import argparse
import json
import os
import time
import uuid

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers PointMaze envs
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from data.registry import get_formatter
from data.pointmaze.variants import POINTMAZE_VARIANTS, get_pointmaze_variant_type
from model.policy import load_from_checkpoint
from utils.action_bins import (
    bin_to_continuous,
    get_action_bin_range,
    get_action_bin_codec,
    get_action_num_bins,
    get_action_token_mode,
)
from utils.chat_template import build_generation_prompt
from utils.prompt_loader import load_templates, render_template
from utils.record_format import format_eval_step_text
from utils.variant_selection import get_available_variants, resolve_selection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="eval.yaml")
    return parser.parse_args()



def resolve_standalone_eval_selection(config: dict):
    available_variants = get_available_variants(config["env_family"])

    eval_mode = config.get("eval_mode")
    variants = config.get("variants")
    legacy_variant = config.get("variant")

    if eval_mode is None and variants is None and legacy_variant is not None:
        if legacy_variant == "all":
            eval_mode = "all"
            variants = []
        else:
            eval_mode = "single"
            variants = [legacy_variant]

    resolved_eval_mode = eval_mode or "single"
    selection = resolve_selection(
        mode=resolved_eval_mode,
        variants=variants,
        available_variants=available_variants,
        field_name="variants",
    )
    return selection


ACTION_CONFIG_KEYS = (
    "action_token_mode",
    "action_num_bins",
    "action_bin_min",
    "action_bin_max",
    "new_token",
    "action_soft_label_sigma",
    "action_loss_weight",
    "action_stop_loss_weight",
)


def _load_checkpoint_action_config(model_path: str) -> dict:
    config_path = os.path.join(model_path, "config.yaml")
    saved_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            saved_config = yaml.safe_load(f) or {}

    action_config = {
        "action_token_mode": saved_config.get("action_token_mode", "text"),
        "action_num_bins": saved_config.get("action_num_bins", 10),
        "action_bin_min": saved_config.get("action_bin_min", -1.0),
        "action_bin_max": saved_config.get("action_bin_max", 1.0),
        "new_token": saved_config.get("new_token", False),
    }
    for key in ("action_soft_label_sigma", "action_loss_weight", "action_stop_loss_weight"):
        if key in saved_config:
            action_config[key] = saved_config[key]
    return action_config


def apply_checkpoint_action_config(config: dict) -> dict:
    action_config = _load_checkpoint_action_config(config["model_path"])
    for key in ACTION_CONFIG_KEYS:
        if key in config and config[key] != action_config.get(key):
            raise ValueError(
                f"Standalone eval action config must come from checkpoint config.yaml; "
                f"remove {key}={config[key]!r} from eval.yaml or match checkpoint value {action_config.get(key)!r}."
            )
    merged = dict(config)
    merged.update(action_config)
    get_action_token_mode(merged)
    get_action_num_bins(merged)
    get_action_bin_range(merged)
    return merged



def get_results_base_dir(config: dict) -> str:
    """Build the base results directory from model/training context only."""
    from model.policy import get_model_slug

    model_path = config["model_path"]
    result_root = config.get("result_root", "results")
    norm_path = model_path.replace("\\", "/").rstrip("/")
    parts = [part for part in norm_path.split("/") if part]

    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        ckpt_parts = parts[idx + 1 :]
        if len(ckpt_parts) >= 5:
            env_family, model_slug, train_selection_tag, experiment_id, _checkpoint_tag = ckpt_parts[-5:]
            train_tag = f"train={env_family}-{train_selection_tag}"
            exp_tag = f"exp={experiment_id}"
            return os.path.join(result_root, model_slug, train_tag, exp_tag)

    model_slug = get_model_slug(model_path)
    train_tag = "train=pretrained"
    return os.path.join(result_root, model_slug, train_tag)


def get_standalone_results_dir(base_results_dir: str, standalone_eval_id: str) -> str:
    return os.path.join(base_results_dir, f"standalone_{standalone_eval_id}")


def get_variant_results_dir(parent_results_dir: str, env_family: str, variant: str) -> str:
    return os.path.join(parent_results_dir, f"eval={env_family}-{variant}")


def get_episode_dir(artifacts_dir: str, episode_index: int) -> str:
    return os.path.join(artifacts_dir, f"episode_{episode_index + 1:04d}")


def write_step_log(
    episode_dir: str,
    step_index: int,
    *,
    prompt: str,
    action_text: str,
    executed_action: str,
    parse_status: str,
    attempt_count: int,
    action_bin_probabilities: str | None = None,
):
    steps_dir = os.path.join(episode_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)
    step_path = os.path.join(steps_dir, f"step_{step_index + 1:04d}.txt")
    payload = format_eval_step_text(
        prompt,
        action_text,
        executed_action=executed_action,
        parse_status=parse_status,
        attempt_count=attempt_count,
        action_bin_probabilities=action_bin_probabilities,
    )
    with open(step_path, "w", encoding="utf-8") as f:
        f.write(payload)



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
    skip_special_tokens: bool = True,
    collect_scores: bool = False,
    action_codec=None,
) -> tuple[str, list[int], tuple[torch.Tensor, ...] | None]:
    """Run inference and return display text plus generated action token IDs."""
    encoded = tokenizer(
        text=build_generation_prompt(tokenizer, prompt),
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    generate_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if collect_scores:
        generate_kwargs.update(
            {
                "return_dict_in_generate": True,
                "output_scores": True,
            }
        )

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **generate_kwargs,
        )
    output_ids = outputs.sequences if collect_scores else outputs
    new_tokens = output_ids[0, input_ids.shape[1]:]
    generated_token_ids = [int(token_id) for token_id in new_tokens.detach().cpu().tolist()]
    if action_codec is None:
        text = tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
    else:
        text = action_codec.display_text_for_token_ids(tokenizer, generated_token_ids)
    if not collect_scores:
        return text, generated_token_ids, None
    return text, generated_token_ids, outputs.scores


def _collect_action_bin_probabilities(
    scores,
    action_codec,
    action_dim: int,
) -> list[list[float]]:
    if not scores:
        return []
    bin_token_ids = torch.tensor(action_codec.model_token_ids, device=scores[0].device)
    distributions = []
    for score in scores[:action_dim]:
        bin_logits = score[0].index_select(dim=-1, index=bin_token_ids).float()
        bin_probs = torch.softmax(bin_logits, dim=-1)
        distributions.append([float(value) for value in bin_probs.detach().cpu().tolist()])
    return distributions


def _format_action_bin_probability_log(distributions: list[list[float]], config: dict, action_codec) -> str:
    if not distributions:
        return ""
    num_bins = get_action_num_bins(config)
    low, high = get_action_bin_range(config)
    lines = []
    for dim_idx, probs in enumerate(distributions):
        lines.append(f"dim={dim_idx}")
        for bin_idx, prob in enumerate(probs):
            center = bin_to_continuous(bin_idx, num_bins, low, high)
            lines.append(
                f"  {action_codec.display_tokens[bin_idx]} "
                f"token_id={action_codec.model_token_ids[bin_idx]} "
                f"center={center:.6f} prob={prob:.8f}"
            )
    return "\n".join(lines)


def _format_action_for_mode(formatter, action: np.ndarray, config: dict) -> str:
    if get_action_token_mode(config) == "text":
        return formatter.format_action(action)
    return formatter.format_action_bin_tokens(
        action,
        num_bins=get_action_num_bins(config),
        low=float(config.get("action_bin_min", -1.0)),
        high=float(config.get("action_bin_max", 1.0)),
    )


def _parse_action_for_mode(
    formatter,
    text: str,
    token_ids: list[int],
    config: dict,
    *,
    action_dim: int,
    action_codec=None,
) -> tuple[np.ndarray, bool]:
    if get_action_token_mode(config) == "text":
        return formatter.parse_action(text)
    if action_codec is None:
        raise RuntimeError("Action-bin eval requires an initialized action codec.")
    indices = action_codec.bin_indices_from_token_ids(token_ids, action_dim)
    if len(indices) < action_dim:
        return np.zeros(action_dim, dtype=np.float32), False
    low, high = get_action_bin_range(config)
    return (
        np.array(
            [bin_to_continuous(index, action_codec.num_bins, low, high) for index in indices],
            dtype=np.float32,
        ),
        True,
    )



def _resolve_video_episode_indices(config: dict, num_episodes: int) -> list[int]:
    if not config.get("record_video", False):
        return []

    if config.get("record_all", False):
        return list(range(num_episodes))

    raw_indices = config.get("video_episode_index", 0)
    if isinstance(raw_indices, int):
        indices = [raw_indices]
    elif isinstance(raw_indices, (list, tuple)):
        if not raw_indices:
            raise ValueError("video_episode_index must not be empty when record_video=true and record_all=false")
        if not all(isinstance(idx, int) for idx in raw_indices):
            raise ValueError("video_episode_index must be an int or a list of ints")
        indices = list(raw_indices)
    else:
        raise ValueError("video_episode_index must be an int or a list of ints")

    unique_indices = sorted(set(indices))
    for idx in unique_indices:
        if not (0 <= idx < num_episodes):
            raise ValueError(
                f"video_episode_index must satisfy 0 <= index < num_episodes; got index={idx}, num_episodes={num_episodes}"
            )
    return unique_indices


def configure_mujoco_gl(config: dict):
    mujoco_gl = config.get("mujoco_gl")
    if mujoco_gl:
        os.environ["MUJOCO_GL"] = str(mujoco_gl)
        return

    if config.get("record_video", False):
        os.environ.setdefault("MUJOCO_GL", "egl")



def evaluate_variant(
    config: dict,
    variant: str,
    model,
    tokenizer,
    device: torch.device,
    template: str,
    variant_results_dir: str | None = None,
) -> dict:
    formatter = get_formatter(config["env_family"])
    meta = POINTMAZE_VARIANTS[variant]
    if get_pointmaze_variant_type(meta) == "local":
        env_paras = dict(meta["env_paras"])
        env_id = env_paras.pop("id")
        env_kwargs = env_paras
    else:
        env_id = meta["env_id"]
        env_kwargs = {}
    num_episodes = config["num_episodes"]
    parse_retry_limit = config["parse_retry_limit"]
    history_num = int(config.get("history_num", 0))
    history_stride = int(config.get("history_stride", 1))
    if history_num < 0:
        raise ValueError(f"history_num must be >= 0, got {history_num}")
    if history_stride < 1:
        raise ValueError(f"history_stride must be >= 1, got {history_stride}")

    env_kwargs.update(dict(config.get("env_kwargs") or {}))
    record_video = bool(config.get("record_video", False))
    video_episode_indices = _resolve_video_episode_indices(config, num_episodes)
    video_episode_index_set = set(video_episode_indices)
    video_fps = int(config.get("video_fps", 20))
    record_step_logs = bool(config.get("record_step_logs", True))
    action_token_mode = get_action_token_mode(config)
    collect_bin_probabilities = record_step_logs and action_token_mode == "gaussian_bin"
    action_codec = None
    if action_token_mode != "text":
        action_codec = get_action_bin_codec(tokenizer, config, ensure_registered=True)

    if record_video:
        render_mode = env_kwargs.get("render_mode")
        if render_mode != "rgb_array":
            if render_mode is not None:
                print(
                    f"[eval] record_video=true: overriding env_kwargs.render_mode={render_mode!r} to 'rgb_array'"
                )
            env_kwargs["render_mode"] = "rgb_array"

    env = gym.make(env_id, **env_kwargs)

    episode_returns = []
    episode_successes = []
    episode_steps = []
    total_parse_failures = 0
    total_fallbacks = 0
    total_action_time = 0.0
    total_actions = 0
    saved_video_paths = []
    episode_artifact_dirs = []

    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        history_buffer = []
        record_this_episode = record_video and ep_idx in video_episode_index_set
        episode_frames = [] if record_this_episode else None
        episode_dir = None

        if variant_results_dir is not None:
            episode_dir = get_episode_dir(variant_results_dir, ep_idx)
            os.makedirs(episode_dir, exist_ok=True)
            episode_artifact_dirs.append(episode_dir)

        if episode_frames is not None:
            _capture_render_frame(env, episode_frames)

        ep_return = 0.0
        ep_success = False
        ep_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if history_num > 0:
                sampled_history = []
                hist_idx = len(history_buffer) - 1
                while hist_idx >= 0 and len(sampled_history) < history_num:
                    sampled_entry = dict(history_buffer[hist_idx])
                    sampled_entry["steps_ago"] = len(history_buffer) - hist_idx
                    sampled_history.append(sampled_entry)
                    hist_idx -= history_stride
                sampled_history.reverse()
            else:
                sampled_history = []
            history_payload = formatter.format_history(sampled_history, meta["prompt_vars"])
            obs_payload = formatter.format_obs(obs, meta["prompt_vars"])
            prompt = render_template(template, meta["prompt_vars"], **obs_payload, **history_payload)

            action = None
            executed_action_text = None
            generated_attempts = []
            generated_probability_logs = []
            attempt_count = 0
            current_obs_vec = obs["observation"].astype(np.float32)
            for _attempt in range(parse_retry_limit + 1):
                attempt_count += 1
                t0 = time.perf_counter()
                generated, generated_token_ids, generation_scores = generate_action(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    skip_special_tokens=action_token_mode == "text",
                    collect_scores=collect_bin_probabilities,
                    action_codec=action_codec,
                )
                total_action_time += time.perf_counter() - t0
                total_actions += 1
                generated_attempts.append(generated)
                if collect_bin_probabilities:
                    distributions = _collect_action_bin_probabilities(
                        generation_scores,
                        action_codec,
                        action_dim=env.action_space.shape[0],
                    )
                    probability_log = _format_action_bin_probability_log(distributions, config, action_codec)
                    generated_probability_logs.append(
                        f"[Attempt {attempt_count}]\n{probability_log}" if probability_log else f"[Attempt {attempt_count}]"
                    )
                parsed_action, success = _parse_action_for_mode(
                    formatter,
                    generated,
                    generated_token_ids,
                    config,
                    action_dim=env.action_space.shape[0],
                    action_codec=action_codec,
                )
                if success and formatter.validate_action(parsed_action):
                    action = np.clip(parsed_action, -1.0, 1.0)
                    executed_action_text = _format_action_for_mode(formatter, action, config)
                    break
                total_parse_failures += 1

            if action is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                executed_action_text = _format_action_for_mode(formatter, action, config)
                total_fallbacks += 1
                parse_status = "fallback"
            else:
                parse_status = "success"

            generated_text = "\n\n".join(
                f"[Attempt {idx + 1}]\n{text}" for idx, text in enumerate(generated_attempts)
            )

            if record_step_logs and episode_dir is not None:
                write_step_log(
                    episode_dir,
                    ep_steps,
                    prompt=prompt,
                    action_text=generated_text,
                    executed_action=executed_action_text,
                    parse_status=parse_status,
                    attempt_count=attempt_count,
                    action_bin_probabilities=(
                        "\n\n".join(generated_probability_logs) if generated_probability_logs else None
                    ),
                )

            obs, reward, terminated, truncated, info = env.step(action)
            history_buffer.append(
                {
                    "observation": current_obs_vec,
                    "action_text": executed_action_text,
                }
            )

            if episode_frames is not None:
                _capture_render_frame(env, episode_frames)

            ep_return += float(reward)
            ep_steps += 1

            if terminated:
                ep_success = True

        if episode_frames is not None and episode_dir is not None:
            video_ext = str(config.get("video_format", "gif")).lstrip(".")
            video_path = os.path.join(episode_dir, f"rollout.{video_ext}")
            _save_video(episode_frames, video_path, video_fps)
            saved_video_paths.append(video_path)
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
        "video_path": saved_video_paths[0] if len(saved_video_paths) == 1 else None,
        "video_paths": saved_video_paths,
        "episode_artifact_dirs": episode_artifact_dirs,
        "episode_artifacts_dir": variant_results_dir,
    }



def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = apply_checkpoint_action_config(config)

    eval_selection = resolve_standalone_eval_selection(config)
    configure_mujoco_gl(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")
    print(f"[eval] Loading model from: {config['model_path']}")
    print(f"[eval] Resolved eval variants: {eval_selection.selected_variants}")

    model, tokenizer = load_from_checkpoint(
        config["model_path"],
        load_in_4bit=config.get("load_in_4bit"),
    )
    model.to(device)
    model.eval()

    env_family = config["env_family"]
    standalone_eval_id = uuid.uuid4().hex[:8]
    print(f"[eval] Eval ID: {standalone_eval_id}")

    base_results_dir = get_results_base_dir(config)
    run_results_dir = get_standalone_results_dir(base_results_dir, standalone_eval_id)
    os.makedirs(run_results_dir, exist_ok=True)

    for variant in eval_selection.selected_variants:
        print(f"\n[eval] Evaluating variant: {variant}")
        templates = load_templates(env_family)
        template = templates[0]
        results_dir = get_variant_results_dir(run_results_dir, env_family, variant)
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, "result.json")

        result = evaluate_variant(
            config,
            variant,
            model,
            tokenizer,
            device,
            template,
            variant_results_dir=results_dir,
        )
        result["result_path"] = result_path
        print(
            f"[eval] {variant}: mean_return={result['mean_return']:.4f}, "
            f"success_rate={result['success_rate']:.2%}, "
            f"parse_failures={result['total_parse_failures']}, "
            f"fallbacks={result['total_fallbacks']}"
        )
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] Results saved to: {result_path}")


if __name__ == "__main__":
    main()
