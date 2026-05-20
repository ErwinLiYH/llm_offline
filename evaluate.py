"""Evaluation entry point: rollout the fine-tuned policy in gymnasium environments.

Usage:
    python evaluate.py --config eval.yaml
"""

import argparse
import json
import os
import time
import uuid
import warnings

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers PointMaze envs
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from data.registry import get_formatter
from data.pointmaze.variants import POINTMAZE_VARIANTS, get_pointmaze_variant_type
from model.policy import load_from_checkpoint
from transformers import LogitsProcessor, LogitsProcessorList
from utils.action_bins import (
    bin_to_continuous,
    get_action_bin_range,
    get_action_bin_codec,
    get_action_num_bins,
    get_action_token_mode,
)
from utils.chat_template import build_generation_prompt
from utils.eval_rollout import (
    build_action_rollout_context,
    generate_valid_action as rollout_generate_valid_action,
    render_policy_prompt,
    validate_history_config,
)
from utils.prompt_loader import load_named_templates, load_template_names, render_template
from utils.record_format import format_eval_step_text
from utils.variant_selection import get_available_variants, resolve_selection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="eval.yaml")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm strong warnings.")
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


def _load_checkpoint_config(model_path: str) -> dict:
    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


class _AllowedTokenIdsLogitsProcessor(LogitsProcessor):
    """Mask generation logits so only the provided token IDs remain valid."""

    def __init__(self, allowed_token_ids):
        token_ids = sorted({int(token_id) for token_id in allowed_token_ids})
        if not token_ids:
            raise ValueError("allowed_token_ids must contain at least one token ID")
        self.allowed_token_ids = tuple(token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        allowed = torch.tensor(self.allowed_token_ids, device=scores.device, dtype=torch.long)
        masked_scores = scores.new_full(scores.shape, -float("inf"))
        masked_scores[:, allowed] = scores[:, allowed]
        return masked_scores


def _resolve_action_generation_config(config: dict) -> dict:
    action_sampling = bool(config.get("action_sampling", False))
    action_temperature = float(config.get("action_temperature", 1.0))
    action_top_p = float(config.get("action_top_p", 1.0))
    action_top_k = int(config.get("action_top_k", 0))

    if action_temperature <= 0:
        raise ValueError(f"action_temperature must be > 0, got {action_temperature}")
    if action_top_p <= 0 or action_top_p > 1:
        raise ValueError(f"action_top_p must satisfy 0 < action_top_p <= 1, got {action_top_p}")
    if action_top_k < 0:
        raise ValueError(f"action_top_k must be >= 0, got {action_top_k}")

    return {
        "action_sampling": action_sampling,
        "action_temperature": action_temperature,
        "action_top_p": action_top_p,
        "action_top_k": action_top_k,
    }


def _load_checkpoint_action_config(model_path: str) -> dict:
    saved_config = _load_checkpoint_config(model_path)

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


def _normalize_prompt_name_list(value, *, field_name: str, allow_single_string: bool) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if not allow_single_string:
            raise ValueError(f"{field_name} must be a list of prompt names, got str")
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of prompt names, got {type(value).__name__}")

    names = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings, got {item!r}")
        names.append(item.strip())
    if not names:
        raise ValueError(f"{field_name} must not be empty")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{field_name} contains duplicate prompt names: {duplicates}")
    return names


def _resolve_prompt_config_values(config: dict, *, allow_single_string: bool) -> list[str] | None:
    primary_key = "prompt_templete_index"
    legacy_key = "prompt_template_index"
    primary_present = primary_key in config
    legacy_present = legacy_key in config
    primary_value = config.get(primary_key)
    legacy_value = config.get(legacy_key)
    if primary_present and legacy_present and primary_value != legacy_value:
        raise ValueError(
            f"{primary_key} and {legacy_key} both exist but differ; keep only {primary_key}."
        )
    if not primary_present and not legacy_present:
        return None
    field_name = primary_key if primary_present else legacy_key
    raw_value = primary_value if primary_present else legacy_value
    return _normalize_prompt_name_list(
        raw_value,
        field_name=field_name,
        allow_single_string=allow_single_string,
    )


def _confirm_strong_warning(message: str, *, assume_yes: bool):
    print(f"\n[eval] STRONG WARNING: {message}")
    if assume_yes:
        print("[eval] -y/--yes supplied; continuing despite the warning.")
        return
    try:
        response = input("[eval] Type uppercase Y to continue: ")
    except EOFError as exc:
        raise SystemExit("[eval] Aborted: confirmation required but stdin is unavailable.") from exc
    if response != "Y":
        raise SystemExit("[eval] Aborted by user.")


def apply_checkpoint_prompt_config(config: dict, *, assume_yes: bool) -> dict:
    saved_config = _load_checkpoint_config(config["model_path"])
    train_prompt_names = _resolve_prompt_config_values(saved_config, allow_single_string=False)
    eval_prompt_names = _resolve_prompt_config_values(config, allow_single_string=True)

    if eval_prompt_names is not None:
        if len(eval_prompt_names) != 1:
            raise ValueError(
                "Standalone eval accepts exactly one prompt in prompt_templete_index; "
                f"got {eval_prompt_names}."
            )
        eval_prompt_name = eval_prompt_names[0]
        if train_prompt_names and eval_prompt_name not in train_prompt_names:
            _confirm_strong_warning(
                "eval prompt is outside the checkpoint training prompt list. "
                f"train_prompts={train_prompt_names}, eval_prompt={eval_prompt_name!r}. "
                "This changes the prompt distribution used for rollout.",
                assume_yes=assume_yes,
            )
    elif train_prompt_names:
        eval_prompt_name = train_prompt_names[0]
    else:
        warnings.warn(
            "Checkpoint config.yaml does not contain prompt_templete_index; "
            "standalone eval will use the first prompt template in filename order.",
            RuntimeWarning,
            stacklevel=2,
        )
        eval_prompt_name = None

    merged = dict(config)
    if train_prompt_names:
        merged["checkpoint_prompt_templete_index"] = train_prompt_names
    if eval_prompt_name is not None:
        merged["prompt_templete_index"] = [eval_prompt_name]
    merged.pop("prompt_template_index", None)
    merged["resolved_eval_prompt_name"] = eval_prompt_name
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
    action_sampling: bool = False,
    action_temperature: float = 1.0,
    action_top_p: float = 1.0,
    action_top_k: int = 0,
    allowed_token_ids=None,
) -> tuple[str, list[int], tuple[torch.Tensor, ...] | None]:
    """Run inference and return display text plus generated action token IDs."""
    encoded = tokenizer(
        text=build_generation_prompt(tokenizer, prompt),
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    generate_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": action_sampling,
    }
    if action_sampling:
        generate_kwargs.update(
            {
                "temperature": action_temperature,
                "top_p": action_top_p,
                "top_k": action_top_k,
            }
        )
    if allowed_token_ids is not None:
        generate_kwargs["logits_processor"] = LogitsProcessorList(
            [_AllowedTokenIdsLogitsProcessor(allowed_token_ids)]
        )
    if eos_token_id is None:
        warnings.warn(
            "Tokenizer/model does not define eos_token_id; generation will stop only at max_new_tokens.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        generate_kwargs["eos_token_id"] = eos_token_id
        generate_kwargs["pad_token_id"] = eos_token_id
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


def _format_action_for_mode(formatter, action: np.ndarray, config: dict, action_codec=None) -> str:
    if get_action_token_mode(config) == "text":
        return formatter.format_action(action)
    if action_codec is None:
        raise RuntimeError("Action-bin display formatting requires an initialized action codec.")
    low, high = get_action_bin_range(config)
    return action_codec.display_text_for_action(action, low, high)


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
    low, high = get_action_bin_range(config)
    return action_codec.action_from_token_ids(token_ids, action_dim, low, high)



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
    history_num, history_stride = validate_history_config(config)

    env_kwargs.update(dict(config.get("env_kwargs") or {}))
    record_video = bool(config.get("record_video", False))
    video_episode_indices = _resolve_video_episode_indices(config, num_episodes)
    video_episode_index_set = set(video_episode_indices)
    video_fps = int(config.get("video_fps", 20))
    record_step_logs = bool(config.get("record_step_logs", True))
    action_token_mode = get_action_token_mode(config)
    collect_bin_probabilities = record_step_logs and action_token_mode != "text"

    if record_video:
        render_mode = env_kwargs.get("render_mode")
        if render_mode != "rgb_array":
            if render_mode is not None:
                print(
                    f"[eval] record_video=true: overriding env_kwargs.render_mode={render_mode!r} to 'rgb_array'"
                )
            env_kwargs["render_mode"] = "rgb_array"

    env = gym.make(env_id, **env_kwargs)
    action_dim = int(env.action_space.shape[0])
    action_context = build_action_rollout_context(
        config=config,
        tokenizer=tokenizer,
        action_dim=action_dim,
        collect_bin_probabilities=collect_bin_probabilities,
    )

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
            prompt = render_policy_prompt(
                formatter=formatter,
                template=template,
                prompt_vars=meta["prompt_vars"],
                obs=obs,
                history_buffer=history_buffer,
                history_num=history_num,
                history_stride=history_stride,
            )
            current_obs_vec = obs["observation"].astype(np.float32)
            action_result = rollout_generate_valid_action(
                model=model,
                tokenizer=tokenizer,
                device=device,
                formatter=formatter,
                prompt=prompt,
                config=config,
                action_context=action_context,
                action_shape=env.action_space.shape,
                action_dim=action_dim,
                parse_retry_limit=parse_retry_limit,
            )
            action = action_result.action
            executed_action_text = action_result.executed_action_text
            total_action_time += action_result.action_time_seconds
            total_actions += action_result.generation_count
            total_parse_failures += action_result.parse_failures
            total_fallbacks += action_result.fallback_count
            parse_status = action_result.parse_status
            attempt_count = action_result.attempt_count

            generated_text = "\n\n".join(
                f"[Attempt {idx + 1}]\n{text}" for idx, text in enumerate(action_result.generated_attempts)
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
                        "\n\n".join(action_result.generated_probability_logs)
                        if action_result.generated_probability_logs
                        else None
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
    config = apply_checkpoint_prompt_config(config, assume_yes=args.yes)

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
    prompt_name = config.get("resolved_eval_prompt_name")
    if prompt_name is None:
        prompt_name = load_template_names(env_family)[0]
        config["resolved_eval_prompt_name"] = prompt_name
    template = load_named_templates(env_family, [prompt_name])[0]
    config["eval_config_source"] = args.config
    config["standalone_eval_id"] = standalone_eval_id
    config["standalone_results_dir"] = run_results_dir
    config["resolved_eval_variants"] = eval_selection.selected_variants
    eval_config_path = os.path.join(run_results_dir, "eval_config.yaml")
    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"[eval] Eval config saved to: {eval_config_path}")

    for variant in eval_selection.selected_variants:
        print(f"\n[eval] Evaluating variant: {variant}")
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
        result["prompt_template_name"] = prompt_name
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
