"""Evaluation entry point: rollout the fine-tuned policy in gymnasium environments.

Usage:
    python evaluate.py --config eval.yaml
"""

import argparse
import json
import os
import random
import time
import uuid
import warnings
from dataclasses import dataclass

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers maze environments
import numpy as np
import torch
import yaml

from data.registry import get_formatter, resolve_variant_env_spec
from model.continuous_action import (
    resolve_action_head_dropout,
    resolve_action_head_num_blocks,
    resolve_action_query_len,
    resolve_gaussian_log_std_bounds,
    resolve_gaussian_log_std_init,
    resolve_student_t_df,
)
from model.mtp_bin import resolve_mtp_k
from model.policy import load_from_checkpoint
from transformers import LogitsProcessor, LogitsProcessorList
from utils.action_bins import (
    bin_to_continuous,
    get_action_bin_range,
    get_action_bin_codec,
    get_action_num_bins,
    get_action_token_mode,
    uses_action_bins,
    uses_continuous_actions,
)
from utils.chat_template import build_generation_prompt
from utils.eval_rollout import (
    build_action_rollout_context,
    generate_valid_continuous_actions_batch,
    generate_valid_action as rollout_generate_valid_action,
    render_policy_prompt,
    validate_history_config,
)
from utils.distributed import (
    all_gather_objects,
    barrier,
    broadcast_object,
    cleanup_distributed,
    init_distributed_context,
    resolve_parallel_backend,
)
from utils.eval_parallel import (
    assigned_eval_variants,
    eval_variant_assignments,
    resolve_eval_distribute_variants,
    resolve_eval_parallel_episodes,
)
from utils.prompt_loader import load_named_templates, load_template_names, render_template
from utils.record_format import format_eval_step_text
from utils.variant_selection import get_available_variants, resolve_selection
from utils.video_writer import VideoSaveManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="eval.yaml")
    parser.add_argument("--parallel_backend", type=str, choices=["single", "ddp"], default=None)
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
    "mtp_k",
    "mtp_lcm_weight",
    "action_soft_label_sigma",
    "action_loss_weight",
    "action_stop_loss_weight",
    "action_dim",
    "action_query_len",
    "action_head_num_blocks",
    "action_head_dropout",
    "gaussian_log_std_min",
    "gaussian_log_std_max",
    "gaussian_log_std_init",
    "student_t_df",
    "max_length",
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
        "mtp_k": saved_config.get("mtp_k"),
        "mtp_lcm_weight": saved_config.get("mtp_lcm_weight", 1.0),
    }
    for key in ("action_soft_label_sigma", "action_loss_weight", "action_stop_loss_weight"):
        if key in saved_config:
            action_config[key] = saved_config[key]
    if "action_dim" in saved_config:
        action_config["action_dim"] = saved_config["action_dim"]
    if "max_length" in saved_config:
        action_config["max_length"] = saved_config["max_length"]
    if action_config["action_token_mode"] in {
        "mtp_bin",
        "simple_mtp_bin",
        "parallel_l1",
        "parallel_gaussian",
        "parallel_t",
    }:
        if "action_dim" not in action_config:
            raise ValueError(
                "Checkpoint config.yaml uses a parallel action mode but does not contain action_dim."
            )
    if action_config["action_token_mode"] == "mtp_bin":
        action_config["mtp_k"] = resolve_mtp_k(
            int(action_config["action_dim"]),
            saved_config.get("mtp_k"),
        )
    if action_config["action_token_mode"] == "simple_mtp_bin":
        action_config.pop("mtp_k", None)
    if action_config["action_token_mode"] in {"parallel_l1", "parallel_gaussian", "parallel_t"}:
        action_config["action_query_len"] = resolve_action_query_len(
            int(action_config["action_dim"]),
            saved_config.get("action_query_len"),
        )
        action_config["action_head_num_blocks"] = resolve_action_head_num_blocks(
            saved_config.get("action_head_num_blocks")
        )
        action_config["action_head_dropout"] = resolve_action_head_dropout(
            saved_config.get("action_head_dropout")
        )
        if action_config["action_token_mode"] in {"parallel_gaussian", "parallel_t"}:
            gaussian_log_std_min, gaussian_log_std_max = resolve_gaussian_log_std_bounds(
                saved_config
            )
            action_config["gaussian_log_std_min"] = gaussian_log_std_min
            action_config["gaussian_log_std_max"] = gaussian_log_std_max
        if action_config["action_token_mode"] == "parallel_gaussian":
            gaussian_log_std_init = resolve_gaussian_log_std_init(saved_config)
            action_config["gaussian_log_std_init"] = max(
                action_config["gaussian_log_std_min"],
                min(gaussian_log_std_init, action_config["gaussian_log_std_max"]),
            )
        if action_config["action_token_mode"] == "parallel_t":
            action_config["student_t_df"] = resolve_student_t_df(saved_config)
    return action_config


def apply_checkpoint_action_config(config: dict) -> dict:
    action_config = _load_checkpoint_action_config(config["model_path"])
    saved_config = _load_checkpoint_config(config["model_path"])
    for key in ACTION_CONFIG_KEYS:
        if key in config and config[key] != action_config.get(key):
            raise ValueError(
                f"Standalone eval action config must come from checkpoint config.yaml; "
                f"remove {key}={config[key]!r} from eval.yaml or match checkpoint value {action_config.get(key)!r}."
            )
    merged = dict(config)
    merged.update(action_config)
    if "mtp_quadratic_decoding" not in merged:
        merged["mtp_quadratic_decoding"] = saved_config.get("mtp_quadratic_decoding", True)
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


def resolve_eval_output_mode(config: dict) -> str:
    mode = str(config.get("eval_output_mode", "standalone")).strip().lower()
    if mode not in {"standalone", "training"}:
        raise ValueError(
            "eval_output_mode must be 'standalone' or 'training', "
            f"got {mode!r}"
        )
    return mode


TRAINING_EVAL_CONTEXT_KEYS = (
    "eval_type",
    "epoch",
    "batch_step",
    "epoch_step",
    "optimizer_step",
    "scheduled_step",
    "scheduled_epoch_step",
    "train_loss",
    "val_loss",
    "val_metrics",
    "checkpoint_path",
    "experiment_id",
)


def resolve_training_eval_context(config: dict) -> dict:
    context = config.get("training_eval_context")
    if not isinstance(context, dict):
        raise ValueError(
            "training_eval_context must be provided as a mapping when "
            "eval_output_mode='training'"
        )

    missing = [key for key in TRAINING_EVAL_CONTEXT_KEYS if key not in context]
    if missing:
        raise ValueError(f"training_eval_context is missing required keys: {missing}")

    eval_type = context.get("eval_type")
    if eval_type not in {"epoch", "step"}:
        raise ValueError(
            "training_eval_context.eval_type must be 'epoch' or 'step', "
            f"got {eval_type!r}"
        )
    if eval_type == "epoch" and context.get("epoch") is None:
        raise ValueError("training_eval_context.epoch is required for epoch eval")
    if eval_type == "step" and context.get("batch_step") is None:
        raise ValueError("training_eval_context.batch_step is required for step eval")

    resolved = dict(context)
    if not isinstance(resolved.get("val_metrics"), dict):
        raise ValueError("training_eval_context.val_metrics must be a mapping")
    return resolved


def get_training_eval_tag(training_eval_context: dict) -> str:
    if training_eval_context["eval_type"] == "step":
        return f"step{training_eval_context['batch_step']}"
    return f"epoch_{training_eval_context['epoch']}"


def get_training_results_dir(base_results_dir: str, training_eval_context: dict) -> str:
    return os.path.join(base_results_dir, get_training_eval_tag(training_eval_context))


def apply_training_eval_context_to_result(
    result: dict,
    training_eval_context: dict,
) -> dict:
    val_metrics = training_eval_context.get("val_metrics") or {}
    result["train_loss"] = training_eval_context.get("train_loss")
    result["val_loss"] = training_eval_context.get("val_loss")
    result["val_metrics"] = val_metrics
    if "mae" in val_metrics:
        result["val_mae"] = val_metrics["mae"]
    result["experiment_id"] = training_eval_context.get("experiment_id")
    result["eval_type"] = training_eval_context.get("eval_type")
    result["eval_tag"] = get_training_eval_tag(training_eval_context)
    result["epoch"] = training_eval_context.get("epoch")
    result["batch_step"] = training_eval_context.get("batch_step")
    result["epoch_step"] = training_eval_context.get("epoch_step")
    result["optimizer_step"] = training_eval_context.get("optimizer_step")
    result["scheduled_step"] = training_eval_context.get("scheduled_step")
    result["scheduled_epoch_step"] = training_eval_context.get("scheduled_epoch_step")
    result["checkpoint_path"] = training_eval_context.get("checkpoint_path")
    return result


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
    raw_continuous_action: list[float] | None = None,
    gaussian_action_mean: list[float] | None = None,
    gaussian_action_std: list[float] | None = None,
    student_t_action_mean: list[float] | None = None,
    student_t_action_scale: list[float] | None = None,
):
    os.makedirs(episode_dir, exist_ok=True)
    step_path = os.path.join(episode_dir, "steps.txt")
    payload = format_eval_step_text(
        prompt,
        action_text,
        executed_action=executed_action,
        parse_status=parse_status,
        attempt_count=attempt_count,
        action_bin_probabilities=action_bin_probabilities,
        raw_continuous_action=raw_continuous_action,
        gaussian_action_mean=gaussian_action_mean,
        gaussian_action_std=gaussian_action_std,
        student_t_action_mean=student_t_action_mean,
        student_t_action_scale=student_t_action_scale,
    )
    separator = "=" * 80
    step_payload = (
        f"{separator}\n"
        f"Step {step_index + 1:04d}\n"
        f"{separator}\n"
        f"{payload.rstrip()}\n"
    )
    mode = "w" if step_index == 0 else "a"
    with open(step_path, mode, encoding="utf-8") as f:
        if step_index > 0:
            f.write("\n")
        f.write(step_payload)



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


def _should_record_antmaze_global_video(config: dict) -> bool:
    return bool(config.get("record_video", False)) and config.get("env_family") == "antmaze"


def _resolve_antmaze_global_camera(env) -> dict:
    unwrapped = getattr(env, "unwrapped", env)
    maze = getattr(unwrapped, "maze", None)
    if maze is not None:
        span_x = float(maze.map_width) * float(maze.maze_size_scaling)
        span_y = float(maze.map_length) * float(maze.maze_size_scaling)
        distance = max(14.0, max(span_x, span_y) * 1.6)
    else:
        ant_env = getattr(unwrapped, "ant_env", None)
        model = getattr(ant_env, "model", None)
        extent = float(getattr(getattr(model, "stat", None), "extent", 8.0))
        distance = max(14.0, extent * 3.0)

    return {
        "lookat": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "distance": distance,
        "elevation": -90.0,
        "azimuth": 90.0,
    }


def _capture_antmaze_global_render_frame(env, frames: list[np.ndarray]):
    unwrapped = getattr(env, "unwrapped", env)
    ant_env = getattr(unwrapped, "ant_env", None)
    renderer = getattr(ant_env, "mujoco_renderer", None)
    if renderer is None or not hasattr(renderer, "_get_viewer"):
        raise ValueError("AntMaze global video recording requires a MuJoCo renderer")

    viewer = renderer._get_viewer("rgb_array")
    cam = viewer.cam
    camera = _resolve_antmaze_global_camera(env)
    original_renderer_camera_id = getattr(renderer, "camera_id", None)
    original_cam_state = {
        "type": cam.type,
        "fixedcamid": cam.fixedcamid,
        "lookat": np.array(cam.lookat, dtype=np.float64),
        "distance": cam.distance,
        "elevation": cam.elevation,
        "azimuth": cam.azimuth,
    }

    try:
        renderer.camera_id = -1
        cam.lookat[:] = camera["lookat"]
        cam.distance = camera["distance"]
        cam.elevation = camera["elevation"]
        cam.azimuth = camera["azimuth"]
        frame = env.render()
        if frame is None:
            raise ValueError("render() returned None while recording AntMaze global view")
        frames.append(_normalize_render_frame(frame))
    finally:
        renderer.camera_id = original_renderer_camera_id
        cam.type = original_cam_state["type"]
        cam.fixedcamid = original_cam_state["fixedcamid"]
        cam.lookat[:] = original_cam_state["lookat"]
        cam.distance = original_cam_state["distance"]
        cam.elevation = original_cam_state["elevation"]
        cam.azimuth = original_cam_state["azimuth"]


def _capture_video_frames(
    env,
    frames: list[np.ndarray],
    global_frames: list[np.ndarray] | None = None,
):
    _capture_render_frame(env, frames)
    if global_frames is not None:
        _capture_antmaze_global_render_frame(env, global_frames)


def _record_episode_videos(
    *,
    video_saver: VideoSaveManager,
    frames: list[np.ndarray],
    global_frames: list[np.ndarray] | None,
    episode_dir: str,
    video_ext: str,
    video_fps: int,
) -> tuple[str, str | None]:
    video_path = os.path.join(episode_dir, f"rollout.{video_ext}")
    video_saver.submit(frames, video_path, video_fps)

    global_video_path = None
    if global_frames is not None:
        global_video_path = os.path.join(episode_dir, f"rollout_global.{video_ext}")
        video_saver.submit(global_frames, global_video_path, video_fps)

    return video_path, global_video_path



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
    if not uses_action_bins(config):
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


def _resolve_eval_seed(config: dict) -> int:
    seed = int(config.get("seed", 1))
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return seed


def _resolve_episode_seeds(config: dict, num_episodes: int) -> tuple[int, list[int]]:
    eval_seed = _resolve_eval_seed(config)
    episode_seeds = [eval_seed + ep_idx for ep_idx in range(num_episodes)]
    max_seed = 2**32 - 1
    if episode_seeds and episode_seeds[-1] > max_seed:
        raise ValueError(
            "episode seeds must fit numpy/gymnasium seed range; "
            f"got last seed {episode_seeds[-1]} > {max_seed}"
        )
    return eval_seed, episode_seeds


def _set_episode_rng_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_mujoco_gl(config: dict):
    mujoco_gl = config.get("mujoco_gl")
    if mujoco_gl:
        os.environ["MUJOCO_GL"] = str(mujoco_gl)
        return

    if config.get("record_video", False):
        os.environ.setdefault("MUJOCO_GL", "egl")


def _resolve_variant_env_spec(config: dict, variant: str) -> tuple[dict, str, dict]:
    meta, env_id, env_kwargs = resolve_variant_env_spec(
        config["env_family"],
        variant,
    )
    env_kwargs.update(dict(config.get("env_kwargs") or {}))
    return meta, env_id, env_kwargs


def _history_observation(formatter, obs) -> np.ndarray:
    if hasattr(formatter, "format_history_observation"):
        return np.asarray(formatter.format_history_observation(obs), dtype=np.float32)
    return np.asarray(obs["observation"], dtype=np.float32)


def _prepare_eval_prompt_vars(formatter, prompt_vars: dict, env) -> dict:
    if hasattr(formatter, "prepare_eval_prompt_vars"):
        return formatter.prepare_eval_prompt_vars(prompt_vars, env)
    return prompt_vars


def _prepare_video_env_kwargs(config: dict, env_kwargs: dict) -> dict:
    resolved = dict(env_kwargs)
    if not bool(config.get("record_video", False)):
        return resolved
    render_mode = resolved.get("render_mode")
    if render_mode != "rgb_array":
        if render_mode is not None:
            print(
                "[eval] record_video=true: overriding "
                f"env_kwargs.render_mode={render_mode!r} to 'rgb_array'"
            )
        resolved["render_mode"] = "rgb_array"
    return resolved


def _validate_eval_action_dim(config: dict, variant: str, env) -> int:
    action_dim = int(env.action_space.shape[0])
    checkpoint_action_dim = config.get("action_dim")
    if checkpoint_action_dim is not None and int(checkpoint_action_dim) != action_dim:
        raise ValueError(
            "Checkpoint action_dim does not match evaluation env action space: "
            f"checkpoint={checkpoint_action_dim}, env={action_dim}, variant={variant}"
        )
    return action_dim


@dataclass
class _BatchedEpisodeState:
    env: object
    episode_index: int
    obs: object
    history_buffer: list[dict]
    frames: list[np.ndarray] | None
    global_frames: list[np.ndarray] | None
    episode_dir: str | None
    episode_return: float = 0.0
    episode_success: bool = False
    episode_steps: int = 0


def _evaluate_variant_continuous_batched(
    config: dict,
    variant: str,
    model,
    tokenizer,
    device: torch.device,
    template: str,
    variant_results_dir: str | None,
    parallel_episodes: int,
) -> dict:
    formatter = get_formatter(config["env_family"])
    meta, env_id, env_kwargs = _resolve_variant_env_spec(config, variant)
    env_kwargs = _prepare_video_env_kwargs(config, env_kwargs)
    num_episodes = int(config["num_episodes"])
    history_num, history_stride = validate_history_config(config)
    eval_seed, episode_seeds = _resolve_episode_seeds(config, num_episodes)
    record_video = bool(config.get("record_video", False))
    record_global_video = _should_record_antmaze_global_video(config)
    video_episode_index_set = set(_resolve_video_episode_indices(config, num_episodes))
    video_fps = int(config.get("video_fps", 20))
    record_step_logs = bool(config.get("record_step_logs", True))
    active_parallel_episodes = min(parallel_episodes, num_episodes)

    envs = []
    try:
        for _ in range(active_parallel_episodes):
            envs.append(gym.make(env_id, **env_kwargs))
    except Exception:
        for env in envs:
            env.close()
        raise

    episode_returns = [0.0] * num_episodes
    episode_successes = [False] * num_episodes
    episode_steps = [0] * num_episodes
    episode_artifact_dirs: list[str | None] = [None] * num_episodes
    saved_video_paths: list[str | None] = [None] * num_episodes
    saved_global_video_paths: list[str | None] = [None] * num_episodes
    total_parse_failures = 0
    total_fallbacks = 0
    total_action_time = 0.0
    total_actions = 0
    video_saver = VideoSaveManager(config)

    try:
        action_dim = _validate_eval_action_dim(config, variant, envs[0])
        prompt_vars = _prepare_eval_prompt_vars(
            formatter,
            meta["prompt_vars"],
            envs[0],
        )
        action_shape = envs[0].action_space.shape
        action_low = getattr(envs[0].action_space, "low", None)
        action_high = getattr(envs[0].action_space, "high", None)
        for env in envs[1:]:
            if env.action_space.shape != action_shape:
                raise ValueError(
                    "Parallel eval environments must share one action shape: "
                    f"expected {action_shape}, got {env.action_space.shape}"
                )

        action_context = build_action_rollout_context(
            config=config,
            tokenizer=tokenizer,
            action_dim=action_dim,
            collect_bin_probabilities=False,
        )
        _set_episode_rng_seed(eval_seed)

        def start_episode(env, episode_index: int) -> _BatchedEpisodeState:
            episode_seed = episode_seeds[episode_index]
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            record_this_episode = (
                record_video and episode_index in video_episode_index_set
            )
            frames = [] if record_this_episode else None
            global_frames = [] if record_this_episode and record_global_video else None
            episode_dir = None
            if variant_results_dir is not None:
                episode_dir = get_episode_dir(variant_results_dir, episode_index)
                os.makedirs(episode_dir, exist_ok=True)
                episode_artifact_dirs[episode_index] = episode_dir
            if frames is not None:
                _capture_video_frames(env, frames, global_frames)
            return _BatchedEpisodeState(
                env=env,
                episode_index=episode_index,
                obs=obs,
                history_buffer=[],
                frames=frames,
                global_frames=global_frames,
                episode_dir=episode_dir,
            )

        next_episode_index = 0
        active_states = []
        for env in envs:
            active_states.append(start_episode(env, next_episode_index))
            next_episode_index += 1

        while active_states:
            prompts = [
                render_policy_prompt(
                    formatter=formatter,
                    template=template,
                    prompt_vars=prompt_vars,
                    obs=state.obs,
                    history_buffer=state.history_buffer,
                    history_num=history_num,
                    history_stride=history_stride,
                )
                for state in active_states
            ]
            current_history_observations = [
                _history_observation(formatter, state.obs)
                for state in active_states
            ]
            action_results = generate_valid_continuous_actions_batch(
                model=model,
                tokenizer=tokenizer,
                device=device,
                formatter=formatter,
                prompts=prompts,
                config=config,
                action_context=action_context,
                action_shape=action_shape,
                action_dim=action_dim,
                action_low=action_low,
                action_high=action_high,
            )

            next_active_states = []
            for state, prompt, current_history_observation, action_result in zip(
                active_states,
                prompts,
                current_history_observations,
                action_results,
                strict=True,
            ):
                total_action_time += action_result.action_time_seconds
                total_actions += action_result.generation_count
                total_parse_failures += action_result.parse_failures
                total_fallbacks += action_result.fallback_count
                generated_text = "\n\n".join(
                    f"[Attempt {idx + 1}]\n{text}"
                    for idx, text in enumerate(action_result.generated_attempts)
                )

                if record_step_logs and state.episode_dir is not None:
                    write_step_log(
                        state.episode_dir,
                        state.episode_steps,
                        prompt=prompt,
                        action_text=generated_text,
                        executed_action=action_result.executed_action_text,
                        parse_status=action_result.parse_status,
                        attempt_count=action_result.attempt_count,
                        raw_continuous_action=action_result.raw_continuous_action,
                        gaussian_action_mean=action_result.gaussian_action_mean,
                        gaussian_action_std=action_result.gaussian_action_std,
                        student_t_action_mean=action_result.student_t_action_mean,
                        student_t_action_scale=action_result.student_t_action_scale,
                    )

                obs, reward, terminated, truncated, info = state.env.step(
                    action_result.action
                )
                state.history_buffer.append(
                    {
                        "observation": current_history_observation,
                        "action_text": action_result.executed_action_text,
                    }
                )
                state.obs = obs
                state.episode_return += float(reward)
                state.episode_steps += 1
                if bool(info.get("success", terminated)):
                    state.episode_success = True
                if state.frames is not None:
                    _capture_video_frames(
                        state.env,
                        state.frames,
                        state.global_frames,
                    )

                if not (terminated or truncated):
                    next_active_states.append(state)
                    continue

                episode_index = state.episode_index
                episode_returns[episode_index] = state.episode_return
                episode_successes[episode_index] = state.episode_success
                episode_steps[episode_index] = state.episode_steps
                if state.frames is not None and state.episode_dir is not None:
                    video_ext = str(config.get("video_format", "gif")).lstrip(".")
                    video_path, global_video_path = _record_episode_videos(
                        video_saver=video_saver,
                        frames=state.frames,
                        global_frames=state.global_frames,
                        episode_dir=state.episode_dir,
                        video_ext=video_ext,
                        video_fps=video_fps,
                    )
                    saved_video_paths[episode_index] = video_path
                    saved_global_video_paths[episode_index] = global_video_path

                if next_episode_index < num_episodes:
                    next_active_states.append(
                        start_episode(state.env, next_episode_index)
                    )
                    next_episode_index += 1

            active_states = next_active_states
    finally:
        for env in envs:
            env.close()
        video_saver.close()

    resolved_video_paths = [path for path in saved_video_paths if path is not None]
    resolved_global_video_paths = [
        path for path in saved_global_video_paths if path is not None
    ]
    resolved_artifact_dirs = [
        path for path in episode_artifact_dirs if path is not None
    ]
    mean_action_time_ms = (
        total_action_time / total_actions * 1000
        if total_actions > 0
        else 0.0
    )
    return {
        "variant": variant,
        "num_episodes": num_episodes,
        "seed": eval_seed,
        "episode_seeds": episode_seeds,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "success_rate": float(np.mean(episode_successes)),
        "mean_episode_steps": float(np.mean(episode_steps)),
        "std_episode_steps": float(np.std(episode_steps)),
        "total_parse_failures": total_parse_failures,
        "total_fallbacks": total_fallbacks,
        "mean_action_time_ms": round(mean_action_time_ms, 2),
        "video_path": (
            resolved_video_paths[0]
            if len(resolved_video_paths) == 1
            else None
        ),
        "video_paths": resolved_video_paths,
        "global_video_path": (
            resolved_global_video_paths[0]
            if len(resolved_global_video_paths) == 1
            else None
        ),
        "global_video_paths": resolved_global_video_paths,
        "all_video_paths": resolved_video_paths + resolved_global_video_paths,
        "episode_artifact_dirs": resolved_artifact_dirs,
        "episode_artifacts_dir": variant_results_dir,
        "eval_parallel_episodes_requested": parallel_episodes,
        "eval_parallel_episodes_used": active_parallel_episodes,
        "video_save_workers": video_saver.workers,
        "video_save_max_pending": video_saver.max_pending,
    }



def evaluate_variant(
    config: dict,
    variant: str,
    model,
    tokenizer,
    device: torch.device,
    template: str,
    variant_results_dir: str | None = None,
) -> dict:
    num_episodes = int(config["num_episodes"])
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    parallel_episodes = resolve_eval_parallel_episodes(config)
    if parallel_episodes > 1 and uses_continuous_actions(config):
        return _evaluate_variant_continuous_batched(
            config,
            variant,
            model,
            tokenizer,
            device,
            template,
            variant_results_dir,
            parallel_episodes,
        )
    if parallel_episodes > 1:
        print(
            f"[eval] action_token_mode={get_action_token_mode(config)!r} does not "
            "support episode batching yet; falling back to serial rollout."
        )

    formatter = get_formatter(config["env_family"])
    meta, env_id, env_kwargs = _resolve_variant_env_spec(config, variant)
    num_episodes = int(config["num_episodes"])
    parse_retry_limit = config["parse_retry_limit"]
    history_num, history_stride = validate_history_config(config)
    eval_seed, episode_seeds = _resolve_episode_seeds(config, num_episodes)

    record_video = bool(config.get("record_video", False))
    video_episode_indices = _resolve_video_episode_indices(config, num_episodes)
    video_episode_index_set = set(video_episode_indices)
    video_fps = int(config.get("video_fps", 20))
    record_step_logs = bool(config.get("record_step_logs", True))
    get_action_token_mode(config)
    collect_bin_probabilities = record_step_logs and uses_action_bins(config)

    env_kwargs = _prepare_video_env_kwargs(config, env_kwargs)
    record_global_video = _should_record_antmaze_global_video(config)

    env = gym.make(env_id, **env_kwargs)
    action_dim = _validate_eval_action_dim(config, variant, env)
    prompt_vars = _prepare_eval_prompt_vars(formatter, meta["prompt_vars"], env)
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
    saved_global_video_paths = []
    episode_artifact_dirs = []
    video_saver = VideoSaveManager(config)

    for ep_idx in range(num_episodes):
        episode_seed = episode_seeds[ep_idx]
        _set_episode_rng_seed(episode_seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(episode_seed)
        obs, info = env.reset(seed=episode_seed)
        history_buffer = []
        record_this_episode = record_video and ep_idx in video_episode_index_set
        episode_frames = [] if record_this_episode else None
        episode_global_frames = [] if record_this_episode and record_global_video else None
        episode_dir = None

        if variant_results_dir is not None:
            episode_dir = get_episode_dir(variant_results_dir, ep_idx)
            os.makedirs(episode_dir, exist_ok=True)
            episode_artifact_dirs.append(episode_dir)

        if episode_frames is not None:
            _capture_video_frames(env, episode_frames, episode_global_frames)

        ep_return = 0.0
        ep_success = False
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
            current_history_observation = _history_observation(formatter, obs)
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
                action_low=getattr(env.action_space, "low", None),
                action_high=getattr(env.action_space, "high", None),
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
                    raw_continuous_action=action_result.raw_continuous_action,
                    gaussian_action_mean=action_result.gaussian_action_mean,
                    gaussian_action_std=action_result.gaussian_action_std,
                    student_t_action_mean=action_result.student_t_action_mean,
                    student_t_action_scale=action_result.student_t_action_scale,
                )

            obs, reward, terminated, truncated, info = env.step(action)
            history_buffer.append(
                {
                    "observation": current_history_observation,
                    "action_text": executed_action_text,
                }
            )

            if episode_frames is not None:
                _capture_video_frames(env, episode_frames, episode_global_frames)

            ep_return += float(reward)
            ep_steps += 1

            if bool(info.get("success", terminated)):
                ep_success = True

        if episode_frames is not None and episode_dir is not None:
            video_ext = str(config.get("video_format", "gif")).lstrip(".")
            video_path, global_video_path = _record_episode_videos(
                video_saver=video_saver,
                frames=episode_frames,
                global_frames=episode_global_frames,
                episode_dir=episode_dir,
                video_ext=video_ext,
                video_fps=video_fps,
            )
            saved_video_paths.append(video_path)
            if global_video_path is not None:
                saved_global_video_paths.append(global_video_path)
            status = "queued video save" if video_saver.asynchronous else "saved video"
            print(f"  [{variant}] {status}: {video_path}")
            if global_video_path is not None:
                print(f"  [{variant}] {status}: {global_video_path}")

        episode_returns.append(ep_return)
        episode_successes.append(ep_success)
        episode_steps.append(ep_steps)

        if (ep_idx + 1) % 5 == 0 or record_this_episode:
            print(
                f"  [{variant}] episode {ep_idx+1}/{num_episodes} | "
                f"return={ep_return:.2f} | steps={ep_steps} | success={ep_success}"
            )

    env.close()
    video_saver.close()

    mean_action_time_ms = (total_action_time / total_actions * 1000) if total_actions > 0 else 0.0
    return {
        "variant": variant,
        "num_episodes": num_episodes,
        "seed": eval_seed,
        "episode_seeds": episode_seeds,
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
        "global_video_path": (
            saved_global_video_paths[0]
            if len(saved_global_video_paths) == 1
            else None
        ),
        "global_video_paths": saved_global_video_paths,
        "all_video_paths": saved_video_paths + saved_global_video_paths,
        "episode_artifact_dirs": episode_artifact_dirs,
        "episode_artifacts_dir": variant_results_dir,
        "eval_parallel_episodes_requested": parallel_episodes,
        "eval_parallel_episodes_used": 1,
        "video_save_workers": video_saver.workers,
        "video_save_max_pending": video_saver.max_pending,
    }



def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    parallel_backend = resolve_parallel_backend(config, args.parallel_backend)
    dist_context = init_distributed_context(config, parallel_backend)
    try:
        if dist_context.is_main_process:
            config = apply_checkpoint_action_config(config)
            config = apply_checkpoint_prompt_config(config, assume_yes=args.yes)
            config.setdefault("seed", 1)
        else:
            config = None
        config = broadcast_object(config, dist_context)

        eval_output_mode = resolve_eval_output_mode(config)
        training_eval_context = (
            resolve_training_eval_context(config)
            if eval_output_mode == "training"
            else None
        )
        eval_selection = resolve_standalone_eval_selection(config)
        distribute_variants = resolve_eval_distribute_variants(config)
        parallel_episodes = resolve_eval_parallel_episodes(config)
        assignments = eval_variant_assignments(
            eval_selection.selected_variants,
            dist_context,
            distribute_variants=distribute_variants,
        )
        local_variants = assigned_eval_variants(
            eval_selection.selected_variants,
            dist_context,
            distribute_variants=distribute_variants,
        )
        configure_mujoco_gl(config)

        device = dist_context.device
        if dist_context.is_main_process:
            print(f"[eval] Using backend: {dist_context.backend}")
            print(f"[eval] Output mode: {eval_output_mode}")
            print(f"[eval] Loading model from: {config['model_path']}")
            print(f"[eval] Resolved eval variants: {eval_selection.selected_variants}")
            print(f"[eval] Variant assignments: {assignments}")
            print(f"[eval] Parallel episodes per rank: {parallel_episodes}")

        model, tokenizer = load_from_checkpoint(
            config["model_path"],
            load_in_4bit=config.get("load_in_4bit"),
        )
        model.to(device)
        model.eval()

        env_family = config["env_family"]
        base_results_dir = get_results_base_dir(config)
        standalone_eval_id = None
        if eval_output_mode == "standalone":
            standalone_eval_id = (
                uuid.uuid4().hex[:8]
                if dist_context.is_main_process
                else None
            )
            standalone_eval_id = broadcast_object(
                standalone_eval_id,
                dist_context,
            )
            run_results_dir = get_standalone_results_dir(
                base_results_dir,
                standalone_eval_id,
            )
            if dist_context.is_main_process:
                print(f"[eval] Eval ID: {standalone_eval_id}")
        else:
            run_results_dir = get_training_results_dir(
                base_results_dir,
                training_eval_context,
            )
            if dist_context.is_main_process:
                print(
                    f"[eval] Training eval tag: "
                    f"{get_training_eval_tag(training_eval_context)}"
                )
        prompt_name = config.get("resolved_eval_prompt_name")
        if prompt_name is None:
            prompt_name = load_template_names(env_family)[0]
            config["resolved_eval_prompt_name"] = prompt_name
        template = load_named_templates(env_family, [prompt_name])[0]
        config["eval_config_source"] = args.config
        config["eval_output_mode"] = eval_output_mode
        if eval_output_mode == "standalone":
            config["standalone_eval_id"] = standalone_eval_id
            config["standalone_results_dir"] = run_results_dir
        else:
            config["training_eval_context"] = training_eval_context
            config["training_eval_tag"] = get_training_eval_tag(training_eval_context)
            config["training_results_dir"] = run_results_dir
        config["resolved_eval_variants"] = eval_selection.selected_variants
        config["resolved_eval_variant_assignments"] = assignments
        config["eval_world_size"] = (
            training_eval_context.get("eval_world_size", dist_context.world_size)
            if training_eval_context is not None
            else dist_context.world_size
        )
        config["eval_parallel_episodes"] = parallel_episodes
        config["eval_distribute_variants"] = distribute_variants
        eval_config_path = os.path.join(run_results_dir, "eval_config.yaml")
        if dist_context.is_main_process:
            os.makedirs(run_results_dir, exist_ok=True)
            with open(eval_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
            print(f"[eval] Eval config saved to: {eval_config_path}")
        barrier(dist_context)

        local_results = []
        for variant in local_variants:
            print(
                f"\n[eval][rank {dist_context.rank}] "
                f"Evaluating variant: {variant}"
            )
            results_dir = get_variant_results_dir(
                run_results_dir,
                env_family,
                variant,
            )
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
            if training_eval_context is not None:
                apply_training_eval_context_to_result(result, training_eval_context)
            result["eval_rank"] = (
                training_eval_context.get("eval_rank", dist_context.rank)
                if training_eval_context is not None
                else dist_context.rank
            )
            result["eval_world_size"] = (
                training_eval_context.get("eval_world_size", dist_context.world_size)
                if training_eval_context is not None
                else dist_context.world_size
            )
            result["eval_distribute_variants"] = (
                training_eval_context.get("eval_distribute_variants", distribute_variants)
                if training_eval_context is not None
                else distribute_variants
            )
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(
                f"[eval][rank {dist_context.rank}] {variant}: "
                f"mean_return={result['mean_return']:.4f}, "
                f"success_rate={result['success_rate']:.2%}, "
                f"parse_failures={result['total_parse_failures']}, "
                f"fallbacks={result['total_fallbacks']}"
            )
            print(
                f"[eval][rank {dist_context.rank}] "
                f"Results saved to: {result_path}"
            )
            local_results.append(result)

        gathered_results = all_gather_objects(local_results, dist_context)
        if dist_context.is_main_process:
            results_by_variant = {
                result["variant"]: result
                for rank_results in gathered_results
                for result in rank_results
            }
            print("\n[eval] Completed variants:")
            for variant in eval_selection.selected_variants:
                result = results_by_variant[variant]
                print(
                    f"  {variant}: success_rate={result['success_rate']:.2%}, "
                    f"mean_return={result['mean_return']:.4f}, "
                    f"rank={result['eval_rank']}"
                )
    finally:
        cleanup_distributed(dist_context)


if __name__ == "__main__":
    main()
