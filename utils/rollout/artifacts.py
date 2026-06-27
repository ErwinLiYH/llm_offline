from __future__ import annotations

import os
import random

import numpy as np
import torch

from utils.record_format import format_eval_step_text
from utils.video_writer import VideoSaveManager


def set_episode_rng_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_render_frame(frame) -> np.ndarray:
    if isinstance(frame, (list, tuple)):
        if not frame:
            raise ValueError("Environment render() returned an empty frame list")
        frame = frame[-1]

    arr = np.asarray(frame)
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unsupported rendered frame shape: {arr.shape}")
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported rendered frame shape: {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if float(arr.max(initial=0.0)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def capture_render_frame(env, frames: list[np.ndarray]):
    frame = env.render()
    if frame is None:
        raise ValueError("render() returned None; use render_mode='rgb_array' when recording")
    frames.append(normalize_render_frame(frame))


def should_record_antmaze_global_video(config: dict) -> bool:
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


def capture_antmaze_global_render_frame(env, frames: list[np.ndarray]):
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
        frames.append(normalize_render_frame(frame))
    finally:
        renderer.camera_id = original_renderer_camera_id
        cam.type = original_cam_state["type"]
        cam.fixedcamid = original_cam_state["fixedcamid"]
        cam.lookat[:] = original_cam_state["lookat"]
        cam.distance = original_cam_state["distance"]
        cam.elevation = original_cam_state["elevation"]
        cam.azimuth = original_cam_state["azimuth"]


def capture_video_frames(
    env,
    frames: list[np.ndarray],
    global_frames: list[np.ndarray] | None = None,
):
    capture_render_frame(env, frames)
    if global_frames is not None:
        capture_antmaze_global_render_frame(env, global_frames)


def record_episode_videos(
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

