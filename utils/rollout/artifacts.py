from __future__ import annotations

import os
import random

import numpy as np
import torch

from utils.record_format import format_eval_step_text
from utils.video_writer import VideoSaveManager


_ANTMAZE_GLOBAL_FLOOR_RGBA = np.array([0.18, 0.22, 0.26, 1.0], dtype=np.float64)


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


def _mujoco_render_owner(env, env_family: str | None):
    unwrapped = getattr(env, "unwrapped", env)
    candidates = []
    if env_family == "pointmaze":
        candidates.append(getattr(unwrapped, "point_env", None))
    elif env_family == "antmaze":
        candidates.append(getattr(unwrapped, "ant_env", None))
    candidates.extend([unwrapped, env])

    for owner in candidates:
        if owner is None:
            continue
        renderer = getattr(owner, "mujoco_renderer", None)
        if renderer is not None and hasattr(renderer, "_get_viewer"):
            return owner, renderer
    return None, None


def _camera_state(cam) -> dict:
    return {
        "type": cam.type,
        "fixedcamid": cam.fixedcamid,
        "lookat": np.array(cam.lookat, dtype=np.float64),
        "distance": cam.distance,
        "elevation": cam.elevation,
        "azimuth": cam.azimuth,
    }


def _restore_camera_state(cam, state: dict):
    cam.type = state["type"]
    cam.fixedcamid = state["fixedcamid"]
    cam.lookat[:] = state["lookat"]
    cam.distance = state["distance"]
    cam.elevation = state["elevation"]
    cam.azimuth = state["azimuth"]


def _maze_topdown_distance(
    env,
    owner,
    *,
    min_distance: float,
    fallback_extent_multiplier: float,
) -> float:
    unwrapped = getattr(env, "unwrapped", env)
    maze = getattr(unwrapped, "maze", None) or getattr(owner, "maze", None)
    if maze is not None:
        span_x = float(maze.map_width) * float(maze.maze_size_scaling)
        span_y = float(maze.map_length) * float(maze.maze_size_scaling)
        return max(float(min_distance), max(span_x, span_y) * 1.6)

    model = getattr(owner, "model", None)
    extent = float(getattr(getattr(model, "stat", None), "extent", 8.0))
    return max(float(min_distance), extent * float(fallback_extent_multiplier))


def _resolve_topdown_camera(
    env,
    owner,
    *,
    min_distance: float,
    fallback_extent_multiplier: float,
) -> dict:
    return {
        "lookat": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "distance": _maze_topdown_distance(
            env,
            owner,
            min_distance=min_distance,
            fallback_extent_multiplier=fallback_extent_multiplier,
        ),
        "elevation": -90.0,
        "azimuth": 90.0,
    }


def _capture_with_topdown_camera(env, frames: list[np.ndarray], renderer, camera: dict):
    viewer = renderer._get_viewer("rgb_array")
    cam = viewer.cam
    original_renderer_camera_id = getattr(renderer, "camera_id", None)
    original_cam_state = _camera_state(cam)

    try:
        renderer.camera_id = -1
        cam.lookat[:] = camera["lookat"]
        cam.distance = camera["distance"]
        cam.elevation = camera["elevation"]
        cam.azimuth = camera["azimuth"]
        frame = env.render()
        if frame is None:
            raise ValueError("render() returned None while recording top-down view")
        frames.append(normalize_render_frame(frame))
    finally:
        renderer.camera_id = original_renderer_camera_id
        _restore_camera_state(cam, original_cam_state)


def capture_pointmaze_render_frame(env, frames: list[np.ndarray]) -> bool:
    owner, renderer = _mujoco_render_owner(env, "pointmaze")
    if owner is None or renderer is None:
        return False
    camera = _resolve_topdown_camera(
        env,
        owner,
        min_distance=7.0,
        fallback_extent_multiplier=1.6,
    )
    _capture_with_topdown_camera(env, frames, renderer, camera)
    return True


def capture_render_frame(
    env,
    frames: list[np.ndarray],
    *,
    env_family: str | None = None,
):
    if env_family == "pointmaze" and capture_pointmaze_render_frame(env, frames):
        return

    frame = env.render()
    if frame is None:
        raise ValueError("render() returned None; use render_mode='rgb_array' when recording")
    frames.append(normalize_render_frame(frame))


def should_record_antmaze_global_video(config: dict) -> bool:
    return bool(config.get("record_video", False)) and config.get("env_family") == "antmaze"


def _resolve_antmaze_global_camera(env) -> dict:
    unwrapped = getattr(env, "unwrapped", env)
    ant_env = getattr(unwrapped, "ant_env", None)
    return _resolve_topdown_camera(
        env,
        ant_env,
        min_distance=14.0,
        fallback_extent_multiplier=3.0,
    )


def _geom_name(model, geom_id: int) -> str | None:
    geom = None
    geom_accessor = getattr(model, "geom", None)
    if callable(geom_accessor):
        try:
            geom = geom_accessor(geom_id)
        except (KeyError, IndexError, TypeError, ValueError):
            geom = None
    return getattr(geom, "name", None)


def _find_geom_id(model, geom_name: str) -> int | None:
    ngeom = int(getattr(model, "ngeom", 0))
    for geom_id in range(ngeom):
        if _geom_name(model, geom_id) == geom_name:
            return geom_id
    return None


def _set_antmaze_global_floor_color(ant_env) -> tuple[np.ndarray, int] | None:
    model = getattr(ant_env, "model", None)
    geom_rgba = getattr(model, "geom_rgba", None)
    if model is None or geom_rgba is None:
        return None

    floor_geom_id = _find_geom_id(model, "floor")
    if floor_geom_id is None:
        return None

    original_rgba = np.array(geom_rgba[floor_geom_id], dtype=np.float64)
    geom_rgba[floor_geom_id] = _ANTMAZE_GLOBAL_FLOOR_RGBA
    return original_rgba, floor_geom_id


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
    original_cam_state = _camera_state(cam)
    floor_state = _set_antmaze_global_floor_color(ant_env)

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
        if floor_state is not None:
            original_rgba, floor_geom_id = floor_state
            ant_env.model.geom_rgba[floor_geom_id] = original_rgba
        renderer.camera_id = original_renderer_camera_id
        _restore_camera_state(cam, original_cam_state)


def capture_video_frames(
    env,
    frames: list[np.ndarray],
    global_frames: list[np.ndarray] | None = None,
    *,
    env_family: str | None = None,
):
    capture_render_frame(env, frames, env_family=env_family)
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
