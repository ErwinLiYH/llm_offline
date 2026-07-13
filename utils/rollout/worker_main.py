from __future__ import annotations

import os
import queue
import sys
import traceback
import uuid

import gymnasium_robotics  # noqa: F401 registers maze environments
import numpy as np

from crossmaze.eval_position import select_eval_position
from crossmaze import make as crossmaze_make
from data.pointmaze.variants import POINTMAZE_VARIANTS
from data.registry import get_formatter, get_variant
from utils.eval_rollout import render_policy_prompt, validate_history_config
from utils.rollout.artifacts import (
    capture_render_frame,
    capture_video_frames,
    record_episode_videos,
    set_episode_rng_seed,
    should_record_antmaze_global_video,
    write_step_log,
)
from utils.rollout.protocol import ActionRequest, ActionResponse, EpisodeResult
from utils.sensing_config import apply_sensing_config_to_prompt_vars
from utils.video_writer import VideoSaveManager


def _prepare_eval_prompt_vars(formatter, prompt_vars: dict, env) -> dict:
    if hasattr(formatter, "prepare_eval_prompt_vars"):
        return formatter.prepare_eval_prompt_vars(prompt_vars, env)
    return prompt_vars


def _history_observation(formatter, obs) -> np.ndarray:
    if hasattr(formatter, "format_history_observation"):
        return np.asarray(formatter.format_history_observation(obs), dtype=np.float32)
    return np.asarray(obs["observation"], dtype=np.float32)


def _episode_dir(variant_results_dir: str | None, mode: str, episode_index: int) -> str | None:
    if variant_results_dir is None:
        return None
    if mode == "score":
        return os.path.join(variant_results_dir, f"episode_{episode_index}")
    return os.path.join(variant_results_dir, f"episode_{episode_index + 1:04d}")


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
                f"video_episode_index must satisfy 0 <= index < num_episodes; "
                f"got index={idx}, num_episodes={num_episodes}"
            )
    return unique_indices


def _action_bounds(env) -> tuple[list[float] | None, list[float] | None]:
    low = getattr(env.action_space, "low", None)
    high = getattr(env.action_space, "high", None)
    low_values = None if low is None else [float(value) for value in np.asarray(low).reshape(-1)]
    high_values = None if high is None else [float(value) for value in np.asarray(high).reshape(-1)]
    return low_values, high_values


class _RolloutWorker:
    def __init__(self, *, worker_id: int, control_queue, event_queue, worker_config: dict):
        self.worker_id = int(worker_id)
        self.control_queue = control_queue
        self.event_queue = event_queue
        self.worker_config = worker_config
        self.config = dict(worker_config["config"])
        self.variant = worker_config["variant"]
        self.mode = worker_config["mode"]
        self.template = worker_config["template"]
        self.variant_results_dir = worker_config.get("variant_results_dir")
        self.action_timeout_seconds = float(
            worker_config.get("rollout_action_timeout_seconds", 300)
        )
        self.formatter = get_formatter(self.config["env_family"])
        self.env = None
        self.prompt_vars = None
        self.action_dim = None
        self.action_shape = None
        self.action_low = None
        self.action_high = None
        self.video_saver = VideoSaveManager(self.config)

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            self.video_saver.close()

    def ensure_env(self):
        if self.env is not None:
            return

        if self.mode == "score":
            self.env = crossmaze_make(
                self.config["env_family"],
                self.variant,
                mode="score",
                config=self.config,
            )
            self.prompt_vars = apply_sensing_config_to_prompt_vars(
                POINTMAZE_VARIANTS[self.variant]["prompt_vars"],
                self.config,
            )
        else:
            meta = get_variant(self.config["env_family"], self.variant)
            self.env = crossmaze_make(
                self.config["env_family"],
                self.variant,
                mode="eval",
                config=self.config,
            )
            self.prompt_vars = _prepare_eval_prompt_vars(
                self.formatter,
                meta["prompt_vars"],
                self.env,
            )
            self.prompt_vars = apply_sensing_config_to_prompt_vars(
                self.prompt_vars,
                self.config,
            )
        self.env.assert_meta_consistent(self.prompt_vars)

        self.action_shape = tuple(int(value) for value in self.env.action_space.shape)
        self.action_dim = int(self.env.action_space.shape[0])
        checkpoint_action_dim = self.config.get("action_dim")
        if checkpoint_action_dim is not None and int(checkpoint_action_dim) != self.action_dim:
            raise ValueError(
                "Checkpoint action_dim does not match rollout env action space: "
                f"checkpoint={checkpoint_action_dim}, env={self.action_dim}, variant={self.variant}"
            )
        self.action_low, self.action_high = _action_bounds(self.env)

    def request_action(
        self,
        *,
        episode_index: int,
        step_index: int,
        prompt: str,
    ) -> ActionResponse:
        request = ActionRequest(
            request_id=uuid.uuid4().hex,
            worker_id=self.worker_id,
            episode_index=int(episode_index),
            step_index=int(step_index),
            prompt=prompt,
            action_shape=self.action_shape,
            action_dim=int(self.action_dim),
            action_low=self.action_low,
            action_high=self.action_high,
        )
        self.event_queue.put(
            {
                "type": "action_request",
                "worker_id": self.worker_id,
                "pid": os.getpid(),
                "request": request,
            }
        )
        try:
            message = self.control_queue.get(timeout=self.action_timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                "Timed out waiting for policy action response: "
                f"timeout={self.action_timeout_seconds}s request_id={request.request_id}"
            ) from exc
        if message.get("type") == "shutdown":
            raise RuntimeError("Worker received shutdown while waiting for an action response")
        if message.get("type") != "action_response":
            raise RuntimeError(f"Worker expected action_response, got {message!r}")
        response = message["response"]
        if response.request_id != request.request_id:
            raise RuntimeError(
                "Worker received action response for a different request: "
                f"expected={request.request_id}, got={response.request_id}"
            )
        return response

    def run_episode(self, *, episode_index: int, seed: int | None) -> EpisodeResult:
        self.ensure_env()
        eval_position = None
        reset_options = None
        if self.mode != "score":
            set_episode_rng_seed(seed)
            eval_seed = self.config.get("seed")
            eval_position = select_eval_position(
                self.config["env_family"],
                self.variant,
                episode_index=int(episode_index),
                seed=int(eval_seed) if eval_seed is not None else None,
                config=self.config,
            )
            if eval_position is not None:
                reset_options = {
                    "reset_cell": np.asarray(eval_position["start_cell"], dtype=np.int64),
                    "goal_cell": np.asarray(eval_position["goal_cell"], dtype=np.int64),
                }
        if seed is not None and hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(seed)
        if seed is None:
            obs, _ = self.env.reset(options=reset_options)
        else:
            obs, _ = self.env.reset(seed=seed, options=reset_options)

        num_episodes = int(self.config["num_episodes"])
        history_num, history_stride = validate_history_config(self.config)
        record_video = bool(self.config.get("record_video", False))
        record_step_logs = bool(self.config.get("record_step_logs", True)) and self.mode != "score"
        record_global_video = should_record_antmaze_global_video(self.config)
        video_episode_indices = set(_resolve_video_episode_indices(self.config, num_episodes))
        record_this_episode = record_video and int(episode_index) in video_episode_indices
        video_fps = int(self.config.get("video_fps", 20))
        video_ext = str(self.config.get("video_format", "gif")).lstrip(".")
        episode_dir = _episode_dir(self.variant_results_dir, self.mode, int(episode_index))
        if episode_dir is not None and (self.mode != "score" or record_this_episode):
            os.makedirs(episode_dir, exist_ok=True)

        frames = [] if record_this_episode else None
        global_frames = [] if record_this_episode and record_global_video else None
        if frames is not None:
            if self.mode == "score":
                capture_render_frame(
                    self.env,
                    frames,
                    env_family=self.config["env_family"],
                )
            else:
                capture_video_frames(
                    self.env,
                    frames,
                    global_frames,
                    env_family=self.config["env_family"],
                )

        history_buffer = []
        episode_return = 0.0
        episode_success = False
        episode_steps = 0
        parse_failures = 0
        fallbacks = 0
        action_time_seconds = 0.0
        action_count = 0
        terminated = False
        truncated = False
        video_path = None
        global_video_path = None

        while not (terminated or truncated):
            prompt = render_policy_prompt(
                formatter=self.formatter,
                template=self.template,
                prompt_vars=self.prompt_vars,
                obs=obs,
                history_buffer=history_buffer,
                history_num=history_num,
                history_stride=history_stride,
            )
            current_history_observation = _history_observation(self.formatter, obs)
            response = self.request_action(
                episode_index=int(episode_index),
                step_index=episode_steps,
                prompt=prompt,
            )
            action = np.asarray(response.action, dtype=np.float32).reshape(self.action_shape)
            generated_text = "\n\n".join(
                f"[Attempt {idx + 1}]\n{text}"
                for idx, text in enumerate(response.generated_attempts)
            )

            if record_step_logs and episode_dir is not None:
                write_step_log(
                    episode_dir,
                    episode_steps,
                    prompt=prompt,
                    action_text=generated_text,
                    executed_action=response.executed_action_text,
                    parse_status=response.parse_status,
                    attempt_count=response.attempt_count,
                    action_bin_probabilities=(
                        "\n\n".join(response.generated_probability_logs)
                        if response.generated_probability_logs
                        else None
                    ),
                    raw_continuous_action=response.raw_continuous_action,
                    gaussian_action_mean=response.gaussian_action_mean,
                    gaussian_action_std=response.gaussian_action_std,
                    student_t_action_mean=response.student_t_action_mean,
                    student_t_action_scale=response.student_t_action_scale,
                )

            obs, reward, terminated, truncated, info = self.env.step(action)
            history_buffer.append(
                {
                    "observation": current_history_observation,
                    "action_text": response.executed_action_text,
                }
            )

            if frames is not None:
                if self.mode == "score":
                    capture_render_frame(
                        self.env,
                        frames,
                        env_family=self.config["env_family"],
                    )
                else:
                    capture_video_frames(
                        self.env,
                        frames,
                        global_frames,
                        env_family=self.config["env_family"],
                    )

            episode_return += float(reward)
            episode_steps += 1
            parse_failures += int(response.parse_failures)
            fallbacks += int(response.fallback_count)
            action_time_seconds += float(response.action_time_seconds)
            action_count += int(response.generation_count)
            if self.mode != "score" and bool(info.get("success", terminated)):
                episode_success = True

        if frames is not None and episode_dir is not None:
            video_path, global_video_path = record_episode_videos(
                video_saver=self.video_saver,
                frames=frames,
                global_frames=global_frames,
                episode_dir=episode_dir,
                video_ext=video_ext,
                video_fps=video_fps,
            )

        return EpisodeResult(
            variant=self.variant,
            episode_index=int(episode_index),
            seed=seed,
            episode_return=float(episode_return),
            success=bool(episode_success),
            steps=int(episode_steps),
            parse_failures=int(parse_failures),
            fallbacks=int(fallbacks),
            action_time_seconds=float(action_time_seconds),
            action_count=int(action_count),
            worker_id=self.worker_id,
            worker_pid=os.getpid(),
            video_path=video_path,
            global_video_path=global_video_path,
            episode_artifact_dir=episode_dir,
            start_cell=(
                list(eval_position["start_cell"])
                if eval_position is not None
                else None
            ),
            goal_cell=(
                list(eval_position["goal_cell"])
                if eval_position is not None
                else None
            ),
            start_goal_difficulty=(
                float(eval_position["difficulty"])
                if eval_position is not None
                else None
            ),
            start_goal_difficulty_components=(
                dict(eval_position["difficulty_components"])
                if eval_position is not None
                else None
            ),
            start_goal_source=(
                str(eval_position["source"])
                if eval_position is not None
                else None
            ),
            start_goal_index=(
                int(eval_position["index"])
                if eval_position is not None
                else None
            ),
        )


def worker_entry(worker_id: int, control_queue, event_queue, worker_config: dict):
    worker = None
    current_episode = None
    current_attempt = None
    try:
        worker = _RolloutWorker(
            worker_id=worker_id,
            control_queue=control_queue,
            event_queue=event_queue,
            worker_config=worker_config,
        )
        worker.ensure_env()
        event_queue.put(
            {
                "type": "ready",
                "worker_id": worker_id,
                "pid": os.getpid(),
            }
        )
        while True:
            message = control_queue.get()
            message_type = message.get("type")
            if message_type == "shutdown":
                break
            if message_type != "run_episode":
                raise RuntimeError(f"Unknown worker control message: {message!r}")

            current_episode = int(message["episode_index"])
            current_attempt = int(message.get("attempt", 1))
            result = worker.run_episode(
                episode_index=current_episode,
                seed=message.get("seed"),
            )
            event_queue.put(
                {
                    "type": "episode_result",
                    "worker_id": worker_id,
                    "pid": os.getpid(),
                    "episode_index": current_episode,
                    "attempt": current_attempt,
                    "result": result,
                }
            )
            current_episode = None
            current_attempt = None

            if worker_config.get("rollout_worker_lifetime") == "episode":
                break
    except BaseException as exc:  # noqa: BLE001 - child must report failures to parent.
        event_queue.put(
            {
                "type": "worker_error",
                "worker_id": worker_id,
                "pid": os.getpid(),
                "episode_index": current_episode,
                "attempt": current_attempt,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        sys.exit(1)
    finally:
        if worker is not None:
            worker.close()
