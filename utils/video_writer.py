from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import imageio.v2 as imageio
import numpy as np


def resolve_video_save_workers(config: dict) -> int:
    workers = int(config.get("video_save_workers", 1))
    if workers < 0:
        raise ValueError(f"video_save_workers must be >= 0, got {workers}")
    return workers


def resolve_video_save_max_pending(config: dict, workers: int) -> int:
    default = max(2, workers * 2) if workers > 0 else 0
    configured = config.get("video_save_max_pending")
    max_pending = default if configured is None else int(configured)
    if workers == 0:
        if max_pending < 0:
            raise ValueError(
                "video_save_max_pending must be >= 0 when video_save_workers=0, "
                f"got {max_pending}"
            )
        return 0
    if max_pending < workers:
        raise ValueError(
            "video_save_max_pending must be >= video_save_workers when asynchronous "
            f"video saving is enabled, got max_pending={max_pending}, workers={workers}"
        )
    return max_pending


def save_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        duration_sec = 1.0 / max(fps, 1)
        imageio.mimsave(
            output_path,
            frames,
            format="GIF",
            duration=duration_sec,
        )
        return

    try:
        imageio.mimsave(output_path, frames, fps=fps)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save video to {output_path}. mp4 output requires a working "
            "ffmpeg backend; try video_format='gif' if ffmpeg is unavailable."
        ) from exc


class VideoSaveManager:
    def __init__(self, config: dict):
        self.workers = resolve_video_save_workers(config)
        self.max_pending = resolve_video_save_max_pending(config, self.workers)
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="video-save",
            )
            if self.workers > 0
            else None
        )
        self._pending: set[Future] = set()
        self._closed = False

    @property
    def asynchronous(self) -> bool:
        return self._executor is not None

    def submit(
        self,
        frames: list[np.ndarray],
        output_path: str,
        fps: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot submit video after VideoSaveManager.close()")
        if self._executor is None:
            save_video(frames, output_path, fps)
            return

        while len(self._pending) >= self.max_pending:
            self._wait_for_one()
        self._pending.add(
            self._executor.submit(save_video, frames, output_path, fps)
        )

    def _wait_for_one(self) -> None:
        if not self._pending:
            return
        done, pending = wait(self._pending, return_when=FIRST_COMPLETED)
        self._pending = set(pending)
        first_error = None
        for future in done:
            try:
                future.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def wait(self) -> None:
        pending = self._pending
        self._pending = set()
        first_error = None
        for future in pending:
            try:
                future.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.wait()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True

    def __enter__(self) -> VideoSaveManager:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False
