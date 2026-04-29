from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PROCESS_PROGRESS_QUEUE = None


def configure_process_sub_progress(progress_queue):
    """Install the process progress queue in a child process."""
    global _PROCESS_PROGRESS_QUEUE
    _PROCESS_PROGRESS_QUEUE = progress_queue


def get_sub_progress_process(worker_id: str | int | None = None):
    """Return a sub-progress proxy for a process-pool child."""
    if _PROCESS_PROGRESS_QUEUE is None:
        raise RuntimeError("Process worker progress queue is not configured.")
    if worker_id is None:
        worker_id = os.getpid()
    return _QueuedWorkerProgress(_PROCESS_PROGRESS_QUEUE, str(worker_id), os.getpid())


configure_process_worker_progress = configure_process_sub_progress
get_process_worker_progress = get_sub_progress_process


@dataclass
class _ProgressSnapshot:
    desc: str
    current: int
    total: int
    start_time: float
    extra: str = ""
    pid: int | None = None
    updated_at: float = 0.0


class FileProgress:
    def __init__(
        self,
        path: str | os.PathLike | None = None,
        *,
        progress_dir: str | os.PathLike = "progress",
        interval_seconds: float = 5.0,
        width: int = 24,
        cleanup_on_success: bool = True,
        print_on_success: bool = True,
    ):
        self.path = Path(path) if path is not None else Path(progress_dir) / f"{uuid.uuid4().hex}.txt"
        self.interval_seconds = float(interval_seconds)
        self.width = int(width)
        self.cleanup_on_success = cleanup_on_success
        self.print_on_success = print_on_success
        if self.interval_seconds < 0:
            raise ValueError(f"interval_seconds must be >= 0, got {interval_seconds}")
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        self._last_write_time: float | None = None
        self._last_line: str | None = None
        self._closed = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(success=exc_type is None)
        return False

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(int(seconds), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _render(self, desc: str, current: int, total: int, start_time: float, extra: str = "") -> str:
        total = max(int(total), 1)
        current = min(max(int(current), 0), total)
        ratio = current / total
        filled = min(self.width, int(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.monotonic() - start_time, 0.0)
        rate = current / elapsed if elapsed > 0 else 0.0
        remaining = (total - current) / rate if rate > 0 else 0.0
        suffix = f" {extra}" if extra else ""
        return (
            f"{desc} [{bar}] {ratio * 100:6.2f}% "
            f"{current}/{total} elapsed={self._format_duration(elapsed)} "
            f"eta={self._format_duration(remaining)}{suffix}"
        )

    def _should_write(self, current: int, total: int) -> bool:
        if self._last_write_time is None:
            return True
        if current <= 1 or current >= total:
            return True
        return (time.monotonic() - self._last_write_time) >= self.interval_seconds

    def update(self, desc: str, current: int, total: int, start_time: float, *, extra: str = ""):
        if self._closed:
            raise RuntimeError("Cannot update a closed FileProgress.")
        total = max(int(total), 1)
        current = min(max(int(current), 0), total)
        if not self._should_write(current, total):
            return

        line = self._render(desc, current, total, start_time, extra=extra)
        self._file.seek(0)
        self._file.write(line + "\n")
        self._file.truncate()
        self._file.flush()
        self._last_write_time = time.monotonic()
        self._last_line = line

    def close(self, *, success: bool = True):
        if self._closed:
            return
        self._file.close()
        self._closed = True
        if success and self.print_on_success and self._last_line is not None:
            print(self._last_line, flush=True)
        if success and self.cleanup_on_success:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


class _DirectWorkerProgress:
    def __init__(self, owner: "MultiWorkerFileProgress", worker_id: str):
        self._owner = owner
        self._worker_id = str(worker_id)
        self.pid = os.getpid()

    def update(self, desc: str, current: int, total: int, start_time: float, *, extra: str = ""):
        self._owner.update_worker(
            self._worker_id,
            desc,
            current,
            total,
            start_time,
            extra=extra,
            pid=self.pid,
        )

    def increment_total(self, amount: int = 1):
        self._owner.increment_total(amount)


class _QueuedWorkerProgress:
    def __init__(self, progress_queue, worker_id: str, pid: int):
        self._queue = progress_queue
        self._worker_id = str(worker_id)
        self.pid = pid

    def update(self, desc: str, current: int, total: int, start_time: float, *, extra: str = ""):
        self._queue.put(
            {
                "type": "worker_update",
                "worker_id": self._worker_id,
                "pid": self.pid,
                "desc": desc,
                "current": int(current),
                "total": int(total),
                "start_time": float(start_time),
                "extra": extra,
            }
        )

    def increment_total(self, amount: int = 1):
        self._queue.put({"type": "total_increment", "amount": int(amount)})


class MultiWorkerFileProgress:
    """Render several worker progress bars plus an aggregate total to one file."""

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        *,
        progress_dir: str | os.PathLike = "progress",
        desc: str = "total",
        total: int = 1,
        interval_seconds: float = 5.0,
        width: int = 24,
        cleanup_on_success: bool = True,
        print_on_success: bool = True,
    ):
        self.path = Path(path) if path is not None else Path(progress_dir) / f"{uuid.uuid4().hex}.txt"
        self.desc = str(desc)
        self.total = max(int(total), 1)
        self.current = 0
        self.start_time = time.monotonic()
        self.interval_seconds = float(interval_seconds)
        self.width = int(width)
        self.cleanup_on_success = cleanup_on_success
        self.print_on_success = print_on_success
        if self.interval_seconds < 0:
            raise ValueError(f"interval_seconds must be >= 0, got {interval_seconds}")
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {width}")

        self._workers: dict[str, _ProgressSnapshot] = {}
        self._lock = threading.Lock()
        self._last_write_time: float | None = None
        self._last_snapshot: str | None = None
        self._closed = False
        self._queue_stop = threading.Event()
        self._queue_thread: threading.Thread | None = None
        self._process_queue = None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_snapshot(force=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(success=exc_type is None)
        return False

    @staticmethod
    def _format_duration(seconds: float) -> str:
        return FileProgress._format_duration(seconds)

    def _render_line(
        self,
        prefix: str,
        desc: str,
        current: int,
        total: int,
        start_time: float,
        *,
        extra: str = "",
    ) -> str:
        total = max(int(total), 1)
        current = min(max(int(current), 0), total)
        ratio = current / total
        filled = min(self.width, int(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(time.monotonic() - start_time, 0.0)
        rate = current / elapsed if elapsed > 0 else 0.0
        remaining = (total - current) / rate if rate > 0 else 0.0
        suffix = f" {extra}" if extra else ""
        return (
            f"{prefix} {desc} [{bar}] {ratio * 100:6.2f}% "
            f"{current}/{total} elapsed={self._format_duration(elapsed)} "
            f"eta={self._format_duration(remaining)}{suffix}"
        )

    def _render_snapshot_locked(self) -> str:
        lines = []
        for worker_id in sorted(self._workers):
            state = self._workers[worker_id]
            pid_text = f" pid={state.pid}" if state.pid is not None else ""
            lines.append(
                self._render_line(
                    f"worker {worker_id}{pid_text}",
                    state.desc,
                    state.current,
                    state.total,
                    state.start_time,
                    extra=state.extra,
                )
            )
        lines.append(
            self._render_line(
                "total",
                self.desc,
                self.current,
                self.total,
                self.start_time,
            )
        )
        return "\n".join(lines) + "\n"

    def _should_write_locked(self, force: bool) -> bool:
        if force or self._last_write_time is None:
            return True
        if self.current <= 1 or self.current >= self.total:
            return True
        return (time.monotonic() - self._last_write_time) >= self.interval_seconds

    def _write_snapshot(self, *, force: bool = False):
        with self._lock:
            if self._closed:
                return
            if not self._should_write_locked(force):
                return
            snapshot = self._render_snapshot_locked()
            self.path.write_text(snapshot, encoding="utf-8")
            self._last_write_time = time.monotonic()
            self._last_snapshot = snapshot

    def get_sub_progress_thread(self, worker_id: str | int):
        return _DirectWorkerProgress(self, str(worker_id))

    def thread_worker(self, worker_id: str | int):
        return self.get_sub_progress_thread(worker_id)

    def get_process_progress_queue(self, ctx):
        if self._process_queue is not None:
            return self._process_queue
        self._process_queue = ctx.Queue()
        self._queue_thread = threading.Thread(target=self._drain_process_queue, daemon=True)
        self._queue_thread.start()
        return self._process_queue

    def process_initializer(self, ctx):
        progress_queue = self.get_process_progress_queue(ctx)
        return configure_process_sub_progress, (progress_queue,)

    def process_queue(self, ctx):
        return self.get_process_progress_queue(ctx)

    def update_worker(
        self,
        worker_id: str | int,
        desc: str,
        current: int,
        total: int,
        start_time: float,
        *,
        extra: str = "",
        pid: int | None = None,
    ):
        if self._closed:
            raise RuntimeError("Cannot update a closed MultiWorkerFileProgress.")
        with self._lock:
            self._workers[str(worker_id)] = _ProgressSnapshot(
                desc=str(desc),
                current=min(max(int(current), 0), max(int(total), 1)),
                total=max(int(total), 1),
                start_time=float(start_time),
                extra=str(extra),
                pid=pid,
                updated_at=time.monotonic(),
            )
        self._write_snapshot()

    def increment_total(self, amount: int = 1):
        if self._closed:
            raise RuntimeError("Cannot update a closed MultiWorkerFileProgress.")
        with self._lock:
            self.current = min(self.total, max(0, self.current + int(amount)))
        self._write_snapshot()

    def _handle_queue_event(self, event: dict[str, Any]):
        event_type = event.get("type")
        if event_type == "worker_update":
            self.update_worker(
                event["worker_id"],
                event["desc"],
                event["current"],
                event["total"],
                event["start_time"],
                extra=event.get("extra", ""),
                pid=event.get("pid"),
            )
        elif event_type == "total_increment":
            self.increment_total(int(event.get("amount", 1)))

    def _drain_process_queue(self):
        while not self._queue_stop.is_set():
            try:
                event = self._process_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._handle_queue_event(event)

        while True:
            try:
                event = self._process_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_queue_event(event)

    def close(self, *, success: bool = True):
        if self._closed:
            return
        self._queue_stop.set()
        if self._queue_thread is not None:
            self._queue_thread.join(timeout=2.0)
        self._write_snapshot(force=True)
        self._closed = True
        if success and self.print_on_success and self._last_snapshot is not None:
            print(self._last_snapshot, end="", flush=True)
        if success and self.cleanup_on_success:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
