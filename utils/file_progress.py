from __future__ import annotations

import os
import time
import uuid
from pathlib import Path


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
