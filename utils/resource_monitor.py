from __future__ import annotations

import csv
import io
import os
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any


GPU_QUERY_FIELDS = (
    "index",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
)


def resource_monitor_path(experiment_id: str, root: str | os.PathLike = "sys_info") -> Path:
    return Path(root) / f"{experiment_id}.txt"


def parse_meminfo(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].endswith(":"):
            continue
        key = parts[0].rstrip(":")
        try:
            value = int(parts[1])
        except ValueError:
            continue
        unit = parts[2].lower() if len(parts) > 2 else "b"
        if unit == "kb":
            value *= 1024
        values[key] = value
    return values


def read_memory_status(meminfo_path: str | os.PathLike = "/proc/meminfo") -> dict[str, int | None]:
    text = Path(meminfo_path).read_text(encoding="utf-8")
    values = parse_meminfo(text)
    mem_total = values.get("MemTotal")
    mem_available = values.get("MemAvailable", values.get("MemFree"))
    mem_free = values.get("MemFree")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    return {
        "mem_total": mem_total,
        "mem_available": mem_available,
        "mem_free": mem_free,
        "mem_used": (
            mem_total - mem_available
            if mem_total is not None and mem_available is not None
            else None
        ),
        "swap_total": swap_total,
        "swap_free": swap_free,
        "swap_used": (
            swap_total - swap_free
            if swap_total is not None and swap_free is not None
            else None
        ),
    }


def _parse_optional_float(value: str) -> float | None:
    normalized = value.strip()
    if normalized in {"", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_nvidia_smi_query(output: str) -> list[dict[str, Any]]:
    gpus = []
    for row in csv.reader(io.StringIO(output)):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(GPU_QUERY_FIELDS):
            raise ValueError(
                f"Expected {len(GPU_QUERY_FIELDS)} nvidia-smi fields, got {len(row)}: {row!r}"
            )
        values = [value.strip() for value in row]
        gpus.append(
            {
                "index": values[0],
                "name": values[1],
                "memory_used_mib": _parse_optional_float(values[2]),
                "memory_total_mib": _parse_optional_float(values[3]),
                "utilization_gpu_percent": _parse_optional_float(values[4]),
                "temperature_c": _parse_optional_float(values[5]),
                "power_draw_w": _parse_optional_float(values[6]),
            }
        )
    return gpus


def query_gpu_status(timeout_seconds: float = 2.0) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_seconds,
    )
    return parse_nvidia_smi_query(result.stdout)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 3):.2f}GiB"


def _format_optional(value: float | None, suffix: str, precision: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def render_resource_status(
    *,
    status: str,
    interval_seconds: float,
    memory: dict[str, int | None] | None,
    memory_error: str | None,
    gpus: list[dict[str, Any]] | None,
    gpu_error: str | None,
    timestamp_utc: str | None = None,
    pid: int | None = None,
    hostname: str | None = None,
) -> str:
    lines = [
        f"timestamp_utc: {timestamp_utc or _utc_timestamp()}",
        f"status: {status}",
        f"pid: {pid if pid is not None else os.getpid()}",
        f"host: {hostname or socket.gethostname()}",
        f"interval_seconds: {interval_seconds:g}",
    ]
    if memory is None:
        lines.append(f"memory_error: {memory_error or 'unknown'}")
    else:
        lines.append(
            "memory: "
            f"used={_format_bytes(memory.get('mem_used'))} "
            f"available={_format_bytes(memory.get('mem_available'))} "
            f"free={_format_bytes(memory.get('mem_free'))} "
            f"total={_format_bytes(memory.get('mem_total'))}"
        )
        lines.append(
            "swap: "
            f"used={_format_bytes(memory.get('swap_used'))} "
            f"free={_format_bytes(memory.get('swap_free'))} "
            f"total={_format_bytes(memory.get('swap_total'))}"
        )
    if gpu_error is not None:
        lines.append(f"gpu_error: {gpu_error}")
    else:
        gpus = gpus or []
        lines.append(f"gpu_count: {len(gpus)}")
        for gpu in gpus:
            lines.append(
                f"gpu[{gpu.get('index', 'unknown')}]: "
                f"name={gpu.get('name', 'unknown')} "
                f"mem={_format_optional(gpu.get('memory_used_mib'), 'MiB')}/"
                f"{_format_optional(gpu.get('memory_total_mib'), 'MiB')} "
                f"util={_format_optional(gpu.get('utilization_gpu_percent'), '%')} "
                f"temp={_format_optional(gpu.get('temperature_c'), 'C')} "
                f"power={_format_optional(gpu.get('power_draw_w'), 'W', precision=2)}"
            )
    return "\n".join(lines) + "\n"


class ResourceMonitor:
    def __init__(
        self,
        path: str | os.PathLike,
        *,
        interval_seconds: float = 1.0,
        enabled: bool = True,
        gpu_query_timeout_seconds: float = 2.0,
        memory_provider: Callable[[], dict[str, int | None]] | None = None,
        gpu_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.path = Path(path)
        self.interval_seconds = float(interval_seconds)
        if self.interval_seconds <= 0:
            raise ValueError(
                f"resource monitor interval_seconds must be > 0, got {interval_seconds}"
            )
        self.enabled = bool(enabled)
        self.gpu_query_timeout_seconds = float(gpu_query_timeout_seconds)
        self._memory_provider = memory_provider or read_memory_status
        self._gpu_provider = gpu_provider or (
            lambda: query_gpu_status(timeout_seconds=self.gpu_query_timeout_seconds)
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop(final_status="stopped" if exc_type is None else None)
        return False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if not self.enabled or self._thread is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="resource-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, *, final_status: str | None = "stopped"):
        if self._closed:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.gpu_query_timeout_seconds + 1.0, 3.0))
            self._thread = None
        if self.enabled and final_status is not None:
            try:
                self.write_once(status=final_status)
            except Exception:
                pass
        self._closed = True

    def write_once(self, *, status: str = "running"):
        memory = None
        memory_error = None
        gpus = None
        gpu_error = None
        try:
            memory = self._memory_provider()
        except Exception as exc:
            memory_error = f"{type(exc).__name__}: {exc}"
        try:
            gpus = self._gpu_provider()
        except Exception as exc:
            gpu_error = f"{type(exc).__name__}: {exc}"
        content = render_resource_status(
            status=status,
            interval_seconds=self.interval_seconds,
            memory=memory,
            memory_error=memory_error,
            gpus=gpus,
            gpu_error=gpu_error,
        )
        self._atomic_write(content)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self.write_once(status="running")
            except Exception:
                pass
            self._stop_event.wait(self.interval_seconds)

    def _atomic_write(self, content: str):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, self.path)
