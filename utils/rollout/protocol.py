from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionRequest:
    request_id: str
    worker_id: int
    episode_index: int
    step_index: int
    prompt: str
    action_shape: tuple[int, ...]
    action_dim: int
    action_low: list[float] | None = None
    action_high: list[float] | None = None


@dataclass(frozen=True)
class ActionResponse:
    request_id: str
    action: list[float]
    executed_action_text: str
    generated_attempts: list[str]
    generated_probability_logs: list[str]
    attempt_count: int
    parse_status: str
    parse_failures: int
    fallback_count: int
    action_time_seconds: float
    generation_count: int
    raw_continuous_action: list[float] | None = None
    gaussian_action_mean: list[float] | None = None
    gaussian_action_std: list[float] | None = None
    student_t_action_mean: list[float] | None = None
    student_t_action_scale: list[float] | None = None


@dataclass(frozen=True)
class EpisodeResult:
    variant: str
    episode_index: int
    seed: int | None
    episode_return: float
    success: bool
    steps: int
    parse_failures: int
    fallbacks: int
    action_time_seconds: float
    action_count: int
    worker_id: int
    worker_pid: int | None
    video_path: str | None = None
    global_video_path: str | None = None
    episode_artifact_dir: str | None = None
    start_cell: list[int] | None = None
    goal_cell: list[int] | None = None
    start_goal_difficulty: float | None = None
    start_goal_source: str | None = None
    start_goal_index: int | None = None
    worker_failed: bool = False
    failure_error: str | None = None


@dataclass(frozen=True)
class WorkerFailure:
    variant: str
    worker_id: int
    worker_pid: int | None
    episode_index: int | None
    attempt: int | None
    error: str
    exitcode: int | None = None
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupervisorResult:
    variant: str
    episode_results: list[EpisodeResult]
    worker_failures: list[WorkerFailure] = field(default_factory=list)
    workers_used: list[int] = field(default_factory=list)
