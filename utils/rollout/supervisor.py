from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from collections import deque
from dataclasses import dataclass

from utils.eval_parallel import (
    resolve_policy_batch_timeout_ms,
    resolve_rollout_action_timeout_seconds,
    resolve_rollout_worker_lifetime,
    resolve_rollout_worker_num,
    resolve_rollout_worker_retries,
    resolve_rollout_worker_start_timeout_seconds,
)
from utils.rollout.protocol import (
    ActionRequest,
    EpisodeResult,
    SupervisorResult,
    WorkerFailure,
)
from utils.rollout.worker_main import worker_entry


@dataclass
class _EpisodeTask:
    episode_index: int
    seed: int | None
    attempt: int = 1


@dataclass
class _WorkerHandle:
    worker_id: int
    process: object
    control_queue: object
    started_at: float
    ready: bool = False
    pid: int | None = None
    current_task: _EpisodeTask | None = None
    stopping: bool = False


def _placeholder_episode_result(
    *,
    variant: str,
    task: _EpisodeTask,
    failure: WorkerFailure,
) -> EpisodeResult:
    return EpisodeResult(
        variant=variant,
        episode_index=task.episode_index,
        seed=task.seed,
        episode_return=0.0,
        success=False,
        steps=0,
        parse_failures=0,
        fallbacks=0,
        action_time_seconds=0.0,
        action_count=0,
        worker_id=failure.worker_id,
        worker_pid=failure.worker_pid,
        worker_failed=True,
        failure_error=failure.error,
    )


def _episode_seeds(config: dict, mode: str, num_episodes: int) -> list[int | None]:
    seed = int(config.get("seed", 1))
    if mode == "score":
        return [seed if ep_idx == 0 else None for ep_idx in range(num_episodes)]
    return [seed + ep_idx for ep_idx in range(num_episodes)]


def run_episode_supervisor(
    *,
    config: dict,
    variant: str,
    mode: str,
    template: str,
    policy,
    variant_results_dir: str | None,
    worker_target=worker_entry,
    multiprocessing_context: str = "spawn",
) -> SupervisorResult:
    num_episodes = int(config["num_episodes"])
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    worker_num = min(resolve_rollout_worker_num(config), num_episodes)
    worker_lifetime = resolve_rollout_worker_lifetime(config)
    worker_retries = resolve_rollout_worker_retries(config)
    start_timeout_seconds = resolve_rollout_worker_start_timeout_seconds(config)
    action_timeout_seconds = resolve_rollout_action_timeout_seconds(config)
    policy_batch_timeout_seconds = resolve_policy_batch_timeout_ms(config) / 1000.0

    ctx = mp.get_context(multiprocessing_context)
    event_queue = ctx.Queue()
    pending_tasks = deque(
        _EpisodeTask(ep_idx, seed)
        for ep_idx, seed in enumerate(_episode_seeds(config, mode, num_episodes))
    )
    handles: dict[int, _WorkerHandle] = {}
    results_by_episode: dict[int, EpisodeResult] = {}
    failures: list[WorkerFailure] = []
    workers_used: set[int] = set()
    pending_action_requests: list[ActionRequest] = []
    next_worker_id = 0
    startup_failures = 0

    def start_worker() -> _WorkerHandle:
        nonlocal next_worker_id
        worker_id = next_worker_id
        next_worker_id += 1
        control_queue = ctx.Queue()
        worker_config = {
            "config": dict(config),
            "variant": variant,
            "mode": mode,
            "template": template,
            "variant_results_dir": variant_results_dir,
            "rollout_worker_lifetime": worker_lifetime,
            "rollout_action_timeout_seconds": action_timeout_seconds,
        }
        process = ctx.Process(
            target=worker_target,
            args=(worker_id, control_queue, event_queue, worker_config),
            daemon=False,
        )
        process.start()
        handle = _WorkerHandle(
            worker_id=worker_id,
            process=process,
            control_queue=control_queue,
            started_at=time.monotonic(),
            pid=process.pid,
        )
        handles[worker_id] = handle
        if process.pid is not None:
            workers_used.add(int(process.pid))
        return handle

    def live_or_starting_handles() -> list[_WorkerHandle]:
        return [handle for handle in handles.values() if not handle.stopping]

    def assign_task(handle: _WorkerHandle):
        if handle.stopping or handle.current_task is not None:
            return
        if not pending_tasks:
            handle.stopping = True
            try:
                handle.control_queue.put({"type": "shutdown"})
            except Exception:
                pass
            return
        task = pending_tasks.popleft()
        handle.current_task = task
        handle.control_queue.put(
            {
                "type": "run_episode",
                "episode_index": task.episode_index,
                "seed": task.seed,
                "attempt": task.attempt,
            }
        )

    def fail_task(handle: _WorkerHandle, *, error: str, traceback_text=None, exitcode=None):
        task = handle.current_task
        failure = WorkerFailure(
            variant=variant,
            worker_id=handle.worker_id,
            worker_pid=handle.pid,
            episode_index=task.episode_index if task is not None else None,
            attempt=task.attempt if task is not None else None,
            error=error,
            exitcode=exitcode,
            traceback=traceback_text,
        )
        failures.append(failure)
        if task is None:
            return
        if task.attempt <= worker_retries:
            pending_tasks.appendleft(
                _EpisodeTask(
                    episode_index=task.episode_index,
                    seed=task.seed,
                    attempt=task.attempt + 1,
                )
            )
        elif task.episode_index not in results_by_episode:
            results_by_episode[task.episode_index] = _placeholder_episode_result(
                variant=variant,
                task=task,
                failure=failure,
            )

    def close_handle(handle: _WorkerHandle, *, terminate: bool = False):
        handle.stopping = True
        if terminate and handle.process.is_alive():
            handle.process.terminate()
            try:
                handle.process.join(timeout=2.0)
            except Exception:
                pass
            if handle.process.is_alive():
                handle.process.kill()
                handle.process.join(timeout=1.0)
            handles.pop(handle.worker_id, None)
            return
        if handle.process.is_alive():
            try:
                handle.control_queue.put({"type": "shutdown"})
            except Exception:
                pass
        try:
            handle.process.join()
        except Exception:
            pass
        handles.pop(handle.worker_id, None)

    def maybe_start_replacements():
        if len(results_by_episode) >= num_episodes:
            return
        desired = min(worker_num, num_episodes - len(results_by_episode))
        while (
            pending_tasks
            and len(live_or_starting_handles()) < desired
            and startup_failures <= max(1, worker_retries + 1)
        ):
            start_worker()

    def mark_remaining_after_startup_failure():
        while pending_tasks:
            task = pending_tasks.popleft()
            failure = WorkerFailure(
                variant=variant,
                worker_id=-1,
                worker_pid=None,
                episode_index=task.episode_index,
                attempt=task.attempt,
                error="rollout worker startup failed repeatedly",
            )
            failures.append(failure)
            results_by_episode[task.episode_index] = _placeholder_episode_result(
                variant=variant,
                task=task,
                failure=failure,
            )

    def handle_message(message: dict):
        nonlocal startup_failures
        message_type = message.get("type")
        worker_id = int(message.get("worker_id"))
        handle = handles.get(worker_id)
        if handle is None:
            return
        pid = message.get("pid")
        if pid is not None:
            handle.pid = int(pid)
            workers_used.add(int(pid))

        if message_type == "ready":
            handle.ready = True
            assign_task(handle)
            return

        if message_type == "action_request":
            pending_action_requests.append(message["request"])
            return

        if message_type == "episode_result":
            result = message["result"]
            results_by_episode[int(result.episode_index)] = result
            handle.current_task = None
            if worker_lifetime == "episode":
                close_handle(handle)
                maybe_start_replacements()
            else:
                assign_task(handle)
            return

        if message_type == "worker_error":
            fail_task(
                handle,
                error=str(message.get("error", "worker error")),
                traceback_text=message.get("traceback"),
                exitcode=handle.process.exitcode,
            )
            if handle.current_task is None:
                startup_failures += 1
            close_handle(handle, terminate=True)
            if startup_failures > max(1, worker_retries + 1) and not handles:
                mark_remaining_after_startup_failure()
            else:
                maybe_start_replacements()
            return

        raise RuntimeError(f"Unknown rollout worker event: {message!r}")

    def poll_worker_processes():
        nonlocal startup_failures
        now = time.monotonic()
        for handle in list(handles.values()):
            if handle.stopping:
                if not handle.process.is_alive():
                    close_handle(handle)
                continue
            if not handle.ready and now - handle.started_at > start_timeout_seconds:
                fail_task(
                    handle,
                    error=(
                        "rollout worker did not report ready before "
                        f"{start_timeout_seconds}s startup timeout"
                    ),
                    exitcode=handle.process.exitcode,
                )
                startup_failures += 1
                close_handle(handle, terminate=True)
                continue
            if not handle.process.is_alive():
                try:
                    message = event_queue.get(timeout=0.05)
                except queue.Empty:
                    message = None
                if message is not None:
                    handle_message(message)
                    continue
                exitcode = handle.process.exitcode
                fail_task(
                    handle,
                    error=f"rollout worker process exited unexpectedly with exitcode={exitcode}",
                    exitcode=exitcode,
                )
                if handle.current_task is None:
                    startup_failures += 1
                close_handle(handle)
        if startup_failures > max(1, worker_retries + 1) and not handles:
            mark_remaining_after_startup_failure()
        maybe_start_replacements()

    def process_pending_actions():
        if not pending_action_requests:
            return
        if policy_batch_timeout_seconds > 0:
            deadline = time.monotonic() + policy_batch_timeout_seconds
            while time.monotonic() < deadline:
                timeout = max(0.0, deadline - time.monotonic())
                try:
                    message = event_queue.get(timeout=timeout)
                except queue.Empty:
                    break
                handle_message(message)
        requests = pending_action_requests[:]
        pending_action_requests.clear()
        responses = policy.respond(requests)
        for request, response in zip(requests, responses, strict=True):
            handle = handles.get(request.worker_id)
            if handle is None or handle.stopping:
                continue
            handle.control_queue.put(
                {
                    "type": "action_response",
                    "response": response,
                }
            )

    try:
        for _ in range(worker_num):
            start_worker()

        while len(results_by_episode) < num_episodes:
            poll_worker_processes()
            process_pending_actions()
            if len(results_by_episode) >= num_episodes:
                break
            try:
                message = event_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            handle_message(message)
    finally:
        for handle in list(handles.values()):
            close_handle(handle, terminate=handle.current_task is not None)

    ordered_results = [
        results_by_episode[idx]
        for idx in range(num_episodes)
        if idx in results_by_episode
    ]
    return SupervisorResult(
        variant=variant,
        episode_results=ordered_results,
        worker_failures=failures,
        workers_used=sorted(workers_used),
    )
