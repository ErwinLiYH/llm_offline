from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


VALID_PARALLEL_BACKENDS = {"single", "ddp"}


@dataclass(frozen=True)
class DistributedContext:
    backend: str
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_distributed: bool = False
    device: torch.device = torch.device("cpu")

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def resolve_parallel_backend(config: dict, override: str | None = None) -> str:
    backend = override if override is not None else config.get("parallel_backend", "single")
    backend = str(backend).strip().lower()
    if backend not in VALID_PARALLEL_BACKENDS:
        raise ValueError(
            f"parallel_backend must be one of {sorted(VALID_PARALLEL_BACKENDS)}, got {backend!r}"
        )
    config["parallel_backend"] = backend
    return backend


def init_distributed_context(config: dict, backend: str) -> DistributedContext:
    if backend == "single":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(backend=backend, device=device)

    required_env = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required_env if name not in os.environ]
    if missing:
        raise RuntimeError(
            "parallel_backend='ddp' must be launched with torchrun; "
            f"missing environment variables: {missing}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("parallel_backend='ddp' requires CUDA GPUs.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    timeout_seconds = int(config.get("distributed_timeout_seconds", 1800))
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=timeout_seconds),
    )
    return DistributedContext(
        backend=backend,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=True,
        device=torch.device("cuda", local_rank),
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.is_distributed:
        dist.barrier()


def broadcast_object(value: Any, context: DistributedContext, *, src: int = 0) -> Any:
    if not context.is_distributed:
        return value
    values = [value]
    dist.broadcast_object_list(values, src=src)
    return values[0]


def all_gather_objects(value: Any, context: DistributedContext) -> list[Any]:
    if not context.is_distributed:
        return [value]
    values = [None] * context.world_size
    dist.all_gather_object(values, value)
    return values


def reduce_mean(value: float, context: DistributedContext, device: torch.device) -> float:
    if not context.is_distributed:
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= context.world_size
    return float(tensor.item())


def rank_zero_print(context: DistributedContext, *args, **kwargs) -> None:
    if context.is_main_process:
        print(*args, **kwargs)


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model
