from __future__ import annotations

from utils.distributed import DistributedContext


def _as_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Expected a boolean value, got {value!r}")
    return bool(value)


def resolve_eval_parallel_episodes(config: dict) -> int:
    parallel_episodes = int(config.get("eval_parallel_episodes", 1))
    if parallel_episodes < 1:
        raise ValueError(
            "eval_parallel_episodes must be >= 1, "
            f"got {parallel_episodes}"
        )
    return parallel_episodes


def resolve_eval_distribute_variants(config: dict) -> bool:
    return _as_bool(config.get("eval_distribute_variants", True))


def assigned_eval_variants(
    variants,
    context: DistributedContext,
    *,
    distribute_variants: bool,
) -> list[str]:
    resolved = list(variants)
    if not context.is_distributed:
        return resolved
    if not distribute_variants:
        return resolved if context.is_main_process else []
    return resolved[context.rank :: context.world_size]


def eval_variant_assignments(
    variants,
    context: DistributedContext,
    *,
    distribute_variants: bool,
) -> dict[int, list[str]]:
    resolved = list(variants)
    if not context.is_distributed:
        return {0: resolved}
    if not distribute_variants:
        return {
            rank: resolved if rank == 0 else []
            for rank in range(context.world_size)
        }
    return {
        rank: resolved[rank :: context.world_size]
        for rank in range(context.world_size)
    }
