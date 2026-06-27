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


def resolve_rollout_worker_num(config: dict) -> int:
    if "eval_parallel_episodes" in config:
        raise ValueError(
            "eval_parallel_episodes is deprecated and no longer supported; "
            "rename it to rollout_worker_num."
        )
    worker_num = int(config.get("rollout_worker_num", 1))
    if worker_num < 1:
        raise ValueError(f"rollout_worker_num must be >= 1, got {worker_num}")
    return worker_num


def resolve_rollout_worker_lifetime(config: dict) -> str:
    lifetime = str(config.get("rollout_worker_lifetime", "slot")).strip().lower()
    if lifetime not in {"slot", "episode"}:
        raise ValueError(
            "rollout_worker_lifetime must be 'slot' or 'episode', "
            f"got {lifetime!r}"
        )
    return lifetime


def resolve_rollout_worker_retries(config: dict) -> int:
    retries = int(config.get("rollout_worker_retries", 1))
    if retries < 0:
        raise ValueError(f"rollout_worker_retries must be >= 0, got {retries}")
    return retries


def resolve_rollout_worker_start_timeout_seconds(config: dict) -> float:
    timeout = float(config.get("rollout_worker_start_timeout_seconds", 120))
    if timeout <= 0:
        raise ValueError(
            "rollout_worker_start_timeout_seconds must be > 0, "
            f"got {timeout}"
        )
    return timeout


def resolve_rollout_action_timeout_seconds(config: dict) -> float:
    timeout = float(config.get("rollout_action_timeout_seconds", 300))
    if timeout <= 0:
        raise ValueError(f"rollout_action_timeout_seconds must be > 0, got {timeout}")
    return timeout


def resolve_policy_batch_timeout_ms(config: dict) -> int:
    timeout_ms = int(config.get("policy_batch_timeout_ms", 10))
    if timeout_ms < 0:
        raise ValueError(f"policy_batch_timeout_ms must be >= 0, got {timeout_ms}")
    return timeout_ms


def apply_rollout_config_defaults(config: dict) -> dict:
    resolved = dict(config)
    resolved["rollout_worker_num"] = resolve_rollout_worker_num(resolved)
    resolved["rollout_worker_lifetime"] = resolve_rollout_worker_lifetime(resolved)
    resolved["rollout_worker_retries"] = resolve_rollout_worker_retries(resolved)
    resolved["rollout_worker_start_timeout_seconds"] = (
        resolve_rollout_worker_start_timeout_seconds(resolved)
    )
    resolved["rollout_action_timeout_seconds"] = resolve_rollout_action_timeout_seconds(resolved)
    resolved["policy_batch_timeout_ms"] = resolve_policy_batch_timeout_ms(resolved)
    return resolved


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
