from __future__ import annotations

import torch

from utils.distributed import DistributedContext, reduce_mean


class WandbLogger:
    def __init__(self, run=None):
        self.run = run

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict) -> None:
        if self.run is None:
            return
        self.run.log(metrics)

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()
        self.run = None


def _as_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Expected a boolean value, got {value!r}")
    return bool(value)


def wandb_enabled(config: dict) -> bool:
    return _as_bool(config.get("wandb_enabled", False))


def _optional_config_string(config: dict, key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def init_wandb_logger(config: dict, dist_context: DistributedContext) -> WandbLogger:
    if not wandb_enabled(config):
        return WandbLogger()

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb_enabled=true but the wandb package is not installed. "
            "Install it with: micromamba run -n llm_offline python -m pip install wandb"
        ) from exc

    if not dist_context.is_main_process:
        return WandbLogger()

    project = _optional_config_string(config, "wandb_project") or "llm_offline"
    entity = _optional_config_string(config, "wandb_entity")
    mode = _optional_config_string(config, "wandb_mode")
    run_name = str(config["experiment_id"])
    run = wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=run_name,
        config=dict(config),
        job_type="train",
    )
    run.define_metric("train/env_steps")
    run.define_metric("train/*", step_metric="train/env_steps")
    run.define_metric("val/*", step_metric="train/env_steps")
    run.define_metric("eval/*", step_metric="train/env_steps")
    for variant in config.get("resolved_eval_variants", []) or []:
        run.define_metric(f"eval/{variant}/*", step_metric="train/env_steps")
    print(f"[wandb] Initialized run: project={project}, name={run_name}")
    return WandbLogger(run)


def wandb_log_interval(config: dict) -> int:
    interval = int(config.get("wandb_log_interval", 10))
    if interval < 1:
        raise ValueError(f"wandb_log_interval must be >= 1, got {interval}")
    return interval


def prompt_template_multiplier(config: dict) -> int:
    prompt_names = config.get("prompt_templete_index") or []
    if not isinstance(prompt_names, list):
        return 1
    return max(len(prompt_names), 1)


def global_batch_sample_count(
    batch: dict,
    dist_context: DistributedContext,
    device: torch.device,
) -> int:
    local_batch_size = int(batch["input_ids"].shape[0])
    mean_batch_size = reduce_mean(float(local_batch_size), dist_context, device)
    return int(round(mean_batch_size * dist_context.world_size))


def wandb_step_metrics(
    *,
    env_steps: float,
    batch_step: int | None,
    optimizer_step: int | None,
    epoch: int | None,
) -> dict:
    metrics = {"train/env_steps": env_steps}
    if batch_step is not None:
        metrics["train/batch_step"] = batch_step
    if optimizer_step is not None:
        metrics["train/optimizer_step"] = optimizer_step
    if epoch is not None:
        metrics["train/epoch"] = epoch
    return metrics
