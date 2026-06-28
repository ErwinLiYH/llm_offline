from __future__ import annotations


def _format_index(value, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    return str(int(value))


def format_step_tag(step: int | None) -> str:
    return f"step{_format_index(step, 'step')}"


def format_epoch_tag(epoch: int | None, step: int | None = None) -> str:
    epoch_text = _format_index(epoch, "epoch")
    if step is None:
        return f"ep{epoch_text}"
    return f"ep{epoch_text}({format_step_tag(step)})"


def format_training_eval_tag(
    eval_type: str,
    *,
    epoch: int | None = None,
    batch_step: int | None = None,
) -> str:
    if eval_type == "step":
        return format_step_tag(batch_step)
    if eval_type == "epoch":
        return format_epoch_tag(epoch, batch_step)
    raise ValueError(f"Unknown training eval type: {eval_type!r}")
