from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


TRAINER_STATE_VERSION = 1


def capture_trainer_state(*, epoch: int, step: int) -> dict:
    """Capture RNG state needed to continue offline minibatch sampling exactly."""

    return {
        "version": TRAINER_STATE_VERSION,
        "epoch": int(epoch),
        "step": int(step),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def save_trainer_state(path: str | Path, *, epoch: int, step: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(capture_trainer_state(epoch=epoch, step=step), path)


def load_trainer_state(path: str | Path) -> dict:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("version") != TRAINER_STATE_VERSION:
        raise ValueError(f"Unsupported trainer state file: {path}")
    required = {
        "epoch",
        "step",
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Trainer state file is missing fields: {missing}")
    return state


def restore_trainer_state(state: dict) -> None:
    if state.get("version") != TRAINER_STATE_VERSION:
        raise ValueError(
            f"Unsupported trainer state version: {state.get('version')!r}"
        )
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = state["torch_cuda_rng_state_all"]
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Trainer state contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(cuda_states)
