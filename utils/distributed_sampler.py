from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler[int]):
    """Weighted replacement sampler that shards one global draw across DDP ranks."""

    def __init__(
        self,
        weights: Sequence[float],
        *,
        num_replicas: int,
        rank: int,
        num_samples: int | None = None,
        replacement: bool = True,
        seed: int = 0,
    ):
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if not replacement:
            raise ValueError("DistributedWeightedSampler currently requires replacement=True")

        self.weights = torch.as_tensor(list(weights), dtype=torch.double)
        if self.weights.ndim != 1 or self.weights.numel() == 0:
            raise ValueError("weights must be a non-empty 1D sequence")
        if torch.any(self.weights < 0):
            raise ValueError("weights must be non-negative")
        if float(self.weights.sum().item()) <= 0.0:
            raise ValueError("at least one weight must be positive")

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.requested_num_samples = int(num_samples) if num_samples is not None else len(self.weights)
        if self.requested_num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.requested_num_samples}")
        self.num_samples = int(math.ceil(self.requested_num_samples / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = True
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
