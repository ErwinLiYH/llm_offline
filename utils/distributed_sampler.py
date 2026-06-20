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


class LocalShardPaddingSampler(Sampler[int]):
    """Deterministically pad one local shard to a fixed sample count.

    Unlike DistributedSampler, this sampler never draws from outside the local
    shard. Without weights it visits each local sample once per epoch before
    adding replacement padding. With weights it mirrors weighted replacement
    sampling for multi-variant balancing.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        num_samples: int,
        seed: int = 0,
        weights: Sequence[float] | None = None,
        shuffle: bool = True,
    ):
        self.dataset_size = int(dataset_size)
        if self.dataset_size < 1:
            raise ValueError(f"dataset_size must be >= 1, got {self.dataset_size}")
        self.num_samples = int(num_samples)
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.weights = None
        if weights is not None:
            self.weights = torch.as_tensor(list(weights), dtype=torch.double)
            if self.weights.ndim != 1 or self.weights.numel() != self.dataset_size:
                raise ValueError(
                    "weights must be a 1D sequence with one value per local shard sample"
                )
            if torch.any(self.weights < 0):
                raise ValueError("weights must be non-negative")
            if float(self.weights.sum().item()) <= 0.0:
                raise ValueError("at least one local shard weight must be positive")

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.weights is not None:
            indices = torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True,
                generator=generator,
            ).tolist()
            return iter(indices)

        if self.shuffle:
            base = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            base = list(range(self.dataset_size))
        if self.num_samples <= self.dataset_size:
            return iter(base[: self.num_samples])

        padding_count = self.num_samples - self.dataset_size
        padding = torch.randint(
            low=0,
            high=self.dataset_size,
            size=(padding_count,),
            generator=generator,
        ).tolist()
        return iter(base + padding)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
