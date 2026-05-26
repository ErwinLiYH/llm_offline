from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias, TypedDict

import torch
from torch.utils.data import Dataset


class VariantEpisodeStats(TypedDict):
    """Episode statistics used by train.py for multi-variant balancing."""

    variant: str
    total_episodes: int
    total_steps: int
    initial_train_target: int
    sampled_episode_target: int


TokenizedSample: TypeAlias = dict[str, list[int]]
TensorSample: TypeAlias = dict[str, torch.Tensor]
Batch: TypeAlias = dict[str, torch.Tensor]


@dataclass(frozen=True)
class DatasetBuildRequest:
    """One requested loaded dataset split.

    train.py creates one request for each selected `(variant, split)` pair, normally
    in this order:

    - `variant_a/train`
    - `variant_a/val`
    - `variant_b/train`
    - `variant_b/val`

    `BaseOfflineDataset.build_batch()` must return loaded dataset objects in the
    exact same order. A dataset implementation may batch cache checks and
    tokenization across requests, but it must preserve per-request split identity.

    Required output sample schema after `__getitem__`:

    - `input_ids`: `torch.long` tensor shaped `[seq_len]`
    - `attention_mask`: `torch.long` tensor shaped `[seq_len]`
    - `labels`: `torch.long` tensor shaped `[seq_len]`, prompt positions masked as `-100`
    - `action_bin_labels`: `torch.long` tensor shaped `[seq_len]`, non-action positions as `-1`;
      bin/gaussian_bin mark generated action-token positions, mtp_bin marks NTP/AQT prediction positions
    - `action_values`: optional `torch.float32` tensor shaped `[action_dim]` for continuous
      action modes
    - `action_query_mask` and related `action_query_*` tensors: optional metadata for mtp_bin
    """

    # Dataset identity.
    variant: str
    split: str

    # Tokenizer context. `tokenizer_name_or_path` is required if tokenizer.name_or_path is unavailable.
    tokenizer: Any
    tokenizer_name_or_path: str | None = None
    max_length: int = 512

    # Offline tokenization/cache controls.
    num_workers: int = 8
    cache_dir: str | None = None
    max_data_num: int | None = None

    # Prompt selection. `prompt_templete_index` is the historical config spelling.
    prompt_template_count: int = 1
    prompt_templete_index: list[str] | None = None

    # Episode-level train/val sampling.
    train_data_ratio: float = 0.9
    episode_keep_num: int | None = None
    balance_variant_episode_count: bool = False
    balanced_train_episode_count: int | None = None
    sampling_seed: int = 0

    # Optional prompt history.
    history_num: int = 0
    history_stride: int = 1

    # Action encoding.
    action_token_mode: str = "text"
    action_num_bins: int = 10
    action_bin_min: float = -1.0
    action_bin_max: float = 1.0
    new_token: bool = False
    action_dim: int | None = None
    mtp_k: int | None = None

    # File progress update cadence for expensive dataset construction.
    progress_interval_seconds: float = 5.0


class BaseOfflineDataset(ABC, Dataset):
    """Abstract base class for offline RL datasets.

    Implementations are loaded through `build_batch()` only. Direct construction
    should create an already-loaded dataset object or a lightweight container;
    it should not perform independent offline tokenization outside `build_batch()`.
    """

    @classmethod
    @abstractmethod
    def build_batch(cls, requests: list[DatasetBuildRequest]) -> list["BaseOfflineDataset"]:
        """Build loaded datasets in the same order as `requests`.

        Contract:
        - Return length must equal `len(requests)`.
        - Return item `i` must correspond to request `i`.
        - Cache hits may be loaded immediately.
        - Cache misses may share tokenization workers across requests.
        - Implementations must preserve the sample schema documented on
          `DatasetBuildRequest`.
        """

    @classmethod
    @abstractmethod
    def collect_variant_episode_stats(
        cls,
        variant: str,
        episode_keep_num: int | None,
    ) -> VariantEpisodeStats:
        """Return episode statistics used by multi-variant balancing.

        `sampled_episode_target` must match the number of episodes this dataset
        family would include in the pre-split sampled pool for `episode_keep_num`.
        train.py uses the minimum target across selected variants when
        `balance_variant_episode_count` is enabled.
        """

    @classmethod
    def get_action_dim(cls, variants: list[str]) -> int:
        """Return the flat action dimension for the selected variants."""
        raise NotImplementedError(f"{cls.__name__} must implement get_action_dim().")

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> TensorSample:
        pass

    @staticmethod
    def collate_fn(batch: list[TensorSample]) -> Batch:
        """Pad tokenized sequence samples to the same length.

        Padding values:
        - `input_ids`: `0`
        - `attention_mask`: `0`
        - `labels`: `-100`
        - `action_bin_labels`: `-1`
        - `action_query_mask`: `False` if present in a legacy/custom sample
        """
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids_list, attention_mask_list, labels_list, action_bin_labels_list = [], [], [], []
        has_position_ids = any("position_ids" in item for item in batch)
        position_ids_list = []
        has_action_query_mask = any("action_query_mask" in item for item in batch)
        action_query_mask_list = []
        action_query_fields = {
            "action_query_offsets": -1,
            "action_query_source_positions": -1,
            "action_query_anchor_positions": -1,
            "action_query_prev_token_ids": 0,
        }
        has_action_query_field = {
            key: any(key in item for item in batch) for key in action_query_fields
        }
        action_query_field_lists = {key: [] for key in action_query_fields}
        has_action_values = any("action_values" in item for item in batch)
        action_values_list = []

        for item in batch:
            if has_position_ids and "position_ids" not in item:
                raise ValueError("Cannot collate a mixed batch with and without position_ids.")
            if has_action_query_mask and "action_query_mask" not in item:
                raise ValueError("Cannot collate a mixed batch with and without action_query_mask.")
            for key, enabled in has_action_query_field.items():
                if enabled and key not in item:
                    raise ValueError(f"Cannot collate a mixed batch with and without {key}.")
            if has_action_values and "action_values" not in item:
                raise ValueError("Cannot collate a mixed batch with and without action_values.")
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len
            input_ids_list.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
            attention_mask_list.append(
                torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
            action_bin_labels_list.append(
                torch.cat([item["action_bin_labels"], torch.full((pad_len,), -1, dtype=torch.long)])
            )
            if has_position_ids:
                position_ids_list.append(
                    torch.cat([item["position_ids"], torch.zeros(pad_len, dtype=torch.long)])
                )
            if has_action_query_mask:
                action_query_mask_list.append(
                    torch.cat(
                        [
                            item["action_query_mask"].to(dtype=torch.bool),
                            torch.zeros(pad_len, dtype=torch.bool),
                        ]
                    )
                )
            for key, pad_value in action_query_fields.items():
                if has_action_query_field[key]:
                    action_query_field_lists[key].append(
                        torch.cat(
                            [
                                item[key].to(dtype=torch.long),
                                torch.full((pad_len,), pad_value, dtype=torch.long),
                            ]
                        )
                    )
            if has_action_values:
                action_values_list.append(item["action_values"].to(dtype=torch.float32))

        collated = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
            "action_bin_labels": torch.stack(action_bin_labels_list),
        }
        if has_position_ids:
            collated["position_ids"] = torch.stack(position_ids_list)
        if has_action_query_mask:
            collated["action_query_mask"] = torch.stack(action_query_mask_list)
        for key, enabled in has_action_query_field.items():
            if enabled:
                collated[key] = torch.stack(action_query_field_lists[key])
        if has_action_values:
            collated["action_values"] = torch.stack(action_values_list)
        return collated
