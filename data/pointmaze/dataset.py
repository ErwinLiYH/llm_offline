import json
import os
import pickle
import threading
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor

import minari
import numpy as np
import torch
from tqdm import tqdm

TQDM_BAR_FORMAT = "{desc} {percentage:3.0f}% {n_fmt}/{total_fmt} elapsed={elapsed} eta={remaining}"
TQDM_KWARGS = {
    "bar_format": TQDM_BAR_FORMAT,
    "dynamic_ncols": False,
    "ncols": 100,
    "nrows": 100,
    "mininterval": 5.0,
}

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from data.base_dataset import BaseOfflineDataset
from data.pointmaze import formatting
from data.pointmaze.variants import POINTMAZE_VARIANTS
from utils.chat_template import build_generation_prompt, build_training_conversation
from utils.prompt_loader import load_templates, render_template

DEFAULT_TRAIN_DATA_RATIO = 0.9
DEFAULT_EPISODE_KEEP_RATIO = 1.0
DEFAULT_SAMPLING_SEED = 0


def _load_variant_episodes(variant: str):
    meta = POINTMAZE_VARIANTS[variant]
    dataset = minari.load_dataset(meta["dataset_id"], download=True)
    episodes = list(dataset.iterate_episodes())
    step_counts = [len(episode.actions) for episode in episodes]
    return meta, episodes, step_counts


def _compute_train_episode_target(total_episodes: int, episode_keep_ratio: float) -> int:
    if total_episodes < 1:
        raise ValueError("Offline dataset contains no episodes.")
    if not (0.0 < episode_keep_ratio <= 1.0):
        raise ValueError(
            "Invalid episode_keep_ratio: expected 0 < episode_keep_ratio <= 1, "
            f"got episode_keep_ratio={episode_keep_ratio}"
        )
    return min(total_episodes, max(1, math.floor(total_episodes * episode_keep_ratio)))


def _variant_sampling_seed(variant: str, sampling_seed: int) -> int:
    digest = hashlib.sha256(f"{variant}:{sampling_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def collect_variant_episode_stats(variant: str, episode_keep_ratio: float) -> dict:
    _, episodes, step_counts = _load_variant_episodes(variant)
    total_episodes = len(episodes)
    total_steps = sum(step_counts)
    initial_train_target = _compute_train_episode_target(total_episodes, episode_keep_ratio)
    return {
        "variant": variant,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "initial_train_target": initial_train_target,
    }


def select_variant_episode_indices(
    variant: str,
    train_data_ratio: float,
    episode_keep_ratio: float,
    sampling_seed: int,
    balanced_train_target: int | None = None,
) -> dict:
    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(
            "Invalid train_data_ratio: expected 0 < train_data_ratio < 1, "
            f"got train_data_ratio={train_data_ratio}"
        )
    if not isinstance(sampling_seed, int):
        raise ValueError(f"sampling_seed must be an int, got {type(sampling_seed).__name__}")

    _, episodes, step_counts = _load_variant_episodes(variant)
    total_episodes = len(episodes)
    total_steps = sum(step_counts)
    initial_train_target = _compute_train_episode_target(total_episodes, episode_keep_ratio)
    train_target = initial_train_target if balanced_train_target is None else balanced_train_target
    train_target = min(total_episodes, train_target)
    if train_target < 1:
        raise ValueError(f"train_target must be >= 1 for variant={variant}, got {train_target}")

    val_target = math.floor(train_target * (1.0 - train_data_ratio) / train_data_ratio)
    rng = np.random.default_rng(_variant_sampling_seed(variant, sampling_seed))
    permutation = rng.permutation(total_episodes).tolist()
    train_indices = sorted(permutation[:train_target])
    remaining_indices = permutation[train_target:]
    actual_val_target = min(val_target, len(remaining_indices))
    val_indices = sorted(remaining_indices[:actual_val_target])

    train_steps = sum(step_counts[idx] for idx in train_indices)
    val_steps = sum(step_counts[idx] for idx in val_indices)

    val_shortfall_reason = None
    if actual_val_target < val_target:
        val_shortfall_reason = (
            f"requested {val_target} val episodes but only {len(remaining_indices)} remained "
            f"after reserving {train_target} train episodes"
        )

    return {
        "variant": variant,
        "episodes": episodes,
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "initial_train_target": initial_train_target,
        "balanced_train_target": balanced_train_target,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_episode_count": len(train_indices),
        "val_episode_count": len(val_indices),
        "train_steps": train_steps,
        "val_steps": val_steps,
        "val_target": val_target,
        "val_shortfall_reason": val_shortfall_reason,
    }


class PointMazeDataset(BaseOfflineDataset):
    """PyTorch Dataset for PointMaze behavior cloning."""

    def __init__(
        self,
        variant: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        num_workers: int = 8,
        cache_dir: str | None = None,
        max_data_num: int | None = None,
        prompt_template_count: int = 1,
        train_data_ratio: float = DEFAULT_TRAIN_DATA_RATIO,
        episode_keep_ratio: float = DEFAULT_EPISODE_KEEP_RATIO,
        balance_variant_episode_count: bool = False,
        balanced_train_episode_count: int | None = None,
        sampling_seed: int = DEFAULT_SAMPLING_SEED,
        history_num: int = 0,
        history_stride: int = 1,
    ):
        super().__init__()
        self.variant = variant
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.max_data_num = max_data_num
        self.prompt_template_count = prompt_template_count
        self.train_data_ratio = train_data_ratio
        self.episode_keep_ratio = episode_keep_ratio
        self.balance_variant_episode_count = balance_variant_episode_count
        self.balanced_train_episode_count = balanced_train_episode_count
        self.sampling_seed = sampling_seed
        self.history_num = history_num
        self.history_stride = history_stride

        self._local = threading.local()
        self._samples: list[dict] = []
        self.load(variant, split)

    def _split_tag(self) -> str:
        train_pct = int(round(self.train_data_ratio * 100))
        return f"split{train_pct:02d}"

    def _cache_path(self, variant: str, split: str) -> str | None:
        if self.cache_dir is None:
            return None
        fname = (
            f"pointmaze-{variant}-{split}-prompts{self.prompt_template_count}-"
            f"hist{self.history_num}-stride{self.history_stride}-{self._split_tag()}.pkl"
        )
        return os.path.join(self.cache_dir, fname)

    def load(self, variant: str, split: str):
        cache_path = self._cache_path(variant, split)
        if cache_path and os.path.exists(cache_path):
            print(f"[dataset] Loading cached dataset from {cache_path}")
            print(
                "[dataset] Cached split bypasses episode sampling settings: "
                f"episode_keep_ratio={self.episode_keep_ratio}, "
                f"balance_variant_episode_count={self.balance_variant_episode_count}, "
                f"sampling_seed={self.sampling_seed}. This run did not apply them."
            )
            with open(cache_path, "rb") as f:
                self._samples = pickle.load(f)
            if self.max_data_num is not None:
                self._samples = self._samples[: self.max_data_num]
                print(f"[dataset] max_data_num={self.max_data_num}: using {len(self._samples)} samples")
            return

        if not (0.0 < self.train_data_ratio < 1.0):
            raise ValueError(
                "Invalid train_data_ratio: expected 0 < train_data_ratio < 1, "
                f"got train_data_ratio={self.train_data_ratio}"
            )
        if not (0.0 < self.episode_keep_ratio <= 1.0):
            raise ValueError(
                "Invalid episode_keep_ratio: expected 0 < episode_keep_ratio <= 1, "
                f"got episode_keep_ratio={self.episode_keep_ratio}"
            )
        if not isinstance(self.sampling_seed, int):
            raise ValueError(f"sampling_seed must be an int, got {type(self.sampling_seed).__name__}")
        if self.history_num < 0:
            raise ValueError(f"history_num must be >= 0, got {self.history_num}")
        if self.history_stride < 1:
            raise ValueError(f"history_stride must be >= 1, got {self.history_stride}")

        meta = POINTMAZE_VARIANTS[variant]
        all_templates = load_templates("pointmaze")
        if self.prompt_template_count < 1:
            raise ValueError(
                f"prompt_template_count must be >= 1, got {self.prompt_template_count}"
            )
        if self.prompt_template_count > len(all_templates):
            raise ValueError(
                "prompt_template_count exceeds available templates: "
                f"requested {self.prompt_template_count}, available {len(all_templates)}"
            )
        templates = all_templates[: self.prompt_template_count]
        prompt_vars = meta["prompt_vars"]

        selection = select_variant_episode_indices(
            variant=variant,
            train_data_ratio=self.train_data_ratio,
            episode_keep_ratio=self.episode_keep_ratio,
            sampling_seed=self.sampling_seed,
            balanced_train_target=self.balanced_train_episode_count,
        )
        all_episodes = selection["episodes"]
        balanced_target = selection["balanced_train_target"]
        balance_text = (
            "not applied"
            if balanced_target is None
            else (
                f"clipped to {selection['train_episode_count']}"
                if balanced_target < selection["initial_train_target"]
                else "enabled, unchanged"
            )
        )
        if split == "train":
            print(
                f"[dataset] Variant {variant}: total_episodes={selection['total_episodes']}, "
                f"total_steps={selection['total_steps']}, initial_train_target={selection['initial_train_target']}, "
                f"balance={balance_text}, final_train_episodes={selection['train_episode_count']}, "
                f"train_steps={selection['train_steps']}, final_val_episodes={selection['val_episode_count']}, "
                f"val_steps={selection['val_steps']}"
            )
            if selection["val_shortfall_reason"]:
                print(f"[dataset] Variant {variant}: val fallback -> {selection['val_shortfall_reason']}")

        if split == "train":
            episodes = [all_episodes[idx] for idx in selection["train_indices"]]
        elif split == "val":
            episodes = [all_episodes[idx] for idx in selection["val_indices"]]
        else:
            raise ValueError(f"Unknown split: {split!r}. Expected 'train' or 'val'.")

        def process_episode(episode) -> list[tuple[dict, dict]]:
            results = []
            obs_arr = episode.observations["observation"]
            goal_arr = episode.observations["desired_goal"]
            actions = episode.actions
            for t, action in enumerate(actions):
                obs = obs_arr[t].astype(np.float32)
                goal = goal_arr[t].astype(np.float32)
                action = action.astype(np.float32)
                obs = {
                    "observation": obs,
                    "desired_goal": goal,
                }
                action_text = formatting.format_action(action)
                history_entries = []
                if self.history_num > 0:
                    history_indices = []
                    hist_idx = t - 1
                    while hist_idx >= 0 and len(history_indices) < self.history_num:
                        history_indices.append(hist_idx)
                        hist_idx -= self.history_stride
                    history_indices.reverse()
                    for hist_t in history_indices:
                        history_entries.append(
                            {
                                "observation": obs_arr[hist_t].astype(np.float32),
                                "action_text": formatting.format_action(actions[hist_t].astype(np.float32)),
                            }
                        )
                obs_payload = formatting.format_obs(obs, prompt_vars)
                history_payload = formatting.format_history(history_entries, prompt_vars)
                for template in templates:
                    prompt = render_template(template, prompt_vars, **obs_payload, **history_payload)
                    token_sample = self._tokenize(prompt, action_text)
                    text_record = {"prompt": prompt, "action": action_text}
                    results.append((text_record, token_sample))
            return results

        num_workers = min(os.cpu_count() or 1, self.num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = list(
                tqdm(
                    executor.map(process_episode, episodes),
                    total=len(episodes),
                    desc=f"Tokenizing [{split}]",
                    **TQDM_KWARGS,
                )
            )

        text_records = []
        for episode_results in futures:
            for text_record, token_sample in episode_results:
                text_records.append(text_record)
                self._samples.append(token_sample)

        if cache_path:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self._samples, f)
            print(f"[dataset] Saved dataset cache to {cache_path}")
            jsonl_path = cache_path.replace(".pkl", ".jsonl")
            with open(jsonl_path, "w") as f:
                for record in text_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[dataset] Saved human-readable cache to {jsonl_path}")

        if self.max_data_num is not None:
            self._samples = self._samples[: self.max_data_num]
            print(f"[dataset] max_data_num={self.max_data_num}: using {len(self._samples)} samples")

    def _get_local_tokenizer(self):
        if not hasattr(self._local, "tokenizer"):
            self._local.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer.name_or_path, trust_remote_code=True
            )
        return self._local.tokenizer

    def _tokenize(self, prompt: str, action_text: str) -> dict:
        tok = self._get_local_tokenizer()
        prompt_text = build_generation_prompt(tok, prompt)
        full_text = build_training_conversation(tok, prompt, action_text)

        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)

        full_enc = tok(
            full_text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        labels = list(input_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch of variable-length sequences to the same length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids_list, attention_mask_list, labels_list = [], [], []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids_list.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask_list.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }
