import json
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import minari
import numpy as np
import torch
from tqdm import tqdm

TQDM_BAR_FORMAT = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from data.base_dataset import BaseOfflineDataset
from data.pointmaze import formatting
from data.pointmaze.variants import POINTMAZE_VARIANTS
from utils.prompt_loader import load_templates, render_template

DEFAULT_TRAIN_DATA_RATIO = 0.9


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
        if self.history_num < 0:
            raise ValueError(f"history_num must be >= 0, got {self.history_num}")
        if self.history_stride < 1:
            raise ValueError(f"history_stride must be >= 1, got {self.history_stride}")

        meta = POINTMAZE_VARIANTS[variant]
        dataset = minari.load_dataset(meta["dataset_id"], download=True)
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

        all_episodes = list(dataset.iterate_episodes())
        n_total = len(all_episodes)
        train_end = max(1, int(n_total * self.train_data_ratio))
        train_end = min(train_end, n_total - 1)

        if split == "train":
            episodes = all_episodes[:train_end]
        elif split == "val":
            episodes = all_episodes[train_end:]
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
                    dynamic_ncols=True,
                    bar_format=TQDM_BAR_FORMAT,
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
        prompt_ids = tok(prompt, add_special_tokens=True).input_ids
        prompt_len = len(prompt_ids)

        full_enc = tok(
            prompt + action_text,
            add_special_tokens=True,
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
