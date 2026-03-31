import json
import os
import pickle
import threading
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

import minari
from tqdm import tqdm

from data.base_dataset import BaseOfflineDataset
from data.pointmaze.variants import POINTMAZE_VARIANTS
from data.pointmaze import formatting
from utils.prompt_loader import load_templates


class PointMazeDataset(BaseOfflineDataset):
    """PyTorch Dataset for PointMaze behavior cloning.

    Loads a Minari offline dataset, splits episodes 9:1 (train/val),
    and expands each timestep into 5 samples (one per prompt template).
    Loss is computed only on the action target tokens (prompt labels = -100).
    """

    def __init__(
        self,
        variant: str,
        split: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        num_workers: int = 8,
        cache_dir: str | None = None,
        max_data_num: int | None = None,
    ):
        super().__init__()
        self.variant = variant
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.max_data_num = max_data_num

        self._local = threading.local()
        self._samples: list[dict] = []
        self.load(variant, split)

    # ------------------------------------------------------------------
    def _cache_path(self, variant: str, split: str) -> str | None:
        if self.cache_dir is None:
            return None
        fname = f"pointmaze-{variant}-{split}.pkl"
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

        meta = POINTMAZE_VARIANTS[variant]
        dataset = minari.load_dataset(meta["dataset_id"], download=True)
        templates = load_templates("pointmaze", variant)

        all_episodes = list(dataset.iterate_episodes())
        n_total = len(all_episodes)
        n_train = max(1, int(n_total * 0.9))

        if split == "train":
            episodes = all_episodes[:n_train]
        else:
            episodes = all_episodes[n_train:]

        def process_episode(episode) -> list[tuple[dict, dict]]:
            results = []
            obs_arr = episode.observations["observation"]   # (T+1, 4)
            goal_arr = episode.observations["desired_goal"] # (T+1, 2)
            actions = episode.actions                       # (T, 2)
            T = len(actions)
            for t in range(T):
                obs = obs_arr[t].astype(np.float32)
                goal = goal_arr[t].astype(np.float32)
                action = actions[t].astype(np.float32)
                obs_text = formatting.format_obs(obs, goal)
                action_text = formatting.format_action(action)
                for template in templates:
                    prompt = template.format(obs_text=obs_text)
                    token_sample = self._tokenize(prompt, action_text)
                    text_record = {"prompt": prompt, "action": action_text}
                    results.append((text_record, token_sample))
            return results

        num_workers = min(os.cpu_count() or 1, self.num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = list(tqdm(
                executor.map(process_episode, episodes),
                total=len(episodes),
                desc=f"Tokenizing [{split}]",
            ))

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

    # ------------------------------------------------------------------
    def _get_local_tokenizer(self):
        """Return a thread-local tokenizer instance (fast tokenizers are not thread-safe)."""
        if not hasattr(self._local, "tokenizer"):
            self._local.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer.name_or_path, trust_remote_code=True
            )
        return self._local.tokenizer

    def _tokenize(self, prompt: str, action_text: str) -> dict:
        tok = self._get_local_tokenizer()
        prompt_ids = tok(
            prompt,
            add_special_tokens=True,
        ).input_ids
        prompt_len = len(prompt_ids)

        full_text = prompt + action_text
        full_enc = tok(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        labels = list(input_ids)
        # Mask prompt tokens from loss
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # ------------------------------------------------------------------
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

        input_ids_list.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
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
