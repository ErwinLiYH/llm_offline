import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

import minari

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
    ):
        super().__init__()
        self.variant = variant
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

        self._samples: list[dict] = []
        self.load(variant, split)

    # ------------------------------------------------------------------
    def load(self, variant: str, split: str):
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

        for episode in episodes:
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
                    sample = self._tokenize(prompt, action_text)
                    self._samples.append(sample)

    # ------------------------------------------------------------------
    def _tokenize(self, prompt: str, action_text: str) -> dict:
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
        ).input_ids
        prompt_len = len(prompt_ids)

        full_text = prompt + action_text
        full_enc = self.tokenizer(
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
