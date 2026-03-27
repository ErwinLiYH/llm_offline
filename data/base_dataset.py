from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseOfflineDataset(ABC, Dataset):
    """Abstract base class for offline RL datasets."""

    @abstractmethod
    def load(self, variant: str, split: str):
        """Load episodes for the given variant and split (train/val)."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
