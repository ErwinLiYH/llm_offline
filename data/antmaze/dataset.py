import minari

from data.antmaze.variants import ANTMAZE_VARIANTS
from data.pointmaze.dataset import PointMazeDataset


class AntMazeDataset(PointMazeDataset):
    """D4RL AntMaze dataset using the shared goal-maze tokenization pipeline."""

    ENV_FAMILY = "antmaze"
    VARIANTS = ANTMAZE_VARIANTS
    ACTION_DIM = 8
    CACHE_FORMAT = "antmaze_hash_signature_v1"

    @classmethod
    def _load_variant_episodes(cls, variant: str):
        if variant not in cls.VARIANTS:
            raise ValueError(f"Unknown AntMaze variant: {variant}")
        meta = cls.VARIANTS[variant]
        dataset = minari.load_dataset(meta["dataset_id"], download=True)
        episodes = list(dataset.iterate_episodes())
        step_counts = [len(episode.actions) for episode in episodes]
        return meta, episodes, step_counts

    @classmethod
    def _get_variant_type(cls, meta: dict) -> str:
        return "remote"

    @classmethod
    def _local_data_signature(cls, meta: dict) -> None:
        return None
