import h5py
import minari
from minari import MinariDataset

from data.antmaze.variants import (
    ANTMAZE_VARIANTS,
    get_antmaze_variant_type,
    resolve_local_dataset_path,
)
from data.pointmaze.dataset import (
    PointMazeDataset,
    _load_local_hdf5_episodes,
)


def _local_antmaze_dataset_step_signature(meta: dict) -> str:
    dataset_root = resolve_local_dataset_path(meta["dataset_path"])
    data_path = dataset_root / "data"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Local AntMaze dataset not found at {data_path}. "
            "Generate it with local_antmaze_gen.py first."
        )
    try:
        dataset = MinariDataset(data_path)
        total_steps = int(dataset.total_steps)
    except ValueError as exc:
        if "No data found in data path" not in str(exc):
            raise
        h5_path = data_path / "main_data.hdf5"
        if not h5_path.exists():
            raise FileNotFoundError(
                f"Local AntMaze data file not found at {h5_path}. "
                "Generate it with local_antmaze_gen.py first."
            )
        with h5py.File(h5_path, "r") as f:
            if "total_steps" in f.attrs:
                total_steps = int(f.attrs["total_steps"])
            else:
                total_steps = sum(
                    int(f[name]["actions"].shape[0])
                    for name in f.keys()
                    if name.startswith("episode_")
                )
    return f"localsteps{total_steps}"


class AntMazeDataset(PointMazeDataset):
    """D4RL AntMaze dataset using the shared goal-maze tokenization pipeline."""

    ENV_FAMILY = "antmaze"
    VARIANTS = ANTMAZE_VARIANTS
    ACTION_DIM = 8
    CACHE_FORMAT = "antmaze_hash_signature_v3"

    @classmethod
    def _load_variant_episodes(cls, variant: str):
        if variant not in cls.VARIANTS:
            raise ValueError(f"Unknown AntMaze variant: {variant}")
        meta = cls.VARIANTS[variant]
        if cls._get_variant_type(meta) == "local":
            dataset_root = resolve_local_dataset_path(meta["dataset_path"])
            data_path = dataset_root / "data"
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Local AntMaze dataset for variant={variant!r} not found at {data_path}. "
                    "Generate it with local_antmaze_gen.py first."
                )
            try:
                dataset = MinariDataset(data_path)
            except ValueError as exc:
                if "No data found in data path" not in str(exc):
                    raise
                episodes = _load_local_hdf5_episodes(data_path)
                step_counts = [len(episode.actions) for episode in episodes]
                return meta, episodes, step_counts
        else:
            dataset = minari.load_dataset(meta["dataset_id"], download=True)
        episodes = list(dataset.iterate_episodes())
        step_counts = [len(episode.actions) for episode in episodes]
        return meta, episodes, step_counts

    @classmethod
    def _get_variant_type(cls, meta: dict) -> str:
        return get_antmaze_variant_type(meta)

    @classmethod
    def _local_data_signature(cls, meta: dict) -> str | None:
        if cls._get_variant_type(meta) != "local":
            return None
        return _local_antmaze_dataset_step_signature(meta)
