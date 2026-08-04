from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from crossmaze.seed_map import SeedMapSpec, generate_seed_map
from data.seed_map_corpus import (
    append_minari_shard_to_seed_map_corpus,
    create_seed_map_corpus,
    finalize_seed_map_corpus,
)


def _write_fake_shard(
    root: Path,
    *,
    env_family: str,
    episode_count: int,
) -> None:
    path = root / "data" / "main_data.hdf5"
    path.parent.mkdir(parents=True)
    observation_dim = 4 if env_family == "pointmaze" else 27
    action_dim = 2 if env_family == "pointmaze" else 8
    with h5py.File(path, "w") as handle:
        for episode_index in range(episode_count):
            group = handle.create_group(f"episode_{episode_index}")
            observations = group.create_group("observations")
            proprioception = np.zeros((4, observation_dim), dtype=np.float32)
            if env_family == "pointmaze":
                proprioception[:, 0] = np.linspace(-1.0, 1.0, 4)
            observations.create_dataset("observation", data=proprioception)
            observations.create_dataset(
                "achieved_goal",
                data=np.zeros((4, 2), dtype=np.float32),
            )
            observations.create_dataset(
                "desired_goal",
                data=np.ones((4, 2), dtype=np.float32),
            )
            group.create_dataset(
                "actions",
                data=np.tile(
                    np.linspace(-0.25, 0.25, 3, dtype=np.float32)[:, None],
                    (1, action_dim),
                ),
            )
            group.create_dataset("rewards", data=np.zeros(3, dtype=np.float32))
            group.create_dataset(
                "terminations", data=np.array([False, False, True])
            )
            group.create_dataset("truncations", data=np.zeros(3, dtype=bool))


def make_seed_map_corpus(
    parent: Path,
    *,
    env_family: str,
    seed_start: int = 1,
    seed_end: int = 3,
    trajectories_per_seed: int = 2,
    seed_map_spec: SeedMapSpec | None = None,
) -> Path:
    spec = seed_map_spec or SeedMapSpec(
        size_mode="fixed",
        fixed_rows=7,
        fixed_cols=7,
    )
    root = parent / f"{seed_start}-{seed_end}-{trajectories_per_seed}"
    create_seed_map_corpus(
        root,
        env_family=env_family,
        reward_type="sparse",
        seed_start=seed_start,
        seed_end=seed_end,
        trajectories_per_seed=trajectories_per_seed,
        seed_map_spec=spec,
    )
    for map_seed in range(seed_start, seed_end):
        shard = parent / f"shard-{env_family}-{map_seed}"
        _write_fake_shard(
            shard,
            env_family=env_family,
            episode_count=trajectories_per_seed,
        )
        append_minari_shard_to_seed_map_corpus(
            root,
            shard,
            map_seed=map_seed,
            maze_map=generate_seed_map(map_seed, spec),
            trajectory_start_index=0,
            trajectory_count=trajectories_per_seed,
        )
    finalize_seed_map_corpus(root)
    return root
