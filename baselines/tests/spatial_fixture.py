"""Small deterministic spatial observations for both backend test suites."""
import numpy as np

from baselines.config import normalize_baseline_config


def spatial_config(algorithm="mlp_bc", architecture="cnn_map", env_family="pointmaze"):
    return normalize_baseline_config({
        "algorithm": algorithm, "env_family": env_family,
        "train_variants": ["umaze"], "device": False,
        "n_steps": 2, "n_steps_per_epoch": 2, "show_progress": False,
        "observation": {"include_map": True, "include_dynamic_map": True,
                        "include_location_sensing": True, "include_wall_sensing": True},
        "network": {"architecture": architecture, "hidden_units": [8, 8],
                    "spatial": {"cnn_channels": [4], "map_dim": 8,
                                "vector_hidden_units": [8], "fusion_hidden_units": [8],
                                "transformer_dim": 8, "transformer_layers": 1,
                                "transformer_heads": 2, "transformer_ffn_dim": 16}},
        "algorithm_config": {"batch_size": 4},
    })


def spatial_run_config(directory, algorithm, architecture):
    from pathlib import Path
    from baselines.tests.seed_map_fixture import make_seed_map_corpus

    config = spatial_config(algorithm, architecture)
    corpus = make_seed_map_corpus(Path(directory) / "corpus", env_family="pointmaze")
    config.update({
        "device": "cpu", "output_root": str(Path(directory) / "runs"),
        "seed_map_train": {"enabled": True, "dataset_path": str(corpus),
                           "split_unit": "trajectory", "selection_seed": 0,
                           "trajectories_per_seed": 2},
        "seed_map_eval": {"enabled": True, "seed_ranges": [[1001, 1002]],
                          "seed_map_version": "v1", "seed_map_size_mode": "fixed",
                          "seed_map_fixed_rows": 7, "seed_map_fixed_cols": 7,
                          "max_episode_steps": 2},
    })
    config["evaluation"].update({"every_epochs": 1, "num_episodes": 1})
    config["algorithm_config"]["batch_size"] = 2
    if algorithm == "crl":
        config["algorithm_config"]["latent_dim"] = 8
    return config


def spatial_values(layout, count=9):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(count, layout["dimension"])).astype(np.float32)
    rows, cols = layout["shape"]
    for item in layout["maps"]:
        grids = np.zeros((count, rows, cols), dtype=np.float32)
        grids[:, -1, :] = -1
        grids[:, 0, :] = 1
        if item["name"] == "dynamic_map":
            for i in range(count):
                grids[i, 1 + i % (rows - 2), 1] = 2
                grids[i, 2, 2] = 3
            grids[0, 2, 2] = 4
        values[:, item["start"]:item["end"]] = grids.reshape(count, -1)
    return values
