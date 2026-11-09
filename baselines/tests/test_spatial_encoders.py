from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from baselines.algorithms import create_algorithm
from baselines.algorithms.d3rlpy_factory import load_algorithm, resume_algorithm
from baselines.algorithms.spatial_encoder import SpatialEncoderFactory, SpatialObservationScaler, map_channels
from baselines.config import normalize_baseline_config
from baselines.data.spatial import map_indices, spatial_layout
from baselines.data.transitions import MinariTransitionEpisode, build_replay_buffer
from baselines.tests.spatial_fixture import spatial_config, spatial_values, spatial_run_config
from baselines.tests.test_algorithms import _NoopLoggerFactory


class SpatialEncoderTest(unittest.TestCase):
    def test_configuration_rejects_invalid_combinations(self):
        for override in (
            {"observation": {}},
            {"algorithm": "hiql"},
            {"network": {"architecture": "transformer_map", "spatial": {"transformer_dim": 10}}},
            {"network": {"architecture": "cnn_map", "spatial": {"cnn_channels": []}}},
        ):
            raw = {"train_variants": ["umaze"], "network": {"architecture": "cnn_map"},
                   "observation": {"include_map": True}}
            raw.update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                normalize_baseline_config(raw)

    def test_channels_scaler_and_gradient_paths(self):
        config = spatial_config()
        layout = spatial_layout(config["env_family"], config["observation"])
        values = spatial_values(layout)
        indices = map_indices(layout)
        scaler = SpatialObservationScaler(layout=layout, mean=values.mean(0), std=values.std(0))
        scaled = scaler.transform_numpy(values)
        np.testing.assert_array_equal(scaled[:, indices], values[:, indices])
        np.testing.assert_allclose(scaler.reverse_transform_numpy(scaled), values, atol=1e-6)
        tensor = torch.from_numpy(values)
        torch_scaled = scaler.transform(tensor)
        torch.testing.assert_close(torch_scaled[:, indices], tensor[:, indices], rtol=0, atol=0)
        torch.testing.assert_close(scaler.reverse_transform(torch_scaled), tensor)
        channels, valid = map_channels(tensor, layout)
        self.assertFalse(valid[:, -1].any())
        self.assertEqual(channels[0, 2, 2, 2].item(), 1)  # current in dynamic map
        self.assertEqual(channels[0, 2, 2, 3].item(), 1)  # goal in dynamic map
        for architecture in ("cnn_map", "transformer_map"):
            factory = SpatialEncoderFactory(layout, architecture, config["network"]["spatial"])
            encoder = factory.create((layout["dimension"],))
            output = encoder(torch_scaled)
            self.assertEqual(tuple(output.shape), (len(values), 8))
            output.square().mean().backward()
            for name, parameter in encoder.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(sum(p.grad.abs().sum().item() for p in encoder.vector_net.parameters()), 0)
            map_module = encoder.convs if architecture == "cnn_map" else encoder.token_projection
            self.assertGreater(sum(p.grad.abs().sum().item() for p in map_module.parameters()), 0)
            critic = factory.create_with_action((layout["dimension"],), 2)
            actions = torch.ones(len(values), 2, requires_grad=True)
            critic(torch_scaled, actions).sum().backward()
            self.assertGreater(actions.grad.abs().sum().item(), 0)

    def test_five_map_layouts(self):
        for family in ("pointmaze", "antmaze"):
            for observation in ({"include_map": True}, {"include_dynamic_map": True},
                                {"include_map": True, "include_dynamic_map": True}):
                layout = spatial_layout(family, observation)
                self.assertEqual(len(layout["vector_indices"]) + len(map_indices(layout)), layout["dimension"])
                map_channels(torch.from_numpy(spatial_values(layout)), layout)

    def test_runner_with_corpus_and_live_rollout(self):
        import json
        from baselines.runner import train_baseline

        torch.set_num_threads(1)
        for architecture in ("cnn_map", "transformer_map"):
            for algorithm in ("mlp_bc", "td3_bc", "iql", "rebrac"):
                with self.subTest(architecture=architecture, algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                    config = spatial_run_config(directory, algorithm, architecture)
                    run_dir = train_baseline(config)
                    summary = json.loads((run_dir / "summary.json").read_text())
                    self.assertEqual(summary["n_steps"], 2)
                    self.assertTrue(summary["evaluation_history"])
                    self.assertTrue((run_dir / "model_metadata.json").exists())

    def test_all_d3rlpy_algorithms_train_save_load_and_resume(self):
        torch.set_num_threads(1)
        for family in ("pointmaze", "antmaze"):
            for architecture in ("cnn_map", "transformer_map"):
                for algorithm in ("mlp_bc", "td3_bc", "iql", "rebrac"):
                    with self.subTest(family=family, architecture=architecture, algorithm=algorithm):
                        config = spatial_config(algorithm, architecture, family)
                        layout = spatial_layout(family, config["observation"])
                        values = spatial_values(layout)
                        action_dim = 2 if family == "pointmaze" else 8
                        rng = np.random.default_rng(5)
                        episode = MinariTransitionEpisode(
                            observations=values, actions=np.tanh(rng.normal(size=(8, action_dim))).astype(np.float32),
                            rewards=np.ones(8, dtype=np.float32), terminated=True,
                            truncated=False, source_variant="synthetic",
                        )
                        buffer = build_replay_buffer([episode])
                        algo = create_algorithm(config)
                        history = algo.fit(buffer, n_steps=2, n_steps_per_epoch=2, show_progress=False,
                                           logger_adapter=_NoopLoggerFactory(), save_interval=3)
                        self.assertTrue(all(np.isfinite(v) for v in history[0][1].values()))
                        expected = algo.predict(values[:2])
                        self.assertEqual(expected.shape, (2, action_dim))
                        with tempfile.TemporaryDirectory() as directory:
                            path = str(Path(directory) / "model.d3")
                            algo.save(path)
                            loaded = load_algorithm(path, device=False)
                            np.testing.assert_allclose(loaded.predict(values[:2]), expected, atol=1e-6)
                            resumed = resume_algorithm(config, buffer, path, resume_step=2)
                            self.assertEqual(resumed._grad_step, 2)
                            metrics = resumed.update(buffer.sample_transition_batch(4))
                            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))


if __name__ == "__main__":
    unittest.main()
