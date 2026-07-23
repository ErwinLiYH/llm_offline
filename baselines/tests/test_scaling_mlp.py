from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from baselines.algorithms import create_algorithm, load_algorithm
from baselines.algorithms.scaling_mlp import (
    DenseLayerNormSwish,
    PaperResidualBlock,
    ScalingMLPEncoder,
    ScalingMLPEncoderFactory,
)
from baselines.config import normalize_baseline_config
from baselines.data.transitions import MinariTransitionEpisode, build_replay_buffer
from baselines.runner import model_metadata


class ScalingMLPEncoderTest(unittest.TestCase):
    def test_residual_body_matches_paper_depth_and_identity_path(self):
        encoder = ScalingMLPEncoderFactory(
            body_type="residual_mlp", width=8, body_depth=8
        ).create((6,))
        self.assertIsInstance(encoder, ScalingMLPEncoder)
        self.assertEqual(encoder.body_dense_count, 8)
        self.assertEqual(encoder.encoder_dense_count, 9)
        self.assertEqual(len(encoder.body), 2)
        self.assertTrue(all(isinstance(block, PaperResidualBlock) for block in encoder.body))
        self.assertTrue(
            all(len(block.units) == 4 for block in encoder.body)
        )
        self.assertEqual(
            sum(isinstance(module, nn.Linear) for module in encoder.modules()), 9
        )
        self.assertEqual(
            sum(isinstance(module, nn.LayerNorm) for module in encoder.modules()), 9
        )
        self.assertEqual(
            sum(isinstance(module, nn.SiLU) for module in encoder.modules()), 9
        )
        self.assertTrue(
            all(
                torch.count_nonzero(module.bias) == 0
                for module in encoder.modules()
                if isinstance(module, nn.Linear)
            )
        )

        # A zero residual branch must preserve the block input exactly. This
        # verifies that the skip is after all four Dense-LN-Swish units.
        block = encoder.body[0]
        for unit in block.units:
            nn.init.zeros_(unit.dense.weight)
            nn.init.zeros_(unit.dense.bias)
        values = torch.randn(3, 8)
        torch.testing.assert_close(block(values), values)
        self.assertEqual(encoder(torch.randn(5, 6)).shape, (5, 8))

    def test_plain_body_has_the_same_units_without_skip_connections(self):
        encoder = ScalingMLPEncoderFactory(
            body_type="plain_mlp", width=7, body_depth=5
        ).create((4,))
        self.assertEqual(len(encoder.body), 5)
        self.assertTrue(
            all(isinstance(unit, DenseLayerNormSwish) for unit in encoder.body)
        )
        self.assertEqual(
            sum(isinstance(module, nn.Linear) for module in encoder.modules()), 6
        )
        self.assertEqual(encoder(torch.randn(2, 4)).shape, (2, 7))

    def test_invalid_residual_depth_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ScalingMLPEncoderFactory(
                body_type="residual_mlp", width=8, body_depth=6
            ).create((6,))


class ScalingMLPIntegrationTest(unittest.TestCase):
    def _buffer(self):
        rng = np.random.default_rng(19)
        episodes = []
        for _ in range(3):
            episodes.append(
                MinariTransitionEpisode(
                    observations=rng.normal(size=(9, 6)).astype(np.float32),
                    actions=np.tanh(rng.normal(size=(8, 2))).astype(np.float32),
                    rewards=np.zeros(8, dtype=np.float32),
                    terminated=True,
                    truncated=False,
                    source_variant="synthetic",
                )
            )
        return build_replay_buffer(episodes)

    def _config(self, *, architecture: str = "residual_mlp", body_depth: int = 4):
        return normalize_baseline_config(
            {
                "algorithm": "mlp_bc",
                "train_variants": ["umaze"],
                "device": False,
                "n_steps": 1,
                "n_steps_per_epoch": 1,
                "show_progress": False,
                "network": {
                    "architecture": architecture,
                    "width": 16,
                    "body_depth": body_depth,
                    "block_size": 4,
                    "activation": "swish",
                    "use_batch_norm": False,
                    "use_layer_norm": True,
                    "dropout_rate": None,
                },
                "algorithm_config": {"batch_size": 4},
            }
        )

    def test_scaling_config_enforces_the_paper_recipe(self):
        config = self._config()
        self.assertEqual(config["network"]["hidden_units"], None)
        self.assertEqual(config["network"]["architecture"], "residual_mlp")
        with self.assertRaisesRegex(ValueError, "divisible"):
            self._config(body_depth=5)
        with self.assertRaisesRegex(ValueError, "only by algorithm=mlp_bc"):
            raw = self._config()
            raw["algorithm"] = "iql"
            normalize_baseline_config(raw)

    def test_d3rlpy_bc_updates_and_checkpoint_roundtrips(self):
        config = self._config()
        algo = create_algorithm(config)
        history = algo.fit(
            self._buffer(),
            n_steps=1,
            n_steps_per_epoch=1,
            show_progress=False,
            logger_adapter=_NoopLoggerFactory(),
            save_interval=2,
        )
        self.assertTrue(np.isfinite(history[0][1]["loss"]))
        action_input = np.zeros((3, 6), dtype=np.float32)
        before = algo.predict(action_input)
        self.assertEqual(before.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(before)))
        self.assertTrue(np.all(np.abs(before) <= 1.0))

        policy = algo.impl.policy
        policy_parameter_ids = {
            id(parameter) for parameter in policy.parameters() if parameter.requires_grad
        }
        optimizer_parameter_ids = {
            id(parameter)
            for group in algo.impl.policy_optim.param_groups
            for parameter in group["params"]
        }
        self.assertSetEqual(policy_parameter_ids, optimizer_parameter_ids)
        self.assertEqual(
            sum(isinstance(module, nn.Linear) for module in policy.modules()), 6
        )
        metadata = model_metadata(algo, config)
        self.assertEqual(metadata["trainable_parameter_count"], sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ))
        self.assertEqual(metadata["scaling_encoder"]["body_dense_count"], 4)
        self.assertEqual(metadata["scaling_encoder"]["total_dense_count"], 6)

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/model.d3"
            algo.save(path)
            loaded = load_algorithm(path, device=False)
            np.testing.assert_allclose(loaded.predict(action_input), before)


class _NoopLogger:
    def write_params(self, params):
        pass

    def before_write_metric(self, epoch, step):
        pass

    def write_metric(self, epoch, step, name, value):
        pass

    def after_write_metric(self, epoch, step):
        pass

    def save_model(self, epoch, algo):
        pass

    def close(self):
        pass

    def watch_model(self, epoch, step):
        pass


class _NoopLoggerFactory:
    def create(self, algo, experiment_name, n_steps_per_epoch):
        return _NoopLogger()


if __name__ == "__main__":
    unittest.main()
