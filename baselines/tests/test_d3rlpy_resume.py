from __future__ import annotations

import io
import random
import tempfile
import unittest
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from baselines.algorithms import create_algorithm, resume_algorithm
from baselines.config import normalize_baseline_config
from baselines.data.transitions import MinariTransitionEpisode, build_replay_buffer
from baselines.trainer_state import capture_trainer_state, restore_trainer_state


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


def _buffer():
    rng = np.random.default_rng(31)
    episodes = [
        MinariTransitionEpisode(
            observations=rng.normal(size=(13, 6)).astype(np.float32),
            actions=np.tanh(rng.normal(size=(12, 2))).astype(np.float32),
            rewards=rng.normal(size=12).astype(np.float32),
            terminated=True,
            truncated=False,
            source_variant="synthetic",
        )
        for _ in range(3)
    ]
    return build_replay_buffer(episodes)


def _config(algorithm: str) -> dict:
    algorithm_config = {"batch_size": 4, "compile_graph": False}
    if algorithm in {"td3_bc", "rebrac"}:
        algorithm_config["update_actor_interval"] = 2
    return normalize_baseline_config(
        {
            "algorithm": algorithm,
            "train_variants": ["umaze"],
            "device": False,
            "n_steps": 3,
            "n_steps_per_epoch": 1,
            "show_progress": False,
            "network": {"hidden_units": [8, 8]},
            "algorithm_config": algorithm_config,
        }
    )


def _saved_state(algo) -> dict:
    assert algo.impl is not None
    buffer = io.BytesIO()
    algo.impl.save_model(buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def _assert_state_equal(test: unittest.TestCase, expected, actual) -> None:
    if torch.is_tensor(expected):
        test.assertTrue(torch.is_tensor(actual))
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    elif isinstance(expected, Mapping):
        test.assertIsInstance(actual, Mapping)
        test.assertEqual(set(expected), set(actual))
        for key in expected:
            _assert_state_equal(test, expected[key], actual[key])
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        test.assertIsInstance(actual, Sequence)
        test.assertEqual(len(expected), len(actual))
        for expected_item, actual_item in zip(expected, actual, strict=True):
            _assert_state_equal(test, expected_item, actual_item)
    else:
        test.assertEqual(expected, actual)


class D3RLPyResumeTest(unittest.TestCase):
    def test_all_registered_algorithms_resume_bitwise_on_cpu(self) -> None:
        for algorithm in ("mlp_bc", "iql", "td3_bc", "rebrac"):
            with self.subTest(algorithm=algorithm):
                config = _config(algorithm)
                replay_buffer = _buffer()
                random.seed(41)
                np.random.seed(42)
                torch.manual_seed(43)
                uninterrupted = create_algorithm(config)

                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = f"{directory}/step_1.d3"
                    boundary_state = None

                    def save_boundary(algo, _epoch, total_step):
                        nonlocal boundary_state
                        if total_step == 1:
                            algo.save(checkpoint)
                            boundary_state = capture_trainer_state(
                                epoch=1, step=total_step
                            )

                    uninterrupted.fit(
                        replay_buffer,
                        n_steps=3,
                        n_steps_per_epoch=1,
                        show_progress=False,
                        logger_adapter=_NoopLoggerFactory(),
                        save_interval=4,
                        epoch_callback=save_boundary,
                    )
                    self.assertIsNotNone(boundary_state)

                    resumed = resume_algorithm(
                        config,
                        replay_buffer,
                        checkpoint,
                        resume_step=1,
                    )
                    self.assertEqual(resumed._grad_step, 1)
                    restore_trainer_state(boundary_state)
                    resumed.fit(
                        replay_buffer,
                        n_steps=2,
                        n_steps_per_epoch=1,
                        show_progress=False,
                        logger_adapter=_NoopLoggerFactory(),
                        save_interval=3,
                    )

                self.assertEqual(uninterrupted._grad_step, 3)
                self.assertEqual(resumed._grad_step, 3)
                _assert_state_equal(
                    self,
                    _saved_state(uninterrupted),
                    _saved_state(resumed),
                )


if __name__ == "__main__":
    unittest.main()
