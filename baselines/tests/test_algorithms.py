from __future__ import annotations

import unittest

import numpy as np

from baselines.algorithms import create_algorithm
from baselines.config import normalize_baseline_config
from baselines.data.transitions import MinariTransitionEpisode, build_replay_buffer
from baselines.evaluation import evaluate_validation


class AlgorithmSmokeTest(unittest.TestCase):
    def _buffer(self):
        rng = np.random.default_rng(3)
        episodes = []
        for _ in range(3):
            episodes.append(
                MinariTransitionEpisode(
                    observations=rng.normal(size=(9, 6)).astype(np.float32),
                    actions=np.tanh(rng.normal(size=(8, 2))).astype(np.float32),
                    rewards=rng.normal(size=8).astype(np.float32),
                    terminated=True,
                    truncated=False,
                    source_variant="synthetic",
                )
            )
        return build_replay_buffer(episodes)

    def test_each_authoritative_algorithm_updates_and_predicts(self):
        buffer = self._buffer()
        for algorithm in ("mlp_bc", "td3_bc", "iql"):
            with self.subTest(algorithm=algorithm):
                config = normalize_baseline_config(
                    {
                        "algorithm": algorithm,
                        "train_variants": ["umaze"],
                        "device": False,
                        "n_steps": 1,
                        "n_steps_per_epoch": 1,
                        "show_progress": False,
                        "network": {"hidden_units": [16, 16]},
                        "algorithm_config": {"batch_size": 4},
                    }
                )
                algo = create_algorithm(config)
                history = algo.fit(
                    buffer,
                    n_steps=1,
                    n_steps_per_epoch=1,
                    show_progress=False,
                    logger_adapter=_NoopLoggerFactory(),
                    save_interval=2,
                )
                self.assertEqual(len(history), 1)
                action = algo.predict(np.zeros((1, 6), dtype=np.float32))
                self.assertEqual(action.shape, (1, 2))
                validation = evaluate_validation(
                    algo, buffer, algorithm=algorithm
                )
                self.assertTrue(np.isfinite(validation["action_mse_sum"]))
                if algorithm in {"td3_bc", "iql"}:
                    self.assertTrue(np.isfinite(validation["td_error"]))


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
