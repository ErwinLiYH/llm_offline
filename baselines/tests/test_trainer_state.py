from __future__ import annotations

import random
import tempfile
import unittest

import numpy as np
import torch

from baselines.trainer_state import (
    load_trainer_state,
    restore_trainer_state,
    save_trainer_state,
)


class TrainerStateTest(unittest.TestCase):
    def test_rng_sequences_roundtrip(self) -> None:
        random.seed(17)
        np.random.seed(18)
        torch.manual_seed(19)

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/trainer_state.pt"
            save_trainer_state(path, epoch=4, step=100_000)
            expected_python = [random.random() for _ in range(4)]
            expected_numpy = np.random.random(4)
            expected_torch = torch.rand(4)

            random.seed(117)
            np.random.seed(118)
            torch.manual_seed(119)
            state = load_trainer_state(path)
            restore_trainer_state(state)

        self.assertEqual(state["epoch"], 4)
        self.assertEqual(state["step"], 100_000)
        self.assertEqual([random.random() for _ in range(4)], expected_python)
        np.testing.assert_array_equal(np.random.random(4), expected_numpy)
        torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)

    def test_invalid_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "version"):
            restore_trainer_state({"version": 99})


if __name__ == "__main__":
    unittest.main()
