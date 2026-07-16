from __future__ import annotations

import unittest

import numpy as np

from baselines.data.transitions import (
    MinariTransitionEpisode,
    MinariTransitionPicker,
    build_replay_buffer,
)


class TransitionTest(unittest.TestCase):
    def _episode(self, *, terminated: bool):
        return MinariTransitionEpisode(
            observations=np.arange(18, dtype=np.float32).reshape(3, 6),
            actions=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            rewards=np.array([1.0, 2.0], dtype=np.float32),
            terminated=terminated,
            truncated=not terminated,
            source_variant="test",
        )

    def test_preserves_all_t_transitions_and_final_next_observation(self):
        episode = self._episode(terminated=True)
        picker = MinariTransitionPicker()
        self.assertEqual(episode.transition_count, 2)
        final = picker(episode, 1)
        np.testing.assert_array_equal(final.next_observation, episode.observations[2])
        self.assertEqual(final.terminal, 1.0)

    def test_timeout_bootstraps_from_final_observation(self):
        episode = self._episode(terminated=False)
        final = MinariTransitionPicker()(episode, 1)
        self.assertEqual(final.terminal, 0.0)
        np.testing.assert_array_equal(final.next_observation, episode.observations[2])

    def test_replay_buffer_transition_count(self):
        buffer = build_replay_buffer([self._episode(terminated=True)])
        self.assertEqual(buffer.transition_count, 2)
        self.assertEqual(buffer.dataset_info.action_size, 2)


if __name__ == "__main__":
    unittest.main()
