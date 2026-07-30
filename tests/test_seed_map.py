import unittest

from crossmaze import make_seed_map
from crossmaze.seed_map import (
    SeedMapSpec,
    generate_seed_map,
    normalize_seed_map_spec,
    seed_map_hash,
    seed_map_shape,
)


class SeedMapTests(unittest.TestCase):
    def test_random_shape_and_map_are_deterministic(self):
        spec = SeedMapSpec(size_mode="random", min_size=5, max_size=15)
        first = generate_seed_map(123, spec)
        second = generate_seed_map(123, spec)
        self.assertEqual(first, second)
        self.assertEqual(seed_map_hash(first), seed_map_hash(second))
        self.assertEqual((len(first), len(first[0])), seed_map_shape(123, spec))
        self.assertIn(len(first), {5, 7, 9, 11, 13, 15})

    def test_fixed_rectangular_shape(self):
        spec = normalize_seed_map_spec(
            {
                "size_mode": "fixed",
                "fixed_rows": 9,
                "fixed_cols": 13,
            }
        )
        maze_map = generate_seed_map(7, spec)
        self.assertEqual((len(maze_map), len(maze_map[0])), (9, 13))
        self.assertTrue(all(cell == 1 for cell in maze_map[0]))
        self.assertTrue(all(cell == 1 for cell in maze_map[-1]))
        self.assertTrue(all(row[0] == 1 and row[-1] == 1 for row in maze_map))

    def test_different_seeds_produce_multiple_layouts(self):
        spec = SeedMapSpec(size_mode="fixed", fixed_rows=11, fixed_cols=11)
        hashes = {seed_map_hash(generate_seed_map(seed, spec)) for seed in range(8)}
        self.assertGreater(len(hashes), 1)

    def test_even_sizes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be odd"):
            normalize_seed_map_spec(
                {"size_mode": "random", "min_size": 6, "max_size": 15}
            )

    def test_random_mode_rejects_fixed_fields(self):
        with self.assertRaisesRegex(ValueError, "only valid"):
            normalize_seed_map_spec(
                {
                    "size_mode": "random",
                    "min_size": 5,
                    "max_size": 15,
                    "fixed_rows": 9,
                }
            )

    def test_pointmaze_and_antmaze_envs_share_generated_layout(self):
        spec = {
            "size_mode": "fixed",
            "fixed_rows": 5,
            "fixed_cols": 5,
        }
        expected_map = generate_seed_map(3, spec)
        for env_family, action_dim in (("pointmaze", 2), ("antmaze", 8)):
            with self.subTest(env_family=env_family):
                env = make_seed_map(
                    env_family,
                    3,
                    seed_map_spec=spec,
                    config={
                        "reward_type": "sparse",
                        "max_episode_steps": 5,
                    },
                )
                try:
                    obs, _ = env.reset(seed=11)
                    self.assertEqual(env.action_space.shape, (action_dim,))
                    self.assertEqual(
                        obs["crossmaze"]["maze_map"],
                        expected_map,
                    )
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
