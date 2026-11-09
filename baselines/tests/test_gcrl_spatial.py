from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from baselines.data.spatial import spatial_layout
    from baselines.gcrl.agents import create_agent
    from baselines.gcrl.runner import _agent_config, _save_agent, load_gcrl_checkpoint
    from baselines.gcrl.spatial import SpatialEncoder, map_channels
    from baselines.tests.spatial_fixture import spatial_config, spatial_values, spatial_run_config
    _JAX_AVAILABLE = True
except ModuleNotFoundError:
    _JAX_AVAILABLE = False


@unittest.skipUnless(_JAX_AVAILABLE, "requires the llm_offline_gcrl environment")
class CRLSpatialTest(unittest.TestCase):
    def test_crl_update_gradient_paths_predict_and_checkpoint(self):
        for family in ("pointmaze", "antmaze"):
            for architecture in ("cnn_map", "transformer_map"):
                with self.subTest(family=family, architecture=architecture):
                    config = spatial_config("crl", architecture, family)
                    agent_config = _agent_config(config)
                    agent_config["latent_dim"] = 8
                    layout = spatial_layout(family, config["observation"], goal_conditioned=True)
                    observations = jnp.asarray(spatial_values(layout, 4))
                    goals = observations[::-1]
                    actions = jnp.ones((4, 2 if family == "pointmaze" else 8)) * 0.1
                    channels, valid = map_channels(observations, layout)
                    self.assertFalse(np.asarray(valid[:, -1]).any())
                    self.assertEqual(float(channels[0, 2, 2, 2]), 1)
                    self.assertEqual(float(channels[0, 2, 2, 3]), 1)
                    agent = create_agent("crl", seed=1, example_observations=observations[:2],
                                         example_goals=goals[:2], example_actions=actions[:2], config=agent_config)
                    batch = dict(observations=observations, next_observations=observations,
                                 value_goals=goals, actor_goals=goals, actions=actions,
                                 rewards=jnp.zeros(4), masks=jnp.ones(4))
                    updated, info = agent.update(batch)
                    self.assertTrue(all(np.isfinite(np.asarray(x)).all() for x in info.values()))
                    # Verify that both critic branches and actor encoders actually train.
                    before = jax.tree_util.tree_leaves(agent.network.params)
                    after = jax.tree_util.tree_leaves(updated.network.params)
                    self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(before, after)))
                    for module in ("critic", "actor"):
                        old = agent.network.params[f"modules_{module}"]
                        new = updated.network.params[f"modules_{module}"]
                        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
                            jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new))))
                    for branch in ("phi", "psi"):
                        old = agent.network.params["modules_critic"][branch]
                        new = updated.network.params["modules_critic"][branch]
                        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
                            jax.tree_util.tree_leaves(old), jax.tree_util.tree_leaves(new))))
                    expected = updated.predict_actions(observations, goals)
                    self.assertEqual(expected.shape, actions.shape)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "model.msgpack"
                        metadata = {"gcrl_goal_semantics": "full_observation_v1", "network_config": config["network"]}
                        _save_agent(path, updated, metadata)
                        restored = load_gcrl_checkpoint(path, agent, expected=metadata)
                        np.testing.assert_allclose(restored.predict_actions(observations, goals), expected, atol=1e-6)
                    jax.clear_caches()

    def test_runner_with_corpus_and_live_rollout(self):
        import json
        from baselines.gcrl.data import GCRLNormalizer
        from baselines.gcrl.runner import train_gcrl_baseline
        from baselines.data.spatial import map_indices

        for architecture in ("cnn_map", "transformer_map"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                config = spatial_run_config(directory, "crl", architecture)
                run_dir = train_gcrl_baseline(config)
                summary = json.loads((run_dir / "summary.json").read_text())
                self.assertTrue(summary["evaluation_history"])
                normalizer = GCRLNormalizer.from_dict(json.loads((run_dir / "normalizer.json").read_text()))
                layout = spatial_layout("pointmaze", config["observation"], goal_conditioned=True)
                values = spatial_values(layout)
                indices = map_indices(layout)
                np.testing.assert_array_equal(normalizer.normalize_states(values)[:, indices], values[:, indices])
                np.testing.assert_array_equal(normalizer.normalize_goals(values)[:, indices], values[:, indices])
                self.assertTrue((run_dir / "model_metadata.json").exists())
                jax.clear_caches()

    def test_encoder_uses_map_and_numeric_features(self):
        config = spatial_config("crl", "transformer_map")
        layout = spatial_layout("pointmaze", config["observation"], goal_conditioned=True)
        values = jnp.asarray(spatial_values(layout, 2))
        encoder = SpatialEncoder(layout, "transformer_map", config["network"]["spatial"])
        params = encoder.init(jax.random.PRNGKey(1), values)
        expected = encoder.apply(params, values)
        changed_numeric = values.at[:, 0].add(5)
        changed_map = values.at[:, layout["maps"][1]["start"] + 3 * layout["shape"][1] + 3].set(1)
        self.assertFalse(np.allclose(encoder.apply(params, changed_numeric), expected))
        self.assertFalse(np.allclose(encoder.apply(params, changed_map), expected))


if __name__ == "__main__":
    unittest.main()
