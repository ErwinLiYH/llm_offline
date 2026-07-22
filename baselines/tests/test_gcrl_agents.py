from __future__ import annotations

import unittest

try:
    import flax
    import jax
    import jax.numpy as jnp

    from baselines.gcrl.agents import create_agent

    _JAX_AVAILABLE = True
except ModuleNotFoundError:
    _JAX_AVAILABLE = False


@unittest.skipUnless(_JAX_AVAILABLE, "requires the llm_offline_gcrl environment")
class GCRLAgentTest(unittest.TestCase):
    def _config(self, **overrides):
        config = {
            "hidden_dims": (8, 8),
            "activation": "gelu",
            "layer_norm": True,
            "learning_rate": 3e-4,
            "discount": 0.99,
            "latent_dim": 8,
            "alpha": 0.1,
            "const_std": True,
            "tau": 0.005,
            "expectile": 0.7,
            "low_alpha": 3.0,
            "high_alpha": 3.0,
            "rep_dim": 4,
            "low_actor_rep_grad": False,
        }
        config.update(overrides)
        return config

    def _base_batch(self):
        key = jax.random.PRNGKey(4)
        observations = jax.random.normal(key, (8, 3))
        goals = jax.random.normal(jax.random.fold_in(key, 1), (8, 2))
        actions = jnp.tanh(jax.random.normal(jax.random.fold_in(key, 2), (8, 2)))
        return observations, goals, actions

    def test_crl_update_predict_and_checkpoint_roundtrip(self):
        observations, goals, actions = self._base_batch()
        agent = create_agent(
            "crl",
            seed=1,
            example_observations=observations[:2],
            example_goals=goals[:2],
            example_actions=actions[:2],
            config=self._config(),
        )
        batch = {
            "observations": observations,
            "next_observations": observations,
            "actions": actions,
            "value_goals": goals,
            "actor_goals": goals,
            "rewards": jnp.zeros(8),
            "masks": jnp.ones(8),
        }
        updated, info = agent.update(batch)
        self.assertTrue(bool(jnp.isfinite(info["critic/contrastive_loss"])))
        self.assertEqual(updated.predict_actions(observations, goals).shape, (8, 2))
        restored = flax.serialization.from_bytes(
            updated, flax.serialization.to_bytes(updated)
        )
        self.assertEqual(restored.network.step, updated.network.step)

    def test_hiql_update_and_predict(self):
        observations, goals, actions = self._base_batch()
        agent = create_agent(
            "hiql",
            seed=1,
            example_observations=observations[:2],
            example_goals=goals[:2],
            example_actions=actions[:2],
            config=self._config(),
        )
        batch = {
            "observations": observations,
            "next_observations": observations,
            "actions": actions,
            "value_goals": goals,
            "low_actor_goals": goals,
            "high_actor_goals": goals,
            "high_actor_target_states": observations,
            "high_actor_target_goals": goals,
            "rewards": -jnp.ones(8),
            "masks": jnp.ones(8),
        }
        updated, info = agent.update(batch)
        self.assertTrue(bool(jnp.isfinite(info["value/value_loss"])))
        self.assertEqual(updated.predict_actions(observations, goals).shape, (8, 2))


if __name__ == "__main__":
    unittest.main()
