from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

try:
    import flax
    import jax
    import jax.numpy as jnp
    import numpy as np

    from baselines.config import normalize_baseline_config
    from baselines.evaluation import _make_evaluation_env
    from baselines.gcrl.agents import create_agent
    from baselines.gcrl.runner import _save_agent, load_gcrl_checkpoint

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
        # PointMaze GCRL base 4D + one 15x15 dynamic-map slot.
        observations = jax.random.normal(key, (8, 229))
        goals = jax.random.normal(jax.random.fold_in(key, 1), (8, 229))
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.msgpack"
            metadata = {
                "version": 1,
                "gcrl_goal_semantics": "full_observation_v1",
                "observation_dimension": 229,
            }
            _save_agent(path, updated, metadata)
            restored = load_gcrl_checkpoint(
                path,
                updated,
                expected={"observation_dimension": 229},
            )
            self.assertEqual(restored.network.step, updated.network.step)

            path.with_suffix(".msgpack.metadata.json").unlink()
            with self.assertRaisesRegex(ValueError, "compact_xy"):
                load_gcrl_checkpoint(path, updated)

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
            "high_actor_targets": observations,
            "rewards": -jnp.ones(8),
            "masks": jnp.ones(8),
        }
        updated, info = agent.update(batch)
        self.assertTrue(bool(jnp.isfinite(info["value/value_loss"])))
        self.assertEqual(updated.predict_actions(observations, goals).shape, (8, 2))

    def test_point_ant_crl_hiql_dynamic_map_update_and_rollout(self):
        for env_family, action_dim in (("pointmaze", 2), ("antmaze", 8)):
            baseline_config = normalize_baseline_config(
                {
                    "algorithm": "crl",
                    "env_family": env_family,
                    "train_variants": ["umaze"],
                    "device": "cpu",
                    "observation": {"include_dynamic_map": True},
                }
            )
            env = _make_evaluation_env(
                env_family=env_family,
                variant="umaze",
                reward_type="sparse",
                env_config=baseline_config["evaluation"]["env_config"],
                observation_config=baseline_config["observation"],
                goal_conditioned=True,
            )
            try:
                rollout_observation, _ = env.reset(seed=23)
                state = jnp.asarray(rollout_observation["state"])
                goal = jnp.asarray(rollout_observation["goal"])
                observations = jnp.stack(
                    [state + index * 1e-3 for index in range(4)]
                )
                goals = jnp.stack([goal + index * 1e-3 for index in range(4)])
                actions = jnp.zeros((4, action_dim), dtype=jnp.float32)
                for algorithm in ("crl", "hiql"):
                    with self.subTest(env_family=env_family, algorithm=algorithm):
                        agent = create_agent(
                            algorithm,
                            seed=2,
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
                            "rewards": (
                                jnp.zeros(4)
                                if algorithm == "crl"
                                else -jnp.ones(4)
                            ),
                            "masks": jnp.ones(4),
                        }
                        if algorithm == "crl":
                            batch["actor_goals"] = goals
                        else:
                            batch.update(
                                {
                                    "low_actor_goals": goals,
                                    "high_actor_goals": goals,
                                    "high_actor_targets": observations,
                                }
                            )
                        updated, _ = agent.update(batch)
                        predicted = np.asarray(
                            updated.predict_actions(state[None, :], goal[None, :])
                        )
                        self.assertEqual(predicted.shape, (1, action_dim))
                        next_observation, *_ = env.step(predicted[0])
                        self.assertEqual(
                            next_observation["state"].shape,
                            rollout_observation["state"].shape,
                        )
                        if algorithm == "crl":
                            rollout_observation, _ = env.reset(seed=23)
                            state = jnp.asarray(rollout_observation["state"])
                            goal = jnp.asarray(rollout_observation["goal"])
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
