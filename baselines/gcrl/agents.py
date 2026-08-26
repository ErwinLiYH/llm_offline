"""State-based GCRL agents adapted from OGBench's implementations."""

from __future__ import annotations

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from baselines.gcrl.flax_utils import ModuleDict, TrainState, nonpytree_field
from baselines.gcrl.networks import (
    GoalConditionedActor,
    GoalConditionedBilinearValue,
    GoalConditionedEncoder,
    GoalConditionedIQEValue,
    GoalConditionedMRNValue,
    GoalConditionedValue,
    Identity,
    LengthNormalize,
    LogParam,
    MLP,
)


def _safe_exp_mean(values):
    """Return a finite float32 exp mean plus its diagnostic clip fraction."""
    # jnp.mean may reduce by summing before division. Keep the worst-case sum
    # below the dtype maximum while preserving the upstream exp(value) metric
    # everywhere it is representable. This is logging-only and never enters a
    # loss or parameter update.
    max_log = (
        jnp.log(jnp.finfo(values.dtype).max)
        - jnp.log(values.size)
        - 1.0
    )
    clipped = jnp.minimum(values, max_log)
    return jnp.exp(clipped).mean(), jnp.mean(values > max_log)


class CRLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params):
        batch_size = batch["observations"].shape[0]
        values, phi, psi = self.network.select("critic")(
            batch["observations"],
            batch["value_goals"],
            actions=batch["actions"],
            info=True,
            params=grad_params,
        )
        if phi.ndim == 2:
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum("eik,ejk->ije", phi, psi) / jnp.sqrt(phi.shape[-1])
        labels = jnp.eye(batch_size)
        loss = jax.vmap(
            lambda member_logits: optax.sigmoid_binary_cross_entropy(
                logits=member_logits, labels=labels
            ),
            in_axes=-1,
            out_axes=-1,
        )(logits).mean()
        mean_logits = logits.mean(axis=-1)
        positive = (mean_logits * labels).sum() / labels.sum()
        negative = (mean_logits * (1.0 - labels)).sum() / (1.0 - labels).sum()
        v_mean, v_clip_fraction = _safe_exp_mean(values)
        return loss, {
            "critic/contrastive_loss": loss,
            "critic/v_mean": v_mean,
            "critic/v_clip_fraction": v_clip_fraction,
            "critic/binary_accuracy": jnp.mean((mean_logits > 0) == labels),
            "critic/categorical_accuracy": jnp.mean(
                jnp.argmax(mean_logits, axis=1) == jnp.arange(batch_size)
            ),
            "critic/logits_pos": positive,
            "critic/logits_neg": negative,
        }

    def actor_loss(self, batch, grad_params, rng):
        distribution = self.network.select("actor")(
            batch["observations"], batch["actor_goals"], params=grad_params
        )
        actions = (
            distribution.mode()
            if self.config["const_std"]
            else distribution.sample(seed=rng)
        )
        actions = jnp.clip(actions, -1.0, 1.0)
        q1, q2 = self.network.select("critic")(
            batch["observations"], batch["actor_goals"], actions=actions
        )
        q = jnp.minimum(q1, q2)
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = distribution.log_prob(batch["actions"])
        bc_loss = -(self.config["alpha"] * log_prob).mean()
        actor_loss = q_loss + bc_loss
        return actor_loss, {
            "actor/actor_loss": actor_loss,
            "actor/q_loss": q_loss,
            "actor/bc_loss": bc_loss,
            "actor/q_mean": q.mean(),
            "actor/bc_log_prob": log_prob.mean(),
            "actor/mse": jnp.mean((distribution.mode() - batch["actions"]) ** 2),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng):
        critic_loss, info = self.contrastive_loss(batch, grad_params)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng)
        info.update(actor_info)
        return critic_loss + actor_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, loss_rng)

        network, info = self.network.apply_loss_fn(loss_fn)
        return self.replace(rng=new_rng, network=network), info

    @jax.jit
    def predict_actions(self, observations, goals):
        distribution = self.network.select("actor")(observations, goals)
        return jnp.clip(distribution.mode(), -1.0, 1.0)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        example_observations,
        example_goals,
        example_actions,
        config: dict,
    ) -> "CRLAgent":
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        critic = GoalConditionedBilinearValue(
            hidden_dims=tuple(config["hidden_dims"]),
            latent_dim=config["latent_dim"],
            activation=config["activation"],
            layer_norm=config["layer_norm"],
            ensemble=True,
        )
        actor = GoalConditionedActor(
            hidden_dims=tuple(config["hidden_dims"]),
            action_dim=example_actions.shape[-1],
            activation=config["activation"],
            constant_std=config["const_std"],
        )
        model_def = ModuleDict({"critic": critic, "actor": actor})
        model_args = {
            "critic": (
                example_observations,
                example_goals,
                example_actions,
            ),
            "actor": (example_observations, example_goals),
        }
        params = model_def.init(init_rng, **model_args)["params"]
        network = TrainState.create(
            model_def,
            params,
            optax.adam(learning_rate=config["learning_rate"]),
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


class QRLAgent(flax.struct.PyTreeNode):
    """Quasimetric RL with the IQE/MRN and AWR/DDPG+BC upstream variants."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def value_loss(self, batch, grad_params):
        distance_negative = self.network.select("value")(
            batch["observations"],
            batch["value_goals"],
            params=grad_params,
        )
        distance_positive = self.network.select("value")(
            batch["observations"],
            batch["next_observations"],
            params=grad_params,
        )
        dual_lambda = self.network.select("lam")(params=grad_params)
        negative_loss = (
            100.0 * jax.nn.softplus(5.0 - distance_negative / 100.0)
        ).mean()
        positive_loss = (jax.nn.relu(distance_positive - 1.0) ** 2).mean()
        value_loss = negative_loss + positive_loss * jax.lax.stop_gradient(
            dual_lambda
        )
        lambda_loss = dual_lambda * (
            self.config["eps"] - jax.lax.stop_gradient(positive_loss)
        )
        total_loss = value_loss + lambda_loss
        return total_loss, {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "lam_loss": lambda_loss,
            "d_neg_loss": negative_loss,
            "d_neg_mean": distance_negative.mean(),
            "d_neg_max": distance_negative.max(),
            "d_neg_min": distance_negative.min(),
            "d_pos_loss": positive_loss,
            "d_pos_mean": distance_positive.mean(),
            "d_pos_max": distance_positive.max(),
            "d_pos_min": distance_positive.min(),
            "lam": dual_lambda,
        }

    def dynamics_loss(self, batch, grad_params):
        _, observation_representations, next_observation_representations = (
            self.network.select("value")(
                batch["observations"],
                batch["next_observations"],
                info=True,
                params=grad_params,
            )
        )
        predicted_next_representations = (
            observation_representations
            + self.network.select("dynamics")(
                jnp.concatenate(
                    [observation_representations, batch["actions"]],
                    axis=-1,
                ),
                params=grad_params,
            )
        )
        forward_distance = self.network.select("value")(
            next_observation_representations,
            predicted_next_representations,
            is_phi=True,
            params=grad_params,
        )
        reverse_distance = self.network.select("value")(
            predicted_next_representations,
            next_observation_representations,
            is_phi=True,
            params=grad_params,
        )
        loss = (forward_distance + reverse_distance).mean() / 2.0
        return loss, {"dynamics_loss": loss}

    def actor_loss(self, batch, grad_params, rng):
        if self.config["actor_loss"] == "awr":
            value = -self.network.select("value")(
                batch["observations"], batch["actor_goals"]
            )
            next_value = -self.network.select("value")(
                batch["next_observations"], batch["actor_goals"]
            )
            advantage = next_value - value
            weights = jnp.minimum(
                jnp.exp(advantage * self.config["alpha"]),
                100.0,
            )
            distribution = self.network.select("actor")(
                batch["observations"],
                batch["actor_goals"],
                params=grad_params,
            )
            log_prob = distribution.log_prob(batch["actions"])
            actor_loss = -(weights * log_prob).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "adv": advantage.mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean(
                    (distribution.mode() - batch["actions"]) ** 2
                ),
                "std": jnp.mean(distribution.scale_diag),
            }
        if self.config["actor_loss"] != "ddpgbc":
            raise ValueError(
                f"Unsupported QRL actor loss: {self.config['actor_loss']!r}"
            )
        distribution = self.network.select("actor")(
            batch["observations"],
            batch["actor_goals"],
            params=grad_params,
        )
        query_actions = (
            distribution.mode()
            if self.config["const_std"]
            else distribution.sample(seed=rng)
        )
        query_actions = jnp.clip(query_actions, -1.0, 1.0)
        _, observation_representations, goal_representations = (
            self.network.select("value")(
                batch["observations"],
                batch["actor_goals"],
                info=True,
            )
        )
        predicted_next_representations = (
            observation_representations
            + self.network.select("dynamics")(
                jnp.concatenate(
                    [observation_representations, query_actions],
                    axis=-1,
                )
            )
        )
        q_value = -self.network.select("value")(
            predicted_next_representations,
            goal_representations,
            is_phi=True,
        )
        q_loss = -q_value.mean() / jax.lax.stop_gradient(
            jnp.abs(q_value).mean() + 1e-6
        )
        log_prob = distribution.log_prob(batch["actions"])
        bc_loss = -(self.config["alpha"] * log_prob).mean()
        actor_loss = q_loss + bc_loss
        return actor_loss, {
            "actor_loss": actor_loss,
            "q_loss": q_loss,
            "bc_loss": bc_loss,
            "q_mean": q_value.mean(),
            "q_abs_mean": jnp.abs(q_value).mean(),
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((distribution.mode() - batch["actions"]) ** 2),
            "std": jnp.mean(distribution.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        value_loss, value_info = self.value_loss(batch, grad_params)
        info = {f"value/{key}": value for key, value in value_info.items()}
        if self.config["actor_loss"] == "ddpgbc":
            dynamics_loss, dynamics_info = self.dynamics_loss(batch, grad_params)
            info.update(
                {
                    f"dynamics/{key}": value
                    for key, value in dynamics_info.items()
                }
            )
        else:
            dynamics_loss = 0.0
        _, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info.update(
            {f"actor/{key}": value for key, value in actor_info.items()}
        )
        return value_loss + dynamics_loss + actor_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, rng=loss_rng)

        network, info = self.network.apply_loss_fn(loss_fn)
        return self.replace(rng=new_rng, network=network), info

    @jax.jit
    def predict_actions(self, observations, goals):
        distribution = self.network.select("actor")(observations, goals)
        return jnp.clip(distribution.mode(), -1.0, 1.0)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        example_observations,
        example_goals,
        example_actions,
        config: dict,
    ) -> "QRLAgent":
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        value_class = {
            "iqe": GoalConditionedIQEValue,
            "mrn": GoalConditionedMRNValue,
        }[config["quasimetric_type"]]
        value_kwargs = {
            "hidden_dims": tuple(config["hidden_dims"]),
            "latent_dim": config["latent_dim"],
            "activation": config["activation"],
            "layer_norm": config["layer_norm"],
        }
        if config["quasimetric_type"] == "iqe":
            value_kwargs["dim_per_component"] = config[
                "iqe_dim_per_component"
            ]
        value = value_class(**value_kwargs)
        actor = GoalConditionedActor(
            hidden_dims=tuple(config["hidden_dims"]),
            action_dim=example_actions.shape[-1],
            activation=config["activation"],
            constant_std=config["const_std"],
        )
        modules = {"value": value, "actor": actor, "lam": LogParam()}
        model_args = {
            "value": (example_observations, example_goals),
            "actor": (example_observations, example_goals),
            "lam": (),
        }
        if config["actor_loss"] == "ddpgbc":
            dynamics = MLP(
                (*tuple(config["hidden_dims"]), config["latent_dim"]),
                activation=config["activation"],
                activate_final=False,
                layer_norm=config["layer_norm"],
            )
            modules["dynamics"] = dynamics
            example_latents = jnp.zeros(
                (example_observations.shape[0], config["latent_dim"]),
                dtype=example_observations.dtype,
            )
            model_args["dynamics"] = jnp.concatenate(
                [example_latents, example_actions], axis=-1
            )
        model_def = ModuleDict(modules)
        params = model_def.init(init_rng, **model_args)["params"]
        network = TrainState.create(
            model_def,
            params,
            optax.adam(learning_rate=config["learning_rate"]),
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


class GCBCAgent(flax.struct.PyTreeNode):
    """Goal-conditioned behavioral cloning adapted from OGBench."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params):
        distribution = self.network.select("actor")(
            batch["observations"],
            batch["actor_goals"],
            params=grad_params,
        )
        log_prob = distribution.log_prob(batch["actions"])
        actor_loss = -log_prob.mean()
        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_log_prob": log_prob.mean(),
            "mse": jnp.mean((distribution.mode() - batch["actions"]) ** 2),
            "std": jnp.mean(distribution.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        del rng
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        return actor_loss, {
            f"actor/{key}": value for key, value in actor_info.items()
        }

    @jax.jit
    def update(self, batch):
        new_rng, _ = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params)

        network, info = self.network.apply_loss_fn(loss_fn)
        return self.replace(rng=new_rng, network=network), info

    @jax.jit
    def predict_actions(self, observations, goals):
        distribution = self.network.select("actor")(observations, goals)
        return jnp.clip(distribution.mode(), -1.0, 1.0)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        example_observations,
        example_goals,
        example_actions,
        config: dict,
    ) -> "GCBCAgent":
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        actor = GoalConditionedActor(
            hidden_dims=tuple(config["hidden_dims"]),
            action_dim=example_actions.shape[-1],
            activation=config["activation"],
            constant_std=config["const_std"],
        )
        model_def = ModuleDict({"actor": actor})
        params = model_def.init(
            init_rng,
            actor=(example_observations, example_goals),
        )["params"]
        network = TrainState.create(
            model_def,
            params,
            optax.adam(learning_rate=config["learning_rate"]),
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


class HIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(advantage, difference, expectile):
        weight = jnp.where(advantage >= 0, expectile, 1.0 - expectile)
        return weight * difference**2

    def value_loss(self, batch, grad_params):
        next_v1_target, next_v2_target = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_target = jnp.minimum(next_v1_target, next_v2_target)
        q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * next_target
        )
        v1_target, v2_target = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        advantage = q - (v1_target + v2_target) / 2.0
        q1 = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * next_v1_target
        )
        q2 = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * next_v2_target
        )
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        loss = self.expectile_loss(
            advantage, q1 - v1, self.config["expectile"]
        ).mean() + self.expectile_loss(
            advantage, q2 - v2, self.config["expectile"]
        ).mean()
        return loss, {
            "value/value_loss": loss,
            "value/v_mean": ((v1 + v2) / 2.0).mean(),
        }

    def low_actor_loss(self, batch, grad_params):
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["low_actor_goals"]
        )
        next_v1, next_v2 = self.network.select("value")(
            batch["next_observations"], batch["low_actor_goals"]
        )
        advantage = (next_v1 + next_v2 - v1 - v2) / 2.0
        weights = jnp.minimum(
            jnp.exp(advantage * self.config["low_alpha"]), 100.0
        )
        goal_representations = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["low_actor_goals"]], axis=-1
            ),
            params=grad_params,
        )
        if not self.config["low_actor_rep_grad"]:
            goal_representations = jax.lax.stop_gradient(goal_representations)
        distribution = self.network.select("low_actor")(
            batch["observations"],
            goal_representations,
            goal_encoded=True,
            params=grad_params,
        )
        log_prob = distribution.log_prob(batch["actions"])
        loss = -(weights * log_prob).mean()
        return loss, {
            "low_actor/actor_loss": loss,
            "low_actor/adv": advantage.mean(),
            "low_actor/bc_log_prob": log_prob.mean(),
            "low_actor/mse": jnp.mean(
                (distribution.mode() - batch["actions"]) ** 2
            ),
        }

    def high_actor_loss(self, batch, grad_params):
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["high_actor_goals"]
        )
        target_v1, target_v2 = self.network.select("value")(
            batch["high_actor_targets"], batch["high_actor_goals"]
        )
        advantage = (target_v1 + target_v2 - v1 - v2) / 2.0
        weights = jnp.minimum(
            jnp.exp(advantage * self.config["high_alpha"]), 100.0
        )
        distribution = self.network.select("high_actor")(
            batch["observations"],
            batch["high_actor_goals"],
            params=grad_params,
        )
        target_representation = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["high_actor_targets"]],
                axis=-1,
            )
        )
        log_prob = distribution.log_prob(target_representation)
        loss = -(weights * log_prob).mean()
        return loss, {
            "high_actor/actor_loss": loss,
            "high_actor/adv": advantage.mean(),
            "high_actor/bc_log_prob": log_prob.mean(),
            "high_actor/mse": jnp.mean(
                (distribution.mode() - target_representation) ** 2
            ),
        }

    @jax.jit
    def total_loss(self, batch, grad_params):
        value_loss, info = self.value_loss(batch, grad_params)
        low_loss, low_info = self.low_actor_loss(batch, grad_params)
        high_loss, high_info = self.high_actor_loss(batch, grad_params)
        info.update(low_info)
        info.update(high_info)
        return value_loss + low_loss + high_loss, info

    @jax.jit
    def update(self, batch):
        new_rng, _ = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params)

        network, info = self.network.apply_loss_fn(loss_fn)
        target_key = "modules_target_value"
        value_key = "modules_value"
        new_target = jax.tree_util.tree_map(
            lambda current, target: (
                self.config["tau"] * current
                + (1.0 - self.config["tau"]) * target
            ),
            network.params[value_key],
            network.params[target_key],
        )
        params = {**network.params, target_key: new_target}
        network = network.replace(params=params)
        return self.replace(rng=new_rng, network=network), info

    @jax.jit
    def predict_actions(self, observations, goals):
        high_distribution = self.network.select("high_actor")(
            observations, goals
        )
        goal_representations = high_distribution.mode()
        norms = jnp.maximum(
            jnp.linalg.norm(goal_representations, axis=-1, keepdims=True), 1e-6
        )
        goal_representations = (
            goal_representations
            / norms
            * jnp.sqrt(goal_representations.shape[-1])
        )
        low_distribution = self.network.select("low_actor")(
            observations,
            goal_representations,
            goal_encoded=True,
        )
        return jnp.clip(low_distribution.mode(), -1.0, 1.0)

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        example_observations,
        example_goals,
        example_actions,
        config: dict,
    ) -> "HIQLAgent":
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        goal_representation = nn.Sequential(
            [
                MLP(
                    (*tuple(config["hidden_dims"]), config["rep_dim"]),
                    activation=config["activation"],
                    activate_final=False,
                    layer_norm=config["layer_norm"],
                ),
                LengthNormalize(),
            ]
        )
        value_encoder = GoalConditionedEncoder(
            state_encoder=Identity(), concat_encoder=goal_representation
        )
        low_actor_encoder = GoalConditionedEncoder(
            state_encoder=Identity(), concat_encoder=goal_representation
        )
        value = GoalConditionedValue(
            hidden_dims=tuple(config["hidden_dims"]),
            activation=config["activation"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            goal_encoder=value_encoder,
        )
        target_value = GoalConditionedValue(
            hidden_dims=tuple(config["hidden_dims"]),
            activation=config["activation"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            goal_encoder=value_encoder,
        )
        low_actor = GoalConditionedActor(
            hidden_dims=tuple(config["hidden_dims"]),
            action_dim=example_actions.shape[-1],
            activation=config["activation"],
            constant_std=config["const_std"],
            goal_encoder=low_actor_encoder,
        )
        high_actor = GoalConditionedActor(
            hidden_dims=tuple(config["hidden_dims"]),
            action_dim=config["rep_dim"],
            activation=config["activation"],
            constant_std=config["const_std"],
        )
        model_def = ModuleDict(
            {
                "goal_rep": goal_representation,
                "value": value,
                "target_value": target_value,
                "low_actor": low_actor,
                "high_actor": high_actor,
            }
        )
        model_args = {
            "goal_rep": jnp.concatenate(
                [example_observations, example_goals], axis=-1
            ),
            "value": (example_observations, example_goals),
            "target_value": (example_observations, example_goals),
            "low_actor": (example_observations, example_goals),
            "high_actor": (example_observations, example_goals),
        }
        params = model_def.init(init_rng, **model_args)["params"]
        params = {
            **params,
            "modules_target_value": params["modules_value"],
        }
        network = TrainState.create(
            model_def,
            params,
            optax.adam(learning_rate=config["learning_rate"]),
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


def create_agent(
    algorithm: str,
    *,
    seed: int,
    example_observations,
    example_goals,
    example_actions,
    config: dict,
):
    agent_class = {
        "crl": CRLAgent,
        "hiql": HIQLAgent,
        "qrl": QRLAgent,
        "gcbc": GCBCAgent,
    }.get(algorithm)
    if agent_class is None:
        raise ValueError(f"Unsupported GCRL algorithm: {algorithm!r}")
    return agent_class.create(
        seed=seed,
        example_observations=example_observations,
        example_goals=example_goals,
        example_actions=example_actions,
        config=config,
    )
