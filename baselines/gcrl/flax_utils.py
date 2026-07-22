"""Minimal multi-module train state adapted from OGBench's MIT code."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    "ModuleDict initialization arguments must match module names"
                )
            outputs = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    outputs[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    outputs[key] = self.modules[key](*value)
                else:
                    outputs[key] = self.modules[key](value)
            return outputs
        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx):
        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
        )

    def __call__(self, *args, params=None, **kwargs):
        active_params = self.params if params is None else params
        return self.apply_fn({"params": active_params}, *args, **kwargs)

    def select(self, name: str):
        return functools.partial(self, name=name)

    def apply_gradients(self, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        return self.replace(
            step=self.step + 1,
            params=optax.apply_updates(self.params, updates),
            opt_state=new_opt_state,
        )

    def apply_loss_fn(self, loss_fn):
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        leaves = jax.tree_util.tree_leaves(grads)
        info.update(
            {
                "grad/max": jnp.max(
                    jnp.stack([jnp.max(value) for value in leaves])
                ),
                "grad/min": jnp.min(
                    jnp.stack([jnp.min(value) for value in leaves])
                ),
                "grad/norm": sum(jnp.linalg.norm(value) for value in leaves),
            }
        )
        return self.apply_gradients(grads), info
