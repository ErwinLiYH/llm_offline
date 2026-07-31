from __future__ import annotations

import io

from d3rlpy.algos import (
    BC,
    BCConfig,
    IQLConfig,
    ReBRACConfig,
    TD3PlusBCConfig,
)
import d3rlpy
import torch
from d3rlpy.models import VectorEncoderFactory
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.preprocessing import (
    ClipRewardScaler,
    ConstantShiftRewardScaler,
    MultiplyRewardScaler,
    StandardObservationScaler,
)

from baselines.algorithms.scaling_mlp import ScalingMLPEncoderFactory


class InstrumentedBC(BC):
    """Deterministic BC with opt-in, pre-step gradient-norm sampling.

    d3rlpy exposes epoch loss but not the gradient norm used to obtain it.
    The scaling protocol samples the global policy-gradient norm before a
    possible rescue clip.  It changes neither the loss nor the optimizer when
    ``max_grad_norm`` is ``None``.
    """

    def __init__(
        self,
        config,
        device,
        enable_ddp,
        *,
        gradient_sample_interval_updates: int,
        max_grad_norm: float | None = None,
        initial_update_count: int = 0,
    ):
        super().__init__(config, device, enable_ddp)
        self.global_grad_norm_clip = (
            None if max_grad_norm is None else float(max_grad_norm)
        )
        self._gradient_sample_interval_updates = int(
            gradient_sample_interval_updates
        )
        self._update_count = int(initial_update_count)
        self._gradient_norm_samples: list[float] = []

    def inner_create_impl(self, observation_shape, action_size) -> None:
        super().inner_create_impl(observation_shape, action_size)
        assert self.impl is not None
        implementation = self.impl
        parameters = tuple(implementation.policy.parameters())

        def compute_imitator_grad(batch):
            implementation._modules.optim.zero_grad()
            loss = implementation.compute_loss(batch.observations, batch.actions)
            loss.loss.backward()
            self._update_count += 1
            if self._update_count % self._gradient_sample_interval_updates == 0:
                component_norms = [
                    parameter.grad.detach().norm(2)
                    for parameter in parameters
                    if parameter.grad is not None
                ]
                if not component_norms:
                    raise RuntimeError("BC policy has no gradients to diagnose")
                global_norm = torch.stack(component_norms).norm(2)
                self._gradient_norm_samples.append(float(global_norm.item()))
            if self.global_grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.global_grad_norm_clip
                )
            return loss

        # ``compile_graph`` is rejected by config validation for instrumented
        # runs, so replacing this callable cannot invalidate a compiled graph.
        implementation._compute_imitator_grad = compute_imitator_grad

    def consume_training_diagnostics(self) -> dict[str, float | int | None]:
        """Return and reset pre-step gradient samples since the last epoch."""

        samples = self._gradient_norm_samples
        self._gradient_norm_samples = []
        if not samples:
            return {
                "gradient_global_norm_sample_count": 0,
                "gradient_global_norm_mean": None,
                "gradient_global_norm_max": None,
                "gradient_global_norm_last": None,
            }
        return {
            "gradient_global_norm_sample_count": len(samples),
            "gradient_global_norm_mean": float(sum(samples) / len(samples)),
            "gradient_global_norm_max": float(max(samples)),
            "gradient_global_norm_last": float(samples[-1]),
        }


class GradientClippedBC(InstrumentedBC):
    """Instrumented BC with the explicitly labelled gradient-clip rescue."""

    def __init__(
        self,
        config,
        device,
        enable_ddp,
        *,
        max_grad_norm: float,
        gradient_sample_interval_updates: int = 1000,
        initial_update_count: int = 0,
    ):
        super().__init__(
            config,
            device,
            enable_ddp,
            gradient_sample_interval_updates=gradient_sample_interval_updates,
            max_grad_norm=max_grad_norm,
            initial_update_count=initial_update_count,
        )


def _encoder(network: dict) -> EncoderFactory:
    if network["architecture"] != "legacy_mlp":
        return ScalingMLPEncoderFactory(
            body_type=network["architecture"],
            width=network["width"],
            body_depth=network["body_depth"],
            block_size=network["block_size"],
        )
    return VectorEncoderFactory(
        hidden_units=list(network["hidden_units"]),
        activation=network["activation"],
        use_batch_norm=network["use_batch_norm"],
        use_layer_norm=network["use_layer_norm"],
        dropout_rate=network["dropout_rate"],
    )


def _reward_scaler(config: dict | None):
    if config is None:
        return None
    scaler_type = config["type"]
    kwargs = {key: value for key, value in config.items() if key != "type"}
    if scaler_type == "multiply":
        return MultiplyRewardScaler(**kwargs)
    if scaler_type == "constant_shift":
        return ConstantShiftRewardScaler(**kwargs)
    if scaler_type == "clip":
        return ClipRewardScaler(**kwargs)
    raise ValueError(f"Unsupported reward scaler: {scaler_type!r}")


def create_algorithm(config: dict):
    algorithm = config["algorithm"]
    network = config["network"]
    kwargs = dict(config["algorithm_config"])
    reward_scaler_config = kwargs.pop("reward_scaler", None)
    global_grad_norm_clip = kwargs.pop("global_grad_norm_clip", None)
    training_diagnostics = config["training_diagnostics"]
    common = {
        "observation_scaler": StandardObservationScaler(),
        "reward_scaler": _reward_scaler(reward_scaler_config),
    }
    if algorithm == "mlp_bc":
        algo_config = BCConfig(
            encoder_factory=_encoder(network),
            **common,
            **kwargs,
        )
        if global_grad_norm_clip is not None:
            return GradientClippedBC(
                algo_config,
                config["device"],
                False,
                max_grad_norm=global_grad_norm_clip,
                gradient_sample_interval_updates=training_diagnostics[
                    "gradient_sample_interval_updates"
                ],
            )
        if training_diagnostics["enabled"]:
            return InstrumentedBC(
                algo_config,
                config["device"],
                False,
                gradient_sample_interval_updates=training_diagnostics[
                    "gradient_sample_interval_updates"
                ],
            )
    elif algorithm == "td3_bc":
        algo_config = TD3PlusBCConfig(
            actor_encoder_factory=_encoder(network),
            critic_encoder_factory=_encoder(network),
            **common,
            **kwargs,
        )
    elif algorithm == "iql":
        algo_config = IQLConfig(
            actor_encoder_factory=_encoder(network),
            critic_encoder_factory=_encoder(network),
            value_encoder_factory=_encoder(network),
            **common,
            **kwargs,
        )
    elif algorithm == "rebrac":
        algo_config = ReBRACConfig(
            actor_encoder_factory=_encoder(network),
            critic_encoder_factory=_encoder(network),
            **common,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported baseline algorithm: {algorithm!r}")
    return algo_config.create(device=config["device"])


def load_algorithm(path: str, *, device):
    """Load a d3rlpy baseline after registering repository encoder factories."""

    return d3rlpy.load_learnable(path, device=device)


def resume_algorithm(config: dict, dataset, checkpoint: str, *, resume_step: int):
    """Restore a d3rlpy algorithm and its absolute optimizer update count.

    ``load_learnable`` restores modules, targets, optimizers, schedulers, and
    fitted scalers, but constructs the outer algorithm with ``_grad_step=0``.
    Restoring that counter is required by delayed-update algorithms such as
    TD3+BC and ReBRAC.  Instrumented BC additionally needs its repository
    wrapper reconstructed around the loaded d3rlpy implementation.
    """

    loaded = load_algorithm(checkpoint, device=config["device"])
    if config["algorithm"] != "mlp_bc":
        loaded._grad_step = int(resume_step)
        return loaded

    training_diagnostics = config["training_diagnostics"]
    global_grad_norm_clip = config["algorithm_config"].get(
        "global_grad_norm_clip"
    )
    if not training_diagnostics["enabled"] and global_grad_norm_clip is None:
        loaded._grad_step = int(resume_step)
        return loaded

    loaded_config = loaded.config
    if global_grad_norm_clip is not None:
        resumed = GradientClippedBC(
            loaded_config,
            config["device"],
            False,
            max_grad_norm=global_grad_norm_clip,
            gradient_sample_interval_updates=training_diagnostics[
                "gradient_sample_interval_updates"
            ],
            initial_update_count=resume_step,
        )
    else:
        resumed = InstrumentedBC(
            loaded_config,
            config["device"],
            False,
            gradient_sample_interval_updates=training_diagnostics[
                "gradient_sample_interval_updates"
            ],
            initial_update_count=resume_step,
        )
    resumed.build_with_dataset(dataset)
    assert loaded.impl is not None
    assert resumed.impl is not None
    buffer = io.BytesIO()
    loaded.impl.save_model(buffer)
    buffer.seek(0)
    resumed.impl.load_model(buffer)
    resumed._grad_step = int(resume_step)
    return resumed
