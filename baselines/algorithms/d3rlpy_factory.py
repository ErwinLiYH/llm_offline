from __future__ import annotations

from d3rlpy.algos import BCConfig, IQLConfig, TD3PlusBCConfig
from d3rlpy.models import VectorEncoderFactory
from d3rlpy.preprocessing import (
    ClipRewardScaler,
    ConstantShiftRewardScaler,
    MultiplyRewardScaler,
    StandardObservationScaler,
)


def _encoder(network: dict) -> VectorEncoderFactory:
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
    else:
        raise ValueError(f"Unsupported baseline algorithm: {algorithm!r}")
    return algo_config.create(device=config["device"])
