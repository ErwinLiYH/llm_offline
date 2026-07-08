"""Per-family observation adapters for the CrossMaze wrapper.

The dtype casts here must exactly replicate what each family formatter does
before calling `build_sensing`, so that wrapper-computed sensing state renders
byte-identically to formatter-computed sensing text (float32 rounding at cell
boundaries included).
"""

import numpy as np

SUPPORTED_ENV_FAMILIES = ("pointmaze", "antmaze")


def extract_position_goal(env_family: str, obs) -> tuple[np.ndarray, np.ndarray]:
    """Return the (position, goal) arrays a family formatter senses from."""
    if env_family == "pointmaze":
        return (
            obs["observation"].astype(np.float32),
            obs["desired_goal"].astype(np.float32),
        )
    if env_family == "antmaze":
        return (
            np.asarray(obs["achieved_goal"], dtype=np.float32),
            np.asarray(obs["desired_goal"], dtype=np.float32),
        )
    raise ValueError(
        f"Unsupported env_family for CrossMaze: {env_family!r}. "
        f"Supported: {list(SUPPORTED_ENV_FAMILIES)}"
    )
