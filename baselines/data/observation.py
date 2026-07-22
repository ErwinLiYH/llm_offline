from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import gymnasium as gym
import numpy as np

from crossmaze import (
    CROSSMAZE_OBS_KEY,
    compute_sensing_arrays,
    get_env_facts,
    list_variants,
)


BASE_OBSERVATION_DIMS = {
    "pointmaze": 6,
    "antmaze": 31,
}

GOAL_CONDITIONED_STATE_DIMS = {
    "pointmaze": 4,
    "antmaze": 29,
}
GOAL_CONDITIONED_GOAL_DIMS = {
    "pointmaze": 2,
    "antmaze": 2,
}

# Kept as a compatibility alias for code that only needs the legacy dimensions.
OBSERVATION_DIMS = BASE_OBSERVATION_DIMS
MAP_PADDING_VALUE = -1.0
LOCATION_SENSING_DIM = 4
WALL_SENSING_DIM = 4


def _enabled(observation_config: Mapping | None, key: str) -> bool:
    return bool((observation_config or {}).get(key, False))


def uses_structured_observation(observation_config: Mapping | None) -> bool:
    return any(
        _enabled(observation_config, key)
        for key in (
            "include_map",
            "include_location_sensing",
            "include_wall_sensing",
        )
    )


@lru_cache(maxsize=None)
def family_map_shape(env_family: str) -> tuple[int, int]:
    """Return the fixed map slot used by every variant in one family."""
    if env_family not in BASE_OBSERVATION_DIMS:
        raise ValueError(f"Unsupported env_family: {env_family!r}")
    shapes = []
    for variant in list_variants(env_family):
        maze_map = get_env_facts(env_family, variant)["maze_map"]
        shapes.append((len(maze_map), len(maze_map[0])))
    return max(rows for rows, _cols in shapes), max(cols for _rows, cols in shapes)


def observation_dim(env_family: str, observation_config: Mapping | None = None) -> int:
    if env_family not in BASE_OBSERVATION_DIMS:
        raise ValueError(f"Unsupported env_family: {env_family!r}")
    dimension = BASE_OBSERVATION_DIMS[env_family]
    if _enabled(observation_config, "include_map"):
        rows, cols = family_map_shape(env_family)
        dimension += rows * cols
    if _enabled(observation_config, "include_location_sensing"):
        dimension += LOCATION_SENSING_DIM
    if _enabled(observation_config, "include_wall_sensing"):
        dimension += WALL_SENSING_DIM
    return dimension


def observation_schema(env_family: str, observation_config: Mapping | None = None) -> dict:
    base_name = (
        "observation+desired_goal"
        if env_family == "pointmaze"
        else "achieved_goal+observation+desired_goal"
    )
    components = [{"name": "base", "dimension": BASE_OBSERVATION_DIMS[env_family]}]
    if _enabled(observation_config, "include_map"):
        rows, cols = family_map_shape(env_family)
        components.append(
            {
                "name": "map",
                "dimension": rows * cols,
                "shape": [rows, cols],
                "flatten_order": "row-major",
                "padding_value": MAP_PADDING_VALUE,
            }
        )
    if _enabled(observation_config, "include_location_sensing"):
        components.append(
            {
                "name": "location_sensing",
                "dimension": LOCATION_SENSING_DIM,
                "order": ["position_row", "position_col", "goal_row", "goal_col"],
                "index_base": 0,
            }
        )
    if _enabled(observation_config, "include_wall_sensing"):
        components.append(
            {
                "name": "wall_sensing",
                "dimension": WALL_SENSING_DIM,
                "order": ["up", "down", "left", "right"],
                "status_codes": {"free": 0, "wall": 1, "risk": 2},
            }
        )
    return {
        "base": base_name,
        "dimension": observation_dim(env_family, observation_config),
        "components": components,
    }


def goal_conditioned_observation_dims(
    env_family: str,
    observation_config: Mapping | None = None,
) -> tuple[int, int]:
    """Return separate state and goal dimensions for CRL/HIQL.

    Unlike the legacy d3rlpy input, the desired goal is not duplicated inside
    the state vector. Map and wall features describe the state; the goal cell
    belongs to the goal vector.
    """
    if env_family not in GOAL_CONDITIONED_STATE_DIMS:
        raise ValueError(f"Unsupported env_family: {env_family!r}")
    state_dim = GOAL_CONDITIONED_STATE_DIMS[env_family]
    goal_dim = GOAL_CONDITIONED_GOAL_DIMS[env_family]
    if _enabled(observation_config, "include_map"):
        rows, cols = family_map_shape(env_family)
        state_dim += rows * cols
    if _enabled(observation_config, "include_location_sensing"):
        state_dim += 2
        goal_dim += 2
    if _enabled(observation_config, "include_wall_sensing"):
        state_dim += WALL_SENSING_DIM
    return state_dim, goal_dim


def goal_conditioned_observation_schema(
    env_family: str,
    observation_config: Mapping | None = None,
) -> dict:
    state_dim, goal_dim = goal_conditioned_observation_dims(
        env_family, observation_config
    )
    state_components = [
        {
            "name": "state",
            "dimension": GOAL_CONDITIONED_STATE_DIMS[env_family],
            "source": (
                "observation"
                if env_family == "pointmaze"
                else "achieved_goal+observation"
            ),
        }
    ]
    goal_components = [
        {
            "name": "goal_xy",
            "dimension": GOAL_CONDITIONED_GOAL_DIMS[env_family],
        }
    ]
    if _enabled(observation_config, "include_map"):
        rows, cols = family_map_shape(env_family)
        state_components.append(
            {
                "name": "map",
                "dimension": rows * cols,
                "shape": [rows, cols],
                "flatten_order": "row-major",
                "padding_value": MAP_PADDING_VALUE,
            }
        )
    if _enabled(observation_config, "include_location_sensing"):
        state_components.append(
            {
                "name": "position_cell",
                "dimension": 2,
                "order": ["row", "col"],
                "index_base": 0,
            }
        )
        goal_components.append(
            {
                "name": "goal_cell",
                "dimension": 2,
                "order": ["row", "col"],
                "index_base": 0,
            }
        )
    if _enabled(observation_config, "include_wall_sensing"):
        state_components.append(
            {
                "name": "wall_sensing",
                "dimension": WALL_SENSING_DIM,
                "order": ["up", "down", "left", "right"],
                "status_codes": {"free": 0, "wall": 1, "risk": 2},
            }
        )
    return {
        "state_dimension": state_dim,
        "goal_dimension": goal_dim,
        "state_components": state_components,
        "goal_components": goal_components,
    }


def _state_position_and_goal(
    observation: Mapping,
    env_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if env_family == "pointmaze":
        state = np.asarray(observation["observation"], dtype=np.float32)
        position = state[..., :2]
    elif env_family == "antmaze":
        achieved_goal = np.asarray(observation["achieved_goal"], dtype=np.float32)
        proprioception = np.asarray(observation["observation"], dtype=np.float32)
        state = np.concatenate([achieved_goal, proprioception], axis=-1)
        position = achieved_goal[..., :2]
    else:
        raise ValueError(f"Unsupported env_family: {env_family!r}")
    goal = np.asarray(observation["desired_goal"], dtype=np.float32)[..., :2]
    return state, position, goal


def _base_vector_and_position_goal(
    observation: Mapping,
    env_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, position, goal = _state_position_and_goal(observation, env_family)
    return np.concatenate([state, goal], axis=-1), position, goal


def _layout_meta(
    observation: Mapping,
    *,
    env_family: str,
    variant: str | None,
    observation_config: Mapping,
) -> dict:
    attached = observation.get(CROSSMAZE_OBS_KEY)
    if attached is not None:
        if not isinstance(attached, Mapping):
            raise ValueError(f"observation[{CROSSMAZE_OBS_KEY!r}] must be a mapping")
        maze_map = attached.get("maze_map")
        if maze_map is None:
            raise ValueError(
                f"observation[{CROSSMAZE_OBS_KEY!r}] is missing maze_map"
            )
        maze_size_scaling = attached.get("maze_size_scaling", 1.0)
    else:
        if variant is None:
            raise ValueError(
                "variant is required to derive map/sensing features from offline observations"
            )
        facts = get_env_facts(env_family, variant)
        maze_map = facts["maze_map"]
        maze_size_scaling = facts["maze_size_scaling"]
    return {
        "maze_map": [list(row) for row in maze_map],
        "maze_size_scaling": float(maze_size_scaling),
        "wall_sensing_version": observation_config["wall_sensing_version"],
        "map_sensing_boundary_risk_threshold": observation_config[
            "map_sensing_boundary_risk_threshold"
        ],
    }


def _map_features(
    maze_map: list[list[object]],
    *,
    env_family: str,
    leading_shape: tuple[int, ...],
) -> np.ndarray:
    target_rows, target_cols = family_map_shape(env_family)
    rows = len(maze_map)
    cols = len(maze_map[0]) if maze_map else 0
    if rows < 1 or cols < 1 or any(len(row) != cols for row in maze_map):
        raise ValueError("maze_map must be a non-empty rectangular matrix")
    if rows > target_rows or cols > target_cols:
        raise ValueError(
            f"{env_family} map shape {(rows, cols)} exceeds fixed observation map "
            f"shape {(target_rows, target_cols)}"
        )
    padded = np.full(
        (target_rows, target_cols),
        MAP_PADDING_VALUE,
        dtype=np.float32,
    )
    padded[:rows, :cols] = np.asarray(
        [[1.0 if cell == 1 else 0.0 for cell in row] for row in maze_map],
        dtype=np.float32,
    )
    flattened = padded.reshape(-1)
    return np.broadcast_to(flattened, leading_shape + flattened.shape)


def vectorize_observation(
    observation,
    env_family: str,
    *,
    observation_config: Mapping | None = None,
    variant: str | None = None,
) -> np.ndarray:
    if not isinstance(observation, Mapping):
        raise ValueError("CrossMaze baseline observation must be a mapping")
    vector, position, goal = _base_vector_and_position_goal(observation, env_family)
    expected_base_dim = BASE_OBSERVATION_DIMS[env_family]
    if vector.shape[-1] != expected_base_dim:
        raise ValueError(
            f"Unexpected {env_family} baseline observation dimension: "
            f"expected {expected_base_dim}, got {vector.shape[-1]}"
        )

    config = dict(observation_config or {})
    if uses_structured_observation(config):
        required_config = {
            "wall_sensing_version",
            "map_sensing_boundary_risk_threshold",
        }
        missing = sorted(required_config - set(config))
        if missing:
            raise ValueError(f"observation_config is missing resolved keys: {missing}")
        meta = _layout_meta(
            observation,
            env_family=env_family,
            variant=variant,
            observation_config=config,
        )
        features = [vector]
        if _enabled(config, "include_map"):
            features.append(
                _map_features(
                    meta["maze_map"],
                    env_family=env_family,
                    leading_shape=vector.shape[:-1],
                )
            )
        if _enabled(config, "include_location_sensing") or _enabled(
            config, "include_wall_sensing"
        ):
            sensing = compute_sensing_arrays(position, goal, meta)
            if _enabled(config, "include_location_sensing"):
                features.append(
                    np.concatenate(
                        [sensing["position_cell"], sensing["goal_cell"]],
                        axis=-1,
                    ).astype(np.float32)
                )
            if _enabled(config, "include_wall_sensing"):
                features.append(sensing["neighbor_status"].astype(np.float32))
        vector = np.concatenate(features, axis=-1)

    expected_dim = observation_dim(env_family, config)
    if vector.shape[-1] != expected_dim:
        raise ValueError(
            f"Unexpected {env_family} baseline observation dimension: "
            f"expected {expected_dim}, got {vector.shape[-1]}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{env_family} baseline observation contains non-finite values")
    return np.asarray(vector, dtype=np.float32)


def vectorize_goal_conditioned_observation(
    observation,
    env_family: str,
    *,
    observation_config: Mapping | None = None,
    variant: str | None = None,
    goal_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorize a state and a separately supplied or desired goal.

    Offline relabeling passes achieved positions through ``goal_xy``. Online
    rollout leaves it unset and therefore uses ``desired_goal``.
    """
    if not isinstance(observation, Mapping):
        raise ValueError("CrossMaze goal-conditioned observation must be a mapping")
    state, position, desired_goal = _state_position_and_goal(
        observation, env_family
    )
    expected_state_base = GOAL_CONDITIONED_STATE_DIMS[env_family]
    if state.shape[-1] != expected_state_base:
        raise ValueError(
            f"Unexpected {env_family} goal-conditioned state dimension: "
            f"expected {expected_state_base}, got {state.shape[-1]}"
        )
    goal = desired_goal if goal_xy is None else np.asarray(goal_xy, dtype=np.float32)
    if goal.shape[-1] != GOAL_CONDITIONED_GOAL_DIMS[env_family]:
        raise ValueError(
            f"Unexpected {env_family} goal dimension: expected 2, got {goal.shape[-1]}"
        )
    if goal.shape[:-1] != state.shape[:-1]:
        try:
            goal = np.broadcast_to(goal, state.shape[:-1] + (goal.shape[-1],))
        except ValueError as exc:
            raise ValueError(
                "Goal leading dimensions must match the state leading dimensions"
            ) from exc

    config = dict(observation_config or {})
    state_features = [state]
    goal_features = [goal]
    if uses_structured_observation(config):
        required_config = {
            "wall_sensing_version",
            "map_sensing_boundary_risk_threshold",
        }
        missing = sorted(required_config - set(config))
        if missing:
            raise ValueError(f"observation_config is missing resolved keys: {missing}")
        meta = _layout_meta(
            observation,
            env_family=env_family,
            variant=variant,
            observation_config=config,
        )
        if _enabled(config, "include_map"):
            state_features.append(
                _map_features(
                    meta["maze_map"],
                    env_family=env_family,
                    leading_shape=state.shape[:-1],
                )
            )
        if _enabled(config, "include_location_sensing") or _enabled(
            config, "include_wall_sensing"
        ):
            sensing = compute_sensing_arrays(position, goal, meta)
            if _enabled(config, "include_location_sensing"):
                state_features.append(sensing["position_cell"].astype(np.float32))
                goal_features.append(sensing["goal_cell"].astype(np.float32))
            if _enabled(config, "include_wall_sensing"):
                state_features.append(sensing["neighbor_status"].astype(np.float32))

    state_vector = np.concatenate(state_features, axis=-1)
    goal_vector = np.concatenate(goal_features, axis=-1)
    expected_state_dim, expected_goal_dim = goal_conditioned_observation_dims(
        env_family, config
    )
    if state_vector.shape[-1] != expected_state_dim:
        raise ValueError(
            f"Unexpected {env_family} goal-conditioned state dimension: "
            f"expected {expected_state_dim}, got {state_vector.shape[-1]}"
        )
    if goal_vector.shape[-1] != expected_goal_dim:
        raise ValueError(
            f"Unexpected {env_family} goal dimension: "
            f"expected {expected_goal_dim}, got {goal_vector.shape[-1]}"
        )
    if not np.all(np.isfinite(state_vector)) or not np.all(np.isfinite(goal_vector)):
        raise ValueError(
            f"{env_family} goal-conditioned observation contains non-finite values"
        )
    return (
        np.asarray(state_vector, dtype=np.float32),
        np.asarray(goal_vector, dtype=np.float32),
    )


class BaselineObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        *,
        env_family: str,
        observation_config: Mapping | None = None,
    ):
        super().__init__(env)
        if env_family not in BASE_OBSERVATION_DIMS:
            raise ValueError(f"Unsupported env_family: {env_family!r}")
        self.env_family = env_family
        self.observation_config = dict(observation_config or {})
        # Updated before every vectorization. Rollout evaluation reads this
        # immediately after reset so the recorded pair is the one actually
        # sampled by the environment.
        self.last_crossmaze_state: Mapping | None = None
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim(env_family, self.observation_config),),
            dtype=np.float32,
        )

    def observation(self, observation):
        attached = (
            observation.get(CROSSMAZE_OBS_KEY)
            if isinstance(observation, Mapping)
            else None
        )
        self.last_crossmaze_state = attached if isinstance(attached, Mapping) else None
        return vectorize_observation(
            observation,
            self.env_family,
            observation_config=self.observation_config,
        )


class GoalConditionedObservationWrapper(gym.ObservationWrapper):
    """Expose separate state and desired-goal arrays for CRL and HIQL."""

    def __init__(
        self,
        env,
        *,
        env_family: str,
        observation_config: Mapping | None = None,
    ):
        super().__init__(env)
        if env_family not in BASE_OBSERVATION_DIMS:
            raise ValueError(f"Unsupported env_family: {env_family!r}")
        self.env_family = env_family
        self.observation_config = dict(observation_config or {})
        self.last_crossmaze_state: Mapping | None = None
        state_dim, goal_dim = goal_conditioned_observation_dims(
            env_family, self.observation_config
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(state_dim,),
                    dtype=np.float32,
                ),
                "goal": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(goal_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation):
        attached = (
            observation.get(CROSSMAZE_OBS_KEY)
            if isinstance(observation, Mapping)
            else None
        )
        self.last_crossmaze_state = attached if isinstance(attached, Mapping) else None
        state, goal = vectorize_goal_conditioned_observation(
            observation,
            self.env_family,
            observation_config=self.observation_config,
        )
        return {"state": state, "goal": goal}
