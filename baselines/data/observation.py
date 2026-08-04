from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import gymnasium as gym
import numpy as np

from crossmaze import (
    CROSSMAZE_OBS_KEY,
    DYNAMIC_MAP_CURRENT,
    DYNAMIC_MAP_GOAL,
    DYNAMIC_MAP_SUCCESS,
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

GCRL_GOAL_SEMANTICS = "full_observation_v1"
OBSERVATION_SCHEMA_VERSION = "baseline_observation_v2"
DYNAMIC_MAP_GOAL_SEMANTICS = "environment_desired_goal_v1"

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
            "include_dynamic_map",
            "include_location_sensing",
            "include_wall_sensing",
        )
    )


@lru_cache(maxsize=None)
def _registered_family_map_shape(env_family: str) -> tuple[int, int]:
    """Return the legacy fixed map slot derived from registered variants."""
    if env_family not in BASE_OBSERVATION_DIMS:
        raise ValueError(f"Unsupported env_family: {env_family!r}")
    shapes = []
    for variant in list_variants(env_family):
        maze_map = get_env_facts(env_family, variant)["maze_map"]
        shapes.append((len(maze_map), len(maze_map[0])))
    return max(rows for rows, _cols in shapes), max(cols for _rows, cols in shapes)


def family_map_shape(
    env_family: str,
    observation_config: Mapping | None = None,
) -> tuple[int, int]:
    """Return the fixed map slot for one resolved baseline observation schema.

    Registered-only runs preserve their historical family dimensions.  A
    seed-map selection records a possibly larger resolved ``map_shape`` in the
    observation config so every offline and rollout map uses one isomorphic
    padded vector without changing old checkpoint schemas.
    """

    registered = _registered_family_map_shape(env_family)
    configured = (observation_config or {}).get("map_shape")
    if configured is None:
        return registered
    if (
        not isinstance(configured, (list, tuple))
        or len(configured) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in configured
        )
    ):
        raise ValueError(
            "observation.map_shape must be [rows, cols] with positive integers"
        )
    rows, cols = int(configured[0]), int(configured[1])
    if rows < registered[0] or cols < registered[1]:
        raise ValueError(
            f"observation.map_shape {(rows, cols)} is smaller than registered "
            f"{env_family} slot {registered}"
        )
    return rows, cols


def observation_dim(env_family: str, observation_config: Mapping | None = None) -> int:
    if env_family not in BASE_OBSERVATION_DIMS:
        raise ValueError(f"Unsupported env_family: {env_family!r}")
    dimension = BASE_OBSERVATION_DIMS[env_family]
    if _enabled(observation_config, "include_map"):
        rows, cols = family_map_shape(env_family, observation_config)
        dimension += rows * cols
    if _enabled(observation_config, "include_dynamic_map"):
        rows, cols = family_map_shape(env_family, observation_config)
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
        rows, cols = family_map_shape(env_family, observation_config)
        components.append(
            {
                "name": "map",
                "dimension": rows * cols,
                "shape": [rows, cols],
                "flatten_order": "row-major",
                "padding_value": MAP_PADDING_VALUE,
            }
        )
    if _enabled(observation_config, "include_dynamic_map"):
        rows, cols = family_map_shape(env_family, observation_config)
        components.append(
            {
                "name": "dynamic_map",
                "dimension": rows * cols,
                "shape": [rows, cols],
                "flatten_order": "row-major",
                "padding_value": MAP_PADDING_VALUE,
                "status_codes": {
                    "open": 0,
                    "wall": 1,
                    "current": DYNAMIC_MAP_CURRENT,
                    "goal": DYNAMIC_MAP_GOAL,
                    "current_and_goal": DYNAMIC_MAP_SUCCESS,
                },
                "goal_semantics": DYNAMIC_MAP_GOAL_SEMANTICS,
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
        "version": OBSERVATION_SCHEMA_VERSION,
        "base": base_name,
        "dimension": observation_dim(env_family, observation_config),
        "components": components,
    }


def goal_conditioned_observation_dims(
    env_family: str,
    observation_config: Mapping | None = None,
) -> tuple[int, int]:
    """Return the identical full-observation dimensions used by CRL/HIQL."""
    if env_family not in GOAL_CONDITIONED_STATE_DIMS:
        raise ValueError(f"Unsupported env_family: {env_family!r}")
    state_dim = GOAL_CONDITIONED_STATE_DIMS[env_family]
    if _enabled(observation_config, "include_map"):
        rows, cols = family_map_shape(env_family, observation_config)
        state_dim += rows * cols
    if _enabled(observation_config, "include_dynamic_map"):
        rows, cols = family_map_shape(env_family, observation_config)
        state_dim += rows * cols
    if _enabled(observation_config, "include_location_sensing"):
        state_dim += 2
    if _enabled(observation_config, "include_wall_sensing"):
        state_dim += WALL_SENSING_DIM
    return state_dim, state_dim


def goal_conditioned_observation_schema(
    env_family: str,
    observation_config: Mapping | None = None,
) -> dict:
    state_dim, goal_dim = goal_conditioned_observation_dims(
        env_family, observation_config
    )
    components = [
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
    if _enabled(observation_config, "include_map"):
        rows, cols = family_map_shape(env_family, observation_config)
        components.append(
            {
                "name": "map",
                "dimension": rows * cols,
                "shape": [rows, cols],
                "flatten_order": "row-major",
                "padding_value": MAP_PADDING_VALUE,
            }
        )
    if _enabled(observation_config, "include_dynamic_map"):
        rows, cols = family_map_shape(env_family, observation_config)
        components.append(
            {
                "name": "dynamic_map",
                "dimension": rows * cols,
                "shape": [rows, cols],
                "flatten_order": "row-major",
                "padding_value": MAP_PADDING_VALUE,
                "status_codes": {
                    "open": 0,
                    "wall": 1,
                    "current": DYNAMIC_MAP_CURRENT,
                    "goal": DYNAMIC_MAP_GOAL,
                    "current_and_goal": DYNAMIC_MAP_SUCCESS,
                },
                "goal_semantics": DYNAMIC_MAP_GOAL_SEMANTICS,
            }
        )
    if _enabled(observation_config, "include_location_sensing"):
        components.append(
            {
                "name": "position_cell",
                "dimension": 2,
                "order": ["row", "col"],
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
        "version": OBSERVATION_SCHEMA_VERSION,
        "goal_semantics": GCRL_GOAL_SEMANTICS,
        "state_dimension": state_dim,
        "goal_dimension": goal_dim,
        "state_components": components,
        "goal_components": [dict(component) for component in components],
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
    observation_config: Mapping | None = None,
) -> np.ndarray:
    target_rows, target_cols = family_map_shape(env_family, observation_config)
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


def _dynamic_map_features(
    maze_map: list[list[object]],
    *,
    env_family: str,
    position_cell: np.ndarray,
    goal_cell: np.ndarray,
    observation_config: Mapping | None = None,
) -> np.ndarray:
    """Build padded C/G/S maps for arbitrary leading batch dimensions."""
    position_cell = np.asarray(position_cell, dtype=np.int64)
    goal_cell = np.asarray(goal_cell, dtype=np.int64)
    if position_cell.shape != goal_cell.shape or position_cell.shape[-1:] != (2,):
        raise ValueError("position_cell and goal_cell must have matching [..., 2] shapes")
    leading_shape = position_cell.shape[:-1]
    base = np.array(
        _map_features(
            maze_map,
            env_family=env_family,
            leading_shape=leading_shape,
            observation_config=observation_config,
        ),
        dtype=np.float32,
        copy=True,
    )
    target_rows, target_cols = family_map_shape(env_family, observation_config)
    position_indices = position_cell[..., 0] * target_cols + position_cell[..., 1]
    goal_indices = goal_cell[..., 0] * target_cols + goal_cell[..., 1]
    if (
        np.any(position_cell < 0)
        or np.any(goal_cell < 0)
        or np.any(position_cell[..., 0] >= target_rows)
        or np.any(goal_cell[..., 0] >= target_rows)
        or np.any(position_cell[..., 1] >= target_cols)
        or np.any(goal_cell[..., 1] >= target_cols)
    ):
        raise ValueError("Dynamic-map cells fall outside the family map slot")
    rows = base.reshape((-1, base.shape[-1]))
    flat_position = position_indices.reshape(-1)
    flat_goal = goal_indices.reshape(-1)
    row_indices = np.arange(len(rows), dtype=np.int64)
    distinct = flat_position != flat_goal
    rows[row_indices[distinct], flat_position[distinct]] = DYNAMIC_MAP_CURRENT
    rows[row_indices[distinct], flat_goal[distinct]] = DYNAMIC_MAP_GOAL
    rows[row_indices[~distinct], flat_position[~distinct]] = DYNAMIC_MAP_SUCCESS
    return rows.reshape(leading_shape + (base.shape[-1],))


def gcrl_dynamic_map_indices(
    observation: Mapping,
    env_family: str,
    *,
    observation_config: Mapping,
    variant: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded flat current/desired-goal cell indices for compact dmaps."""
    _state, position, goal = _state_position_and_goal(observation, env_family)
    meta = _layout_meta(
        observation,
        env_family=env_family,
        variant=variant,
        observation_config=observation_config,
    )
    sensing = compute_sensing_arrays(position, goal, meta)
    _rows, target_cols = family_map_shape(env_family, observation_config)
    position_cell = np.asarray(sensing["position_cell"], dtype=np.int64)
    goal_cell = np.asarray(sensing["goal_cell"], dtype=np.int64)
    return (
        position_cell[..., 0] * target_cols + position_cell[..., 1],
        goal_cell[..., 0] * target_cols + goal_cell[..., 1],
    )


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
                    observation_config=config,
                )
            )
        if _enabled(config, "include_dynamic_map"):
            sensing = compute_sensing_arrays(position, goal, meta)
            features.append(
                _dynamic_map_features(
                    meta["maze_map"],
                    env_family=env_family,
                    position_cell=sensing["position_cell"],
                    goal_cell=sensing["goal_cell"],
                    observation_config=config,
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


def vectorize_gcrl_state_observation(
    observation,
    env_family: str,
    *,
    observation_config: Mapping | None = None,
    variant: str | None = None,
):
    """Vectorize one full GCRL state; relabeled goals reuse these exact rows."""
    if not isinstance(observation, Mapping):
        raise ValueError("CrossMaze goal-conditioned observation must be a mapping")
    state, position, desired_goal = _state_position_and_goal(observation, env_family)
    expected_state_base = GOAL_CONDITIONED_STATE_DIMS[env_family]
    if state.shape[-1] != expected_state_base:
        raise ValueError(
            f"Unexpected {env_family} goal-conditioned state dimension: "
            f"expected {expected_state_base}, got {state.shape[-1]}"
        )
    config = dict(observation_config or {})
    state_features = [state]
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
                    observation_config=config,
                )
            )
        sensing = None
        if _enabled(config, "include_dynamic_map"):
            sensing = compute_sensing_arrays(position, desired_goal, meta)
            state_features.append(
                _dynamic_map_features(
                    meta["maze_map"],
                    env_family=env_family,
                    position_cell=sensing["position_cell"],
                    goal_cell=sensing["goal_cell"],
                    observation_config=config,
                )
            )
        if _enabled(config, "include_location_sensing") or _enabled(
            config, "include_wall_sensing"
        ):
            if sensing is None:
                sensing = compute_sensing_arrays(position, desired_goal, meta)
            if _enabled(config, "include_location_sensing"):
                state_features.append(sensing["position_cell"].astype(np.float32))
            if _enabled(config, "include_wall_sensing"):
                state_features.append(sensing["neighbor_status"].astype(np.float32))

    state_vector = np.concatenate(state_features, axis=-1)
    expected_state_dim, _ = goal_conditioned_observation_dims(env_family, config)
    if state_vector.shape[-1] != expected_state_dim:
        raise ValueError(
            f"Unexpected {env_family} goal-conditioned state dimension: "
            f"expected {expected_state_dim}, got {state_vector.shape[-1]}"
        )
    if not np.all(np.isfinite(state_vector)):
        raise ValueError(
            f"{env_family} goal-conditioned observation contains non-finite values"
        )
    return np.asarray(state_vector, dtype=np.float32)


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


class GoalConditionedObservationWrapper(gym.Wrapper):
    """Expose isomorphic full state/goal observations for CRL and HIQL.

    The goal is captured with the OGBench maze protocol: reset once, stabilize
    the robot with five seeded random actions, teleport only qpos xy to the
    desired target and observe it, then repeat the original reset for rollout.
    """

    STABILIZATION_STEPS = 5

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
        self._episode_goal: np.ndarray | None = None
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

    def _state_vector(self, observation) -> np.ndarray:
        attached = (
            observation.get(CROSSMAZE_OBS_KEY)
            if isinstance(observation, Mapping)
            else None
        )
        self.last_crossmaze_state = attached if isinstance(attached, Mapping) else None
        return vectorize_gcrl_state_observation(
            observation,
            self.env_family,
            observation_config=self.observation_config,
        )

    def _goal_observation_at_target(self, target_xy: np.ndarray):
        base_env = self.env.unwrapped
        if self.env_family == "pointmaze":
            agent_env = getattr(base_env, "point_env", None)
        else:
            agent_env = getattr(base_env, "ant_env", None)
        if agent_env is None or not hasattr(agent_env, "set_state"):
            raise TypeError(
                "GCRL full-observation rollout requires the Gymnasium Robotics "
                f"{self.env_family} v4 state API"
            )
        qpos = np.asarray(base_env.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(base_env.data.qvel, dtype=np.float64).copy()
        qpos[:2] = np.asarray(target_xy, dtype=np.float64)[:2]
        agent_env.set_state(qpos, qvel)
        agent_observation = agent_env._get_obs()
        if isinstance(agent_observation, tuple):
            agent_observation = agent_observation[0]
        observation = base_env._get_obs(agent_observation)
        enrich = getattr(self.env, "_enrich", None)
        if enrich is None:
            raise TypeError("GCRL full-observation rollout requires CrossMazeEnv enrichment")
        return enrich(observation)

    @staticmethod
    def _assert_repeatable_reset(first, second) -> None:
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in first or key in second:
                if key not in first or key not in second or not np.array_equal(
                    np.asarray(first[key]), np.asarray(second[key])
                ):
                    raise RuntimeError(
                        "GCRL goal capture changed the deterministic reset "
                        f"observation field {key!r}"
                    )
        first_crossmaze = first.get(CROSSMAZE_OBS_KEY)
        second_crossmaze = second.get(CROSSMAZE_OBS_KEY)
        if isinstance(first_crossmaze, Mapping) and isinstance(second_crossmaze, Mapping):
            for key in ("position_cell", "goal_cell", "position_xy", "goal_xy"):
                if not np.array_equal(
                    np.asarray(first_crossmaze.get(key)),
                    np.asarray(second_crossmaze.get(key)),
                ):
                    raise RuntimeError(
                        "GCRL goal capture changed the deterministic CrossMaze "
                        f"reset field {key!r}"
                    )

    def reset(self, *, seed=None, options=None, **kwargs):
        if seed is None:
            raise ValueError("GCRL rollout reset requires an explicit deterministic seed")
        self.action_space.seed(int(seed))
        preliminary, _ = self.env.reset(seed=seed, options=options, **kwargs)
        target_xy = np.asarray(preliminary["desired_goal"], dtype=np.float32).copy()
        for _ in range(self.STABILIZATION_STEPS):
            self.env.step(self.action_space.sample())
        goal_observation = self._goal_observation_at_target(target_xy)
        goal_achieved = (
            np.asarray(goal_observation["observation"], dtype=np.float32)[:2]
            if self.env_family == "pointmaze"
            else np.asarray(goal_observation["achieved_goal"], dtype=np.float32)[:2]
        )
        if not np.allclose(goal_achieved, target_xy, rtol=0.0, atol=1e-6):
            raise RuntimeError("Captured GCRL goal observation is not at desired target xy")
        goal_vector = vectorize_gcrl_state_observation(
            goal_observation,
            self.env_family,
            observation_config=self.observation_config,
        )

        actual, info = self.env.reset(seed=seed, options=options, **kwargs)
        self._assert_repeatable_reset(preliminary, actual)
        state_vector = self._state_vector(actual)
        if state_vector.shape != goal_vector.shape:
            raise RuntimeError("GCRL rollout state and goal schemas are not isomorphic")
        self._episode_goal = np.asarray(goal_vector, dtype=np.float32)
        return {"state": state_vector, "goal": self._episode_goal.copy()}, info

    def step(self, action):
        if self._episode_goal is None:
            raise RuntimeError("GCRL environment must be reset before step")
        observation, reward, terminated, truncated, info = self.env.step(action)
        state_vector = self._state_vector(observation)
        # The evaluation protocol is single-goal and ends at first success.
        terminated = bool(terminated) or bool(info.get("success", False))
        return (
            {"state": state_vector, "goal": self._episode_goal.copy()},
            reward,
            terminated,
            truncated,
            info,
        )
