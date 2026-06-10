"""Robosuite adapter for the pick-and-place evaluator.

The evaluator only needs a tiny simulator interface:

    reset() -> PickAndPlaceState
    step(action) -> PickAndPlaceStepResult

Robosuite has a larger API. This file is the adapter between those two worlds.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from rl_neuro_training.evaluator import (
    JOINT_ACTION_SIZE,
    PANDA_JOINT_COUNT,
    PickAndPlaceState,
    PickAndPlaceStepResult,
)


DEFAULT_OBJECT_POSITION_KEYS = (
    "cube_pos",
    "object_pos",
    "Can_pos",
    "Milk_pos",
    "Bread_pos",
    "Cereal_pos",
)

DEFAULT_TARGET_POSITION_KEYS = (
    "target_pos",
    "goal_pos",
    "target_position",
)


DEFAULT_TARGET_POSITION_ATTRIBUTES = (
    "target_pos",
    "goal_pos",
    "bin2_pos",
)


@dataclass(frozen=True, kw_only=True)
class RobosuitePickAndPlaceConfig:
    """Settings for creating and reading a Robosuite pick-and-place env.

    env_name and robots:
        Passed to robosuite when the real environment is created.

    target_position:
        The position where the object should end up.
        This is explicit because Robosuite tasks expose target data differently
        depending on the environment.

    object_position_keys:
        Observation keys we will try, in order, to find the object position.

    render_each_step:
        True means call env.render() after reset and after every step.
    """

    env_name: str = "PickPlace"
    robots: str = "Panda"
    controller_name: Optional[str] = None
    has_renderer: bool = True
    has_offscreen_renderer: bool = False
    use_camera_obs: bool = False
    render_each_step: bool = True
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)

    gripper_position_key: str = "robot0_eef_pos"
    joint_positions_key: str = "robot0_joint_pos"
    gripper_qpos_key: str = "robot0_gripper_qpos"
    object_position_keys: tuple[str, ...] = DEFAULT_OBJECT_POSITION_KEYS
    target_position_keys: tuple[str, ...] = DEFAULT_TARGET_POSITION_KEYS
    target_position_attributes: tuple[str, ...] = DEFAULT_TARGET_POSITION_ATTRIBUTES
    is_grasped_key: Optional[str] = None
    object_speed_key: Optional[str] = None

    target_position: Optional[Sequence[float]] = None
    table_height: float = 0.0
    fully_open_gripper_width: float = 0.08
    grasp_distance: float = 0.04
    control_timestep: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.env_name:
            raise ValueError("env_name cannot be empty")

        if not self.robots:
            raise ValueError("robots cannot be empty")

        if self.fully_open_gripper_width <= 0.0:
            raise ValueError("fully_open_gripper_width must be greater than zero")

        if self.grasp_distance <= 0.0:
            raise ValueError("grasp_distance must be greater than zero")

        if self.control_timestep is not None and self.control_timestep <= 0.0:
            raise ValueError("control_timestep must be greater than zero")

        if not self.object_position_keys:
            raise ValueError("object_position_keys cannot be empty")

        if self.controller_name is not None and not self.controller_name:
            raise ValueError("controller_name cannot be empty")

        can_find_target_later = bool(
            self.target_position_keys or self.target_position_attributes
        )

        if self.target_position is None and not can_find_target_later:
            raise ValueError(
                "target_position keys or attributes are required when "
                "target_position is not set"
            )

        _finite_number(self.table_height, "table_height")

        if self.target_position is not None:
            _vector3(self.target_position, "target_position")


class RobosuitePickAndPlaceSimulator:
    """Wrap one Robosuite environment so the evaluator can use it."""

    def __init__(
        self,
        env: Any,
        config: Optional[RobosuitePickAndPlaceConfig] = None,
    ) -> None:
        if config is None:
            config = RobosuitePickAndPlaceConfig()

        self.env = env
        self.config = config
        self._last_object_position: Optional[np.ndarray] = None

    def reset(self) -> PickAndPlaceState:
        """Start a new episode and return the first evaluator state."""

        reset_result = self.env.reset()
        observation = _observation_from_reset(reset_result)
        self._last_object_position = None
        state = self._state_from_observation(observation)
        self._render_if_enabled()

        return state

    def step(self, action: Sequence[float]) -> PickAndPlaceStepResult:
        """Apply one action to Robosuite and return the evaluator result."""

        env_action = self._action_for_env(action)
        step_result = self.env.step(env_action)
        observation, done = _observation_and_done_from_step(step_result)
        state = self._state_from_observation(observation)
        self._render_if_enabled()

        return PickAndPlaceStepResult(state=state, done=done)

    def close(self) -> None:
        """Close the Robosuite env if it supports close()."""

        close = getattr(self.env, "close", None)

        if close is None:
            return

        close()

    def _state_from_observation(
        self,
        observation: Mapping[str, Any],
    ) -> PickAndPlaceState:
        gripper_position = _vector3_from_observation(
            observation,
            self.config.gripper_position_key,
        )
        object_position = _first_vector3_from_observation(
            observation,
            self.config.object_position_keys,
            "object_position",
        )
        target_position = self._target_position_from_observation(observation)
        joint_positions = _joint_positions_from_observation(
            observation,
            self.config.joint_positions_key,
        )
        gripper_open_amount = self._gripper_open_amount_from_observation(
            observation,
        )
        is_grasped = self._is_grasped(
            observation=observation,
            gripper_position=gripper_position,
            object_position=object_position,
            gripper_open_amount=gripper_open_amount,
        )
        object_speed = self._object_speed_from_observation(
            observation=observation,
            object_position=object_position,
        )

        self._last_object_position = object_position.copy()

        return PickAndPlaceState(
            gripper_position=tuple(gripper_position),
            object_position=tuple(object_position),
            target_position=tuple(target_position),
            joint_positions=tuple(joint_positions),
            gripper_open_amount=gripper_open_amount,
            is_grasped=is_grasped,
            table_height=float(self.config.table_height),
            object_speed=object_speed,
        )

    def _target_position_from_observation(
        self,
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        if self.config.target_position is not None:
            return _vector3(self.config.target_position, "target_position")

        target_position = _optional_first_vector3_from_observation(
            observation,
            self.config.target_position_keys,
        )

        if target_position is not None:
            return target_position

        target_position = _optional_first_vector3_from_attributes(
            self.env,
            self.config.target_position_attributes,
        )

        if target_position is not None:
            return target_position

        tried_keys = ", ".join(self.config.target_position_keys)
        tried_attributes = ", ".join(self.config.target_position_attributes)

        raise ValueError(
            "could not find target_position; "
            f"tried observation keys: {tried_keys}; "
            f"tried env attributes: {tried_attributes}"
        )

    def _gripper_open_amount_from_observation(
        self,
        observation: Mapping[str, Any],
    ) -> float:
        if self.config.gripper_qpos_key not in observation:
            return 1.0

        gripper_qpos = _flat_array(
            observation[self.config.gripper_qpos_key],
            self.config.gripper_qpos_key,
        )
        gripper_width = float(np.sum(np.abs(gripper_qpos)))
        open_amount = gripper_width / self.config.fully_open_gripper_width

        return _clamp_01(open_amount)

    def _is_grasped(
        self,
        observation: Mapping[str, Any],
        gripper_position: np.ndarray,
        object_position: np.ndarray,
        gripper_open_amount: float,
    ) -> bool:
        if self.config.is_grasped_key is not None:
            if self.config.is_grasped_key in observation:
                return bool(observation[self.config.is_grasped_key])

        gripper_to_object_distance = float(
            np.linalg.norm(gripper_position - object_position)
        )

        if gripper_to_object_distance > self.config.grasp_distance:
            return False

        return gripper_open_amount < 0.5

    def _object_speed_from_observation(
        self,
        observation: Mapping[str, Any],
        object_position: np.ndarray,
    ) -> Optional[float]:
        if self.config.object_speed_key is not None:
            if self.config.object_speed_key in observation:
                return _finite_number(
                    observation[self.config.object_speed_key],
                    self.config.object_speed_key,
                )

        if self._last_object_position is None:
            return None

        timestep = self._control_timestep()
        distance = float(np.linalg.norm(object_position - self._last_object_position))

        return distance / timestep

    def _control_timestep(self) -> float:
        if self.config.control_timestep is not None:
            return self.config.control_timestep

        env_timestep = getattr(self.env, "control_timestep", None)

        if env_timestep is None:
            return 1.0

        return _finite_number(env_timestep, "control_timestep")

    def _action_for_env(self, action: Sequence[float]) -> np.ndarray:
        action_array = _flat_array(action, "action")
        env_action_size = _env_action_size(self.env)

        if action_array.size == env_action_size:
            return _clip_action_to_env_bounds(self.env, action_array)

        if action_array.size == 4 and env_action_size >= 4:
            env_action = np.zeros(env_action_size, dtype=float)
            env_action[:3] = action_array[:3]
            env_action[-1] = action_array[3]

            return _clip_action_to_env_bounds(self.env, env_action)

        if action_array.size == JOINT_ACTION_SIZE and env_action_size >= JOINT_ACTION_SIZE:
            env_action = np.zeros(env_action_size, dtype=float)
            env_action[:PANDA_JOINT_COUNT] = action_array[:PANDA_JOINT_COUNT]
            env_action[-1] = action_array[-1]

            return _clip_action_to_env_bounds(self.env, env_action)

        raise ValueError(
            "action has the wrong size for this Robosuite env: "
            f"got {action_array.size}, expected {env_action_size}"
        )

    def _render_if_enabled(self) -> None:
        if not self.config.render_each_step:
            return

        render = getattr(self.env, "render", None)

        if render is None:
            return

        render()


def make_robosuite_pick_and_place_simulator(
    config: Optional[RobosuitePickAndPlaceConfig] = None,
) -> RobosuitePickAndPlaceSimulator:
    """Create a real Robosuite pick-and-place simulator."""

    if config is None:
        config = RobosuitePickAndPlaceConfig()

    robosuite = _import_robosuite()
    env_kwargs = dict(config.env_kwargs)

    if config.controller_name is not None and "controller_configs" not in env_kwargs:
        env_kwargs["controller_configs"] = _load_controller_config(
            robosuite,
            config.controller_name,
            config.robots,
        )

    env = robosuite.make(
        env_name=config.env_name,
        robots=config.robots,
        has_renderer=config.has_renderer,
        has_offscreen_renderer=config.has_offscreen_renderer,
        use_camera_obs=config.use_camera_obs,
        **env_kwargs,
    )

    return RobosuitePickAndPlaceSimulator(env, config)


def make_robosuite_pick_and_place_simulator_factory(
    config: Optional[RobosuitePickAndPlaceConfig] = None,
):
    """Return a simulator factory that the trainer can call."""

    def simulator_factory() -> RobosuitePickAndPlaceSimulator:
        return make_robosuite_pick_and_place_simulator(config)

    return simulator_factory


def _import_robosuite():
    try:
        import robosuite
    except ImportError as error:
        raise ImportError(
            "robosuite is required for the real simulator adapter. "
            "Install robosuite and MuJoCo before running real training."
        ) from error

    return robosuite


def _load_controller_config(robosuite: Any, controller_name: str, robot_name: str):
    load_from_module = getattr(robosuite, "load_controller_config", None)

    if load_from_module is not None:
        return load_from_module(default_controller=controller_name)

    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            load_composite_controller_config,
        )
    except ImportError as error:
        raise ImportError(
            "could not import robosuite controller loader"
        ) from error

    if controller_name == "default":
        return load_composite_controller_config(robot=robot_name)

    return load_composite_controller_config(controller=controller_name)


def _observation_from_reset(reset_result: Any) -> Mapping[str, Any]:
    if isinstance(reset_result, Mapping):
        return reset_result

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation = reset_result[0]

        if isinstance(observation, Mapping):
            return observation

    raise ValueError("env.reset() must return an observation dictionary")


def _observation_and_done_from_step(step_result: Any) -> tuple[Mapping[str, Any], bool]:
    if not isinstance(step_result, tuple):
        raise ValueError("env.step() must return a tuple")

    if len(step_result) == 4:
        observation, _reward, done, _info = step_result
        return _mapping_observation(observation), bool(done)

    if len(step_result) == 5:
        observation, _reward, terminated, truncated, _info = step_result
        done = bool(terminated) or bool(truncated)
        return _mapping_observation(observation), done

    raise ValueError("env.step() must return 4 or 5 values")


def _mapping_observation(observation: Any) -> Mapping[str, Any]:
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a dictionary")

    return observation


def _vector3_from_observation(
    observation: Mapping[str, Any],
    key: str,
) -> np.ndarray:
    if key not in observation:
        raise ValueError(f"observation is missing required key: {key}")

    return _vector3(observation[key], key)


def _first_vector3_from_observation(
    observation: Mapping[str, Any],
    keys: Sequence[str],
    name: str,
) -> np.ndarray:
    for key in keys:
        if key in observation:
            return _vector3(observation[key], key)

    raise ValueError(
        f"observation is missing {name}; tried keys: {', '.join(keys)}"
    )


def _optional_first_vector3_from_observation(
    observation: Mapping[str, Any],
    keys: Sequence[str],
) -> Optional[np.ndarray]:
    for key in keys:
        if key in observation:
            return _vector3(observation[key], key)

    return None


def _optional_first_vector3_from_attributes(
    value: Any,
    attribute_names: Sequence[str],
) -> Optional[np.ndarray]:
    for attribute_name in attribute_names:
        if not hasattr(value, attribute_name):
            continue

        return _vector3(
            getattr(value, attribute_name),
            attribute_name,
        )

    return None


def _joint_positions_from_observation(
    observation: Mapping[str, Any],
    key: str,
) -> np.ndarray:
    if key not in observation:
        raise ValueError(f"observation is missing required key: {key}")

    joint_positions = _flat_array(observation[key], key)

    if joint_positions.size != PANDA_JOINT_COUNT:
        raise ValueError(
            f"{key} must contain exactly {PANDA_JOINT_COUNT} numbers"
        )

    return joint_positions


def _env_action_size(env: Any) -> int:
    action_dim = getattr(env, "action_dim", None)

    if action_dim is not None:
        return int(action_dim)

    bounds = _env_action_bounds(env)

    if bounds is not None:
        low, _high = bounds
        return int(low.size)

    raise ValueError("could not determine Robosuite env action size")


def _env_action_bounds(env: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    action_spec = getattr(env, "action_spec", None)

    if action_spec is None:
        return None

    if callable(action_spec):
        action_spec = action_spec()

    if not isinstance(action_spec, tuple) or len(action_spec) != 2:
        return None

    low = _flat_array(action_spec[0], "action_spec low")
    high = _flat_array(action_spec[1], "action_spec high")

    if low.size != high.size:
        raise ValueError("action_spec low and high must have the same size")

    return low, high


def _clip_action_to_env_bounds(env: Any, action: np.ndarray) -> np.ndarray:
    bounds = _env_action_bounds(env)

    if bounds is None:
        return action.copy()

    low, high = bounds

    if action.size != low.size:
        raise ValueError("action size must match action_spec size")

    return np.clip(action, low, high)


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)

    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 numbers")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must only contain finite numbers")

    return vector


def _flat_array(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a flat array")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must only contain finite numbers")

    return array


def _finite_number(value: float, name: str) -> float:
    number = float(value)

    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")

    return number


def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0

    if value > 1.0:
        return 1.0

    return value
