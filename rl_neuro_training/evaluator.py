"""Evaluator for running genomes through a pick-and-place episode.

The evaluator connects three things:

    genome -> neural network -> simulator actions

It does not create genomes. It does not choose winners. It does not mutate
anything. Its only job is to let one robot brain try the task and then ask the
fitness function how well that attempt went.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from rl_neuro_training.fitness import (
    PickAndPlaceFitnessConfig,
    PickAndPlaceFitnessResult,
    PickAndPlaceObservation,
    score_pick_and_place_episode,
)
from rl_neuro_training.genome import (
    NetworkShape,
    decode_genome,
    run_network,
)


PANDA_JOINT_COUNT = 7

PICK_AND_PLACE_OBSERVATION_SIZE = 25

END_EFFECTOR_ACTION_MODE = "end_effector"
JOINT_ACTION_MODE = "joint"

END_EFFECTOR_ACTION_SIZE = 4
JOINT_ACTION_SIZE = 8


@dataclass(frozen=True)
class PickAndPlaceState:
    """One simulator state that the evaluator can understand.

    The simulator will later produce this from Robosuite data.

    gripper_position:
        Where the robot hand is.

    object_position:
        Where the object is.

    target_position:
        Where the object should be placed.

    joint_positions:
        The seven Panda arm joint positions.

    gripper_open_amount:
        0.0 means fully closed. 1.0 means fully open.

    is_grasped:
        True when the gripper is holding the object.

    table_height:
        Height of the table surface.

    object_speed:
        How fast the object is moving. The fitness function uses this to reward
        stable placement.
    """

    gripper_position: Sequence[float]
    object_position: Sequence[float]
    target_position: Sequence[float]
    joint_positions: Sequence[float]
    gripper_open_amount: float
    is_grasped: bool
    table_height: float
    object_speed: Optional[float] = None


@dataclass(frozen=True)
class PickAndPlaceStepResult:
    """What the simulator returns after one action."""

    state: PickAndPlaceState
    done: bool = False


class PickAndPlaceSimulator(Protocol):
    """Small simulator interface needed by the evaluator.

    A real Robosuite wrapper can implement this later.
    A fake test simulator can also implement this now.
    """

    def reset(self) -> PickAndPlaceState:
        """Start a fresh episode and return the first state."""

    def step(self, action: Sequence[float]) -> PickAndPlaceStepResult:
        """Apply one action and return the next state."""


@dataclass(frozen=True, kw_only=True)
class PickAndPlaceEvaluatorConfig:
    """Settings for evaluating genomes."""

    network_shape: NetworkShape
    max_steps: int = 500
    action_mode: str = END_EFFECTOR_ACTION_MODE
    end_effector_step_size: float = 0.05
    joint_step_size: float = 0.05
    fitness_config: Optional[PickAndPlaceFitnessConfig] = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        if self.network_shape.input_size != PICK_AND_PLACE_OBSERVATION_SIZE:
            raise ValueError(
                "network_shape input_size must be "
                f"{PICK_AND_PLACE_OBSERVATION_SIZE}"
            )

        expected_output_size = _expected_action_size(self.action_mode)

        if self.network_shape.output_size != expected_output_size:
            raise ValueError(
                "network_shape output_size must be "
                f"{expected_output_size} for {self.action_mode} action mode"
            )

        if self.end_effector_step_size <= 0.0:
            raise ValueError("end_effector_step_size must be greater than zero")

        if self.joint_step_size <= 0.0:
            raise ValueError("joint_step_size must be greater than zero")


@dataclass(frozen=True)
class GenomeEvaluationResult:
    """The result from evaluating one genome."""

    fitness: PickAndPlaceFitnessResult
    states: tuple[PickAndPlaceState, ...]
    actions: tuple[np.ndarray, ...]


def make_pick_and_place_network_shape(
    hidden_size: int = 32,
    action_mode: str = END_EFFECTOR_ACTION_MODE,
) -> NetworkShape:
    """Create the default network shape for this task.

    The input size is 25 because the observation vector has 25 numbers.
    The output size is 4 for end-effector control, or 8 for joint control.
    """

    return NetworkShape(
        input_size=PICK_AND_PLACE_OBSERVATION_SIZE,
        hidden_size=hidden_size,
        output_size=_expected_action_size(action_mode),
    )


def evaluate_genome(
    genome: Sequence[float],
    simulator: PickAndPlaceSimulator,
    config: PickAndPlaceEvaluatorConfig,
) -> GenomeEvaluationResult:
    """Run one genome through one episode and return its fitness."""

    weights = decode_genome(genome, config.network_shape)

    current_state = simulator.reset()
    states = [current_state]
    actions = []

    for _ in range(config.max_steps):
        observation_vector = build_pick_and_place_observation_vector(
            current_state,
        )
        network_output = run_network(observation_vector, weights)
        action = map_network_output_to_action(network_output, config)

        step_result = simulator.step(action)

        actions.append(action)
        current_state = step_result.state
        states.append(current_state)

        if step_result.done:
            break

    fitness_observations = [
        make_fitness_observation(state) for state in states
    ]
    fitness = score_pick_and_place_episode(
        fitness_observations,
        config=config.fitness_config,
    )

    return GenomeEvaluationResult(
        fitness=fitness,
        states=tuple(states),
        actions=tuple(actions),
    )


def evaluate_population(
    population: Sequence[Sequence[float]],
    simulator_factory: Callable[[], PickAndPlaceSimulator],
    config: PickAndPlaceEvaluatorConfig,
) -> list[GenomeEvaluationResult]:
    """Evaluate every genome in a population.

    The simulator factory gives each genome a fresh simulator. That matters
    because one robot's episode should not leak state into the next robot's
    episode.
    """

    population_array = np.asarray(population, dtype=float)

    if population_array.ndim != 2:
        raise ValueError("population must be a 2D array")

    if population_array.shape[1] != config.network_shape.genome_length:
        raise ValueError(
            "population genomes have the wrong length: "
            f"expected {config.network_shape.genome_length}, "
            f"got {population_array.shape[1]}"
        )

    results = []

    for genome in population_array:
        simulator = simulator_factory()
        result = evaluate_genome(genome, simulator, config)
        results.append(result)

    return results


def build_pick_and_place_observation_vector(
    state: PickAndPlaceState,
) -> np.ndarray:
    """Turn one simulator state into the 25 numbers the network sees."""

    gripper_position = _as_vector3(state.gripper_position, "gripper_position")
    object_position = _as_vector3(state.object_position, "object_position")
    target_position = _as_vector3(state.target_position, "target_position")
    joint_positions = _as_joint_positions(state.joint_positions)

    gripper_open_amount = _as_finite_number(
        state.gripper_open_amount,
        "gripper_open_amount",
    )
    table_height = _as_finite_number(state.table_height, "table_height")

    if gripper_open_amount < 0.0 or gripper_open_amount > 1.0:
        raise ValueError("gripper_open_amount must be between 0.0 and 1.0")

    object_to_target = target_position - object_position
    gripper_to_object = object_position - gripper_position
    object_height_above_table = object_position[2] - table_height
    is_grasped_number = 1.0 if state.is_grasped else 0.0

    observation_vector = np.concatenate(
        [
            gripper_position,
            object_position,
            target_position,
            object_to_target,
            gripper_to_object,
            joint_positions,
            np.array(
                [
                    gripper_open_amount,
                    object_height_above_table,
                    is_grasped_number,
                ]
            ),
        ]
    )

    if observation_vector.size != PICK_AND_PLACE_OBSERVATION_SIZE:
        raise ValueError("observation vector has the wrong size")

    return observation_vector


def make_fitness_observation(
    state: PickAndPlaceState,
) -> PickAndPlaceObservation:
    """Convert evaluator state into the smaller fitness observation."""

    gripper_position = _as_vector3(state.gripper_position, "gripper_position")
    object_position = _as_vector3(state.object_position, "object_position")
    target_position = _as_vector3(state.target_position, "target_position")

    table_height = _as_finite_number(state.table_height, "table_height")

    object_speed = None

    if state.object_speed is not None:
        object_speed = _as_finite_number(state.object_speed, "object_speed")

    return PickAndPlaceObservation(
        gripper_position=tuple(gripper_position),
        object_position=tuple(object_position),
        target_position=tuple(target_position),
        is_grasped=state.is_grasped,
        table_height=table_height,
        object_speed=object_speed,
    )


def map_network_output_to_action(
    network_output: Sequence[float],
    config: PickAndPlaceEvaluatorConfig,
) -> np.ndarray:
    """Turn network output into the action sent to the simulator."""

    if config.action_mode == END_EFFECTOR_ACTION_MODE:
        return map_end_effector_action(
            network_output,
            max_position_delta=config.end_effector_step_size,
        )

    if config.action_mode == JOINT_ACTION_MODE:
        return map_joint_action(
            network_output,
            max_joint_delta=config.joint_step_size,
        )

    raise ValueError(f"unknown action_mode: {config.action_mode}")


def map_end_effector_action(
    network_output: Sequence[float],
    max_position_delta: float = 0.05,
) -> np.ndarray:
    """Map 4 network outputs to end-effector control.

    The four outputs mean:

        0: move gripper in x
        1: move gripper in y
        2: move gripper in z
        3: open or close gripper
    """

    if max_position_delta <= 0.0:
        raise ValueError("max_position_delta must be greater than zero")

    output = _as_action_array(
        network_output,
        expected_size=END_EFFECTOR_ACTION_SIZE,
    )

    position_delta = output[:3] * max_position_delta
    gripper_command = output[3:4]

    return np.concatenate([position_delta, gripper_command])


def map_joint_action(
    network_output: Sequence[float],
    max_joint_delta: float = 0.05,
) -> np.ndarray:
    """Map 8 network outputs to direct Panda joint control.

    The first seven outputs move the seven arm joints.
    The last output opens or closes the gripper.
    """

    if max_joint_delta <= 0.0:
        raise ValueError("max_joint_delta must be greater than zero")

    output = _as_action_array(
        network_output,
        expected_size=JOINT_ACTION_SIZE,
    )

    joint_delta = output[:PANDA_JOINT_COUNT] * max_joint_delta
    gripper_command = output[PANDA_JOINT_COUNT:]

    return np.concatenate([joint_delta, gripper_command])


def _expected_action_size(action_mode: str) -> int:
    if action_mode == END_EFFECTOR_ACTION_MODE:
        return END_EFFECTOR_ACTION_SIZE

    if action_mode == JOINT_ACTION_MODE:
        return JOINT_ACTION_SIZE

    raise ValueError(f"unknown action_mode: {action_mode}")


def _as_vector3(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)

    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 numbers")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must only contain finite numbers")

    return vector


def _as_joint_positions(value: Sequence[float]) -> np.ndarray:
    joint_positions = np.asarray(value, dtype=float)

    if joint_positions.shape != (PANDA_JOINT_COUNT,):
        raise ValueError(
            f"joint_positions must contain exactly {PANDA_JOINT_COUNT} numbers"
        )

    if not np.all(np.isfinite(joint_positions)):
        raise ValueError("joint_positions must only contain finite numbers")

    return joint_positions


def _as_action_array(
    value: Sequence[float],
    expected_size: int,
) -> np.ndarray:
    action = np.asarray(value, dtype=float)

    if action.shape != (expected_size,):
        raise ValueError(f"network output must contain {expected_size} numbers")

    if not np.all(np.isfinite(action)):
        raise ValueError("network output must only contain finite numbers")

    return action


def _as_finite_number(value: float, name: str) -> float:
    number = float(value)

    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")

    return number
