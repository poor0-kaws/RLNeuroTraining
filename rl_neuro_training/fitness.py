"""Fitness scoring for the first pick-and-place task.

This file only answers one question:

    "How good was this robot's behavior during one attempt?"

It does not know about genomes, neural networks, DEAP, Robosuite, or MuJoCo.
That separation keeps the learning code easier to change later.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Optional, Sequence


Vector3 = Sequence[float]


@dataclass(frozen=True)
class PickAndPlaceObservation:
    """One snapshot from the simulator.

    The evaluator will later create these snapshots from Robosuite data.

    gripper_position:
        Where the robot hand is.

    object_position:
        Where the object is.

    target_position:
        Where the object should end up.

    is_grasped:
        True when the gripper is holding the object.

    table_height:
        Height of the table surface. We use this to know if the object was
        lifted above the table.

    object_speed:
        How fast the object is moving. Stable placement needs the object to be
        near the target and mostly still.
    """

    gripper_position: Vector3
    object_position: Vector3
    target_position: Vector3
    is_grasped: bool
    table_height: float
    object_speed: Optional[float] = None


@dataclass(frozen=True)
class PickAndPlaceFitnessConfig:
    """Numbers that define what "good" means for pick-and-place."""

    reach_weight: float = 0.05
    grasp_weight: float = 0.10
    lift_weight: float = 0.15
    move_weight: float = 0.20
    place_weight: float = 0.50

    useful_reach_distance: float = 0.25
    target_lift_height: float = 0.10
    placement_tolerance: float = 0.04

    stable_speed_limit: float = 0.02
    stable_steps_required: int = 10

    placement_accuracy_share: float = 0.60
    placement_stability_share: float = 0.40

    stable_duration_share: float = 0.80
    stable_earliness_share: float = 0.20


@dataclass(frozen=True)
class PickAndPlaceStageScores:
    """Unweighted scores for each behavior stage.

    Every value is between 0.0 and 1.0.

    These are separated from the final weighted score so we can inspect what
    the robot is learning.
    """

    reaching: float
    grasping: float
    lifting: float
    moving: float
    placing: float
    placement_accuracy: float
    placement_stability: float


@dataclass(frozen=True)
class PickAndPlaceFitnessResult:
    """Final fitness result for one full robot attempt."""

    total: float
    stages: PickAndPlaceStageScores


def score_pick_and_place_episode(
    observations: Iterable[PickAndPlaceObservation],
    config: Optional[PickAndPlaceFitnessConfig] = None,
) -> PickAndPlaceFitnessResult:
    """Score one full pick-and-place attempt.

    The input is the full episode, not just the final frame.

    That means the robot can get credit for earlier useful actions, like
    reaching and lifting, even if the final frame only shows the object sitting
    on the target.
    """

    if config is None:
        config = PickAndPlaceFitnessConfig()

    episode = list(observations)

    if not episode:
        return _empty_result()

    _validate_config(config)

    start_object_position = episode[0].object_position
    target_position = episode[0].target_position
    start_object_to_target_distance = _distance_3d(
        start_object_position,
        target_position,
    )

    best_reaching = 0.0
    best_grasping = 0.0
    best_lifting = 0.0
    best_moving = 0.0
    object_has_been_lifted = False

    for observation in episode:
        reaching = _score_reaching(observation, config)
        grasping = _score_grasping(observation)
        lifting = _score_lifting(observation, config)

        if lifting > 0.0:
            object_has_been_lifted = True

        moving = _score_moving(
            observation=observation,
            start_object_to_target_distance=start_object_to_target_distance,
            object_has_been_lifted=object_has_been_lifted,
        )

        best_reaching = max(best_reaching, reaching)
        best_grasping = max(best_grasping, grasping)
        best_lifting = max(best_lifting, lifting)
        best_moving = max(best_moving, moving)

    final_observation = episode[-1]
    placement_accuracy = _score_placement_accuracy(final_observation, config)
    placement_stability = _score_placement_stability(episode, config)
    placing = _combine_placement_scores(
        placement_accuracy=placement_accuracy,
        placement_stability=placement_stability,
        config=config,
    )

    stages = PickAndPlaceStageScores(
        reaching=best_reaching,
        grasping=best_grasping,
        lifting=best_lifting,
        moving=best_moving,
        placing=placing,
        placement_accuracy=placement_accuracy,
        placement_stability=placement_stability,
    )

    total = _weighted_total(stages, config)

    return PickAndPlaceFitnessResult(total=total, stages=stages)


def _score_reaching(
    observation: PickAndPlaceObservation,
    config: PickAndPlaceFitnessConfig,
) -> float:
    gripper_to_object_distance = _distance_3d(
        observation.gripper_position,
        observation.object_position,
    )

    return _closeness_score(
        distance=gripper_to_object_distance,
        useful_distance=config.useful_reach_distance,
    )


def _score_grasping(observation: PickAndPlaceObservation) -> float:
    if not observation.is_grasped:
        return 0.0

    return 1.0


def _score_lifting(
    observation: PickAndPlaceObservation,
    config: PickAndPlaceFitnessConfig,
) -> float:
    if not observation.is_grasped:
        return 0.0

    lifted_height = observation.object_position[2] - observation.table_height

    return _progress_score(
        current_value=lifted_height,
        target_value=config.target_lift_height,
    )


def _score_moving(
    observation: PickAndPlaceObservation,
    start_object_to_target_distance: float,
    object_has_been_lifted: bool,
) -> float:
    if not object_has_been_lifted:
        return 0.0

    if start_object_to_target_distance <= 0.0:
        return 1.0

    current_object_to_target_distance = _distance_3d(
        observation.object_position,
        observation.target_position,
    )

    progress = (
        start_object_to_target_distance - current_object_to_target_distance
    ) / start_object_to_target_distance

    return _clamp_01(progress)


def _score_placement_accuracy(
    observation: PickAndPlaceObservation,
    config: PickAndPlaceFitnessConfig,
) -> float:
    object_to_target_distance = _distance_3d(
        observation.object_position,
        observation.target_position,
    )

    return _closeness_score(
        distance=object_to_target_distance,
        useful_distance=config.placement_tolerance,
    )


def _score_placement_stability(
    episode: Sequence[PickAndPlaceObservation],
    config: PickAndPlaceFitnessConfig,
) -> float:
    longest_stable_run = 0
    current_stable_run = 0
    first_stable_index = None

    for index, observation in enumerate(episode):
        if _is_stably_placed(observation, config):
            if first_stable_index is None:
                first_stable_index = index

            current_stable_run += 1
            longest_stable_run = max(longest_stable_run, current_stable_run)
            continue

        current_stable_run = 0

    if first_stable_index is None:
        return 0.0

    duration_score = _progress_score(
        current_value=longest_stable_run,
        target_value=config.stable_steps_required,
    )

    earliness_score = 1.0 - (first_stable_index / len(episode))

    stability_score = 0.0
    stability_score += config.stable_duration_share * duration_score
    stability_score += config.stable_earliness_share * earliness_score

    return _clamp_01(stability_score)


def _is_stably_placed(
    observation: PickAndPlaceObservation,
    config: PickAndPlaceFitnessConfig,
) -> bool:
    if observation.object_speed is None:
        return False

    if observation.object_speed > config.stable_speed_limit:
        return False

    object_to_target_distance = _distance_3d(
        observation.object_position,
        observation.target_position,
    )

    if object_to_target_distance > config.placement_tolerance:
        return False

    return True


def _combine_placement_scores(
    placement_accuracy: float,
    placement_stability: float,
    config: PickAndPlaceFitnessConfig,
) -> float:
    accuracy_score = config.placement_accuracy_share * placement_accuracy
    stability_score = config.placement_stability_share * placement_stability

    return _clamp_01(accuracy_score + stability_score)


def _weighted_total(
    stages: PickAndPlaceStageScores,
    config: PickAndPlaceFitnessConfig,
) -> float:
    total = 0.0
    total += config.reach_weight * stages.reaching
    total += config.grasp_weight * stages.grasping
    total += config.lift_weight * stages.lifting
    total += config.move_weight * stages.moving
    total += config.place_weight * stages.placing

    return _clamp_01(total)


def _closeness_score(distance: float, useful_distance: float) -> float:
    if useful_distance <= 0.0:
        raise ValueError("useful_distance must be greater than zero")

    if distance <= 0.0:
        return 1.0

    if distance >= useful_distance:
        return 0.0

    return 1.0 - (distance / useful_distance)


def _progress_score(current_value: float, target_value: float) -> float:
    if target_value <= 0.0:
        raise ValueError("target_value must be greater than zero")

    return _clamp_01(current_value / target_value)


def _distance_3d(first: Vector3, second: Vector3) -> float:
    _validate_vector3(first, "first")
    _validate_vector3(second, "second")

    x_distance = first[0] - second[0]
    y_distance = first[1] - second[1]
    z_distance = first[2] - second[2]

    return sqrt(
        x_distance * x_distance
        + y_distance * y_distance
        + z_distance * z_distance
    )


def _validate_vector3(value: Vector3, name: str) -> None:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")


def _validate_config(config: PickAndPlaceFitnessConfig) -> None:
    total_weight = (
        config.reach_weight
        + config.grasp_weight
        + config.lift_weight
        + config.move_weight
        + config.place_weight
    )

    if abs(total_weight - 1.0) > 0.000001:
        raise ValueError("fitness weights must add up to 1.0")

    placement_share = (
        config.placement_accuracy_share + config.placement_stability_share
    )

    if abs(placement_share - 1.0) > 0.000001:
        raise ValueError("placement shares must add up to 1.0")

    if config.stable_steps_required <= 0:
        raise ValueError("stable_steps_required must be greater than zero")

    stable_share = config.stable_duration_share + config.stable_earliness_share

    if abs(stable_share - 1.0) > 0.000001:
        raise ValueError("stable placement shares must add up to 1.0")


def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0

    if value > 1.0:
        return 1.0

    return value


def _empty_result() -> PickAndPlaceFitnessResult:
    stages = PickAndPlaceStageScores(
        reaching=0.0,
        grasping=0.0,
        lifting=0.0,
        moving=0.0,
        placing=0.0,
        placement_accuracy=0.0,
        placement_stability=0.0,
    )

    return PickAndPlaceFitnessResult(total=0.0, stages=stages)
