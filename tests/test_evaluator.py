import unittest

import numpy as np

from rl_neuro_training.evaluator import (
    END_EFFECTOR_ACTION_MODE,
    JOINT_ACTION_MODE,
    PickAndPlaceEvaluatorConfig,
    PickAndPlaceState,
    PickAndPlaceStepResult,
    build_pick_and_place_observation_vector,
    evaluate_genome,
    evaluate_population,
    make_fitness_observation,
    make_pick_and_place_network_shape,
    map_end_effector_action,
    map_joint_action,
)
from rl_neuro_training.genome import NetworkShape


class FakePickAndPlaceSimulator:
    def __init__(self, states):
        self.states = list(states)
        self.actions = []
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.states[0]

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float))

        if self.current_index < len(self.states) - 1:
            self.current_index += 1

        done = self.current_index == len(self.states) - 1

        return PickAndPlaceStepResult(
            state=self.states[self.current_index],
            done=done,
        )


def make_state(
    gripper_position=(0.0, 0.0, 0.0),
    object_position=(0.0, 0.0, 0.0),
    target_position=(1.0, 0.0, 0.0),
    joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    gripper_open_amount=1.0,
    is_grasped=False,
    table_height=0.0,
    object_speed=None,
):
    return PickAndPlaceState(
        gripper_position=gripper_position,
        object_position=object_position,
        target_position=target_position,
        joint_positions=joint_positions,
        gripper_open_amount=gripper_open_amount,
        is_grasped=is_grasped,
        table_height=table_height,
        object_speed=object_speed,
    )


class EvaluatorTest(unittest.TestCase):
    def test_default_network_shape_matches_end_effector_contract(self):
        shape = make_pick_and_place_network_shape(hidden_size=16)

        self.assertEqual(shape.input_size, 25)
        self.assertEqual(shape.hidden_size, 16)
        self.assertEqual(shape.output_size, 4)

    def test_joint_network_shape_matches_joint_contract(self):
        shape = make_pick_and_place_network_shape(
            hidden_size=16,
            action_mode=JOINT_ACTION_MODE,
        )

        self.assertEqual(shape.input_size, 25)
        self.assertEqual(shape.hidden_size, 16)
        self.assertEqual(shape.output_size, 8)

    def test_observation_vector_has_expected_order(self):
        state = make_state(
            gripper_position=(1.0, 2.0, 3.0),
            object_position=(4.0, 6.0, 8.0),
            target_position=(7.0, 10.0, 13.0),
            joint_positions=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            gripper_open_amount=0.5,
            is_grasped=True,
            table_height=7.0,
        )

        observation = build_pick_and_place_observation_vector(state)

        np.testing.assert_array_equal(
            observation,
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    8.0,
                    7.0,
                    10.0,
                    13.0,
                    3.0,
                    4.0,
                    5.0,
                    3.0,
                    4.0,
                    5.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    0.5,
                    1.0,
                    1.0,
                ]
            ),
        )

    def test_make_fitness_observation_keeps_fitness_fields(self):
        state = make_state(
            gripper_position=(1.0, 2.0, 3.0),
            object_position=(4.0, 5.0, 6.0),
            target_position=(7.0, 8.0, 9.0),
            is_grasped=True,
            table_height=0.2,
            object_speed=0.01,
        )

        observation = make_fitness_observation(state)

        self.assertEqual(observation.gripper_position, (1.0, 2.0, 3.0))
        self.assertEqual(observation.object_position, (4.0, 5.0, 6.0))
        self.assertEqual(observation.target_position, (7.0, 8.0, 9.0))
        self.assertTrue(observation.is_grasped)
        self.assertEqual(observation.table_height, 0.2)
        self.assertEqual(observation.object_speed, 0.01)

    def test_end_effector_action_scales_position_but_not_gripper(self):
        action = map_end_effector_action(
            [1.0, -0.5, 0.0, -1.0],
            max_position_delta=0.05,
        )

        np.testing.assert_array_equal(
            action,
            np.array([0.05, -0.025, 0.0, -1.0]),
        )

    def test_joint_action_scales_joints_but_not_gripper(self):
        action = map_joint_action(
            [1.0, -1.0, 0.5, 0.0, 0.25, -0.25, 0.75, -1.0],
            max_joint_delta=0.1,
        )

        np.testing.assert_array_equal(
            action,
            np.array([0.1, -0.1, 0.05, 0.0, 0.025, -0.025, 0.075, -1.0]),
        )

    def test_evaluate_genome_runs_one_episode(self):
        shape = make_pick_and_place_network_shape(hidden_size=4)
        config = PickAndPlaceEvaluatorConfig(network_shape=shape, max_steps=5)
        genome = np.zeros(shape.genome_length)

        simulator = FakePickAndPlaceSimulator(
            [
                make_state(),
                make_state(
                    object_position=(1.0, 0.0, 0.0),
                    target_position=(1.0, 0.0, 0.0),
                    object_speed=0.0,
                ),
            ]
        )

        result = evaluate_genome(genome, simulator, config)

        self.assertEqual(len(result.actions), 1)
        self.assertEqual(len(result.states), 2)
        np.testing.assert_array_equal(result.actions[0], np.zeros(4))
        self.assertGreater(result.fitness.total, 0.0)

    def test_evaluate_population_gives_each_genome_a_fresh_simulator(self):
        shape = make_pick_and_place_network_shape(hidden_size=4)
        config = PickAndPlaceEvaluatorConfig(network_shape=shape, max_steps=1)
        population = np.zeros((3, shape.genome_length))
        created_simulators = []

        def simulator_factory():
            simulator = FakePickAndPlaceSimulator(
                [
                    make_state(),
                    make_state(),
                ]
            )
            created_simulators.append(simulator)
            return simulator

        results = evaluate_population(population, simulator_factory, config)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(created_simulators), 3)
        self.assertEqual(len(created_simulators[0].actions), 1)
        self.assertEqual(len(created_simulators[1].actions), 1)
        self.assertEqual(len(created_simulators[2].actions), 1)

    def test_config_rejects_wrong_network_input_size(self):
        shape = NetworkShape(input_size=24, hidden_size=4, output_size=4)

        with self.assertRaisesRegex(ValueError, "input_size"):
            PickAndPlaceEvaluatorConfig(network_shape=shape)

    def test_config_rejects_wrong_network_output_size(self):
        shape = NetworkShape(input_size=25, hidden_size=4, output_size=8)

        with self.assertRaisesRegex(ValueError, "output_size"):
            PickAndPlaceEvaluatorConfig(
                network_shape=shape,
                action_mode=END_EFFECTOR_ACTION_MODE,
            )

    def test_observation_rejects_wrong_joint_count(self):
        state = make_state(joint_positions=(0.0, 0.0))

        with self.assertRaisesRegex(ValueError, "joint_positions"):
            build_pick_and_place_observation_vector(state)

    def test_population_rejects_wrong_genome_length(self):
        shape = make_pick_and_place_network_shape(hidden_size=4)
        config = PickAndPlaceEvaluatorConfig(network_shape=shape)
        population = np.zeros((3, shape.genome_length - 1))

        with self.assertRaisesRegex(ValueError, "wrong length"):
            evaluate_population(population, lambda: FakePickAndPlaceSimulator([]), config)


if __name__ == "__main__":
    unittest.main()
