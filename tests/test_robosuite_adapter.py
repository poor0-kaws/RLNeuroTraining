import unittest

import numpy as np

from rl_neuro_training.robosuite_adapter import (
    RobosuitePickAndPlaceConfig,
    RobosuitePickAndPlaceSimulator,
)


def make_observation(
    gripper_position=(0.0, 0.0, 0.0),
    object_position=(0.02, 0.0, 0.0),
    target_position=(1.0, 0.0, 0.0),
    joint_positions=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    gripper_qpos=(0.0, 0.0),
):
    return {
        "robot0_eef_pos": np.array(gripper_position, dtype=float),
        "cube_pos": np.array(object_position, dtype=float),
        "target_pos": np.array(target_position, dtype=float),
        "robot0_joint_pos": np.array(joint_positions, dtype=float),
        "robot0_gripper_qpos": np.array(gripper_qpos, dtype=float),
    }


class FakeRobosuiteEnv:
    action_dim = 7
    control_timestep = 0.5

    def __init__(self, reset_observation, step_observation=None, step_result_size=4):
        self.reset_observation = reset_observation
        self.step_observation = step_observation or reset_observation
        self.step_result_size = step_result_size
        self.actions = []
        self.render_count = 0
        self.close_count = 0

    def reset(self):
        return self.reset_observation

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float))

        if self.step_result_size == 5:
            return self.step_observation, 0.0, False, True, {}

        return self.step_observation, 0.0, True, {}

    def render(self):
        self.render_count += 1

    def close(self):
        self.close_count += 1

    def action_spec(self):
        return -np.ones(self.action_dim), np.ones(self.action_dim)


class RobosuiteAdapterTest(unittest.TestCase):
    def test_reset_converts_robosuite_observation_to_pick_and_place_state(self):
        env = FakeRobosuiteEnv(
            reset_observation=make_observation(
                gripper_position=(0.1, 0.2, 0.3),
                object_position=(0.1, 0.2, 0.31),
                target_position=(0.8, 0.2, 0.0),
                joint_positions=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                gripper_qpos=(0.0, 0.0),
            )
        )
        simulator = RobosuitePickAndPlaceSimulator(env)

        state = simulator.reset()

        self.assertEqual(state.gripper_position, (0.1, 0.2, 0.3))
        self.assertEqual(state.object_position, (0.1, 0.2, 0.31))
        self.assertEqual(state.target_position, (0.8, 0.2, 0.0))
        self.assertEqual(state.joint_positions, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual(state.gripper_open_amount, 0.0)
        self.assertTrue(state.is_grasped)
        self.assertIsNone(state.object_speed)
        self.assertEqual(env.render_count, 1)

    def test_step_maps_four_number_action_to_seven_number_robosuite_action(self):
        env = FakeRobosuiteEnv(
            reset_observation=make_observation(),
            step_observation=make_observation(
                object_position=(0.12, 0.0, 0.0),
            ),
        )
        simulator = RobosuitePickAndPlaceSimulator(env)
        simulator.reset()

        result = simulator.step([0.2, -0.1, 0.3, -1.0])

        np.testing.assert_array_equal(
            env.actions[0],
            np.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, -1.0]),
        )
        self.assertTrue(result.done)
        self.assertAlmostEqual(result.state.object_speed, 0.2)

    def test_step_supports_gymnasium_five_value_step_result(self):
        env = FakeRobosuiteEnv(
            reset_observation=make_observation(),
            step_observation=make_observation(),
            step_result_size=5,
        )
        simulator = RobosuitePickAndPlaceSimulator(env)
        simulator.reset()

        result = simulator.step([0.0, 0.0, 0.0, 0.0])

        self.assertTrue(result.done)

    def test_fixed_target_position_can_replace_observation_target_key(self):
        observation = make_observation()
        del observation["target_pos"]
        config = RobosuitePickAndPlaceConfig(
            target_position=(0.4, 0.5, 0.6),
            render_each_step=False,
        )
        env = FakeRobosuiteEnv(reset_observation=observation)
        simulator = RobosuitePickAndPlaceSimulator(env, config)

        state = simulator.reset()

        self.assertEqual(state.target_position, (0.4, 0.5, 0.6))
        self.assertEqual(env.render_count, 0)

    def test_observation_is_grasped_key_overrides_distance_guess(self):
        observation = make_observation(
            gripper_position=(0.0, 0.0, 0.0),
            object_position=(1.0, 1.0, 1.0),
        )
        observation["is_grasped"] = True
        config = RobosuitePickAndPlaceConfig(is_grasped_key="is_grasped")
        env = FakeRobosuiteEnv(reset_observation=observation)
        simulator = RobosuitePickAndPlaceSimulator(env, config)

        state = simulator.reset()

        self.assertTrue(state.is_grasped)

    def test_action_is_clipped_to_env_action_bounds(self):
        env = FakeRobosuiteEnv(reset_observation=make_observation())
        simulator = RobosuitePickAndPlaceSimulator(env)
        simulator.reset()

        simulator.step([2.0, -2.0, 0.0, -3.0])

        np.testing.assert_array_equal(
            env.actions[0],
            np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        )

    def test_missing_required_observation_key_raises_clear_error(self):
        observation = make_observation()
        del observation["robot0_eef_pos"]
        env = FakeRobosuiteEnv(reset_observation=observation)
        simulator = RobosuitePickAndPlaceSimulator(env)

        with self.assertRaisesRegex(ValueError, "robot0_eef_pos"):
            simulator.reset()

    def test_close_calls_env_close_when_available(self):
        env = FakeRobosuiteEnv(reset_observation=make_observation())
        simulator = RobosuitePickAndPlaceSimulator(env)

        simulator.close()

        self.assertEqual(env.close_count, 1)


if __name__ == "__main__":
    unittest.main()
