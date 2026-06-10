import unittest

from rl_neuro_training.fitness import (
    PickAndPlaceObservation,
    score_pick_and_place_episode,
)


class PickAndPlaceFitnessTest(unittest.TestCase):
    def test_empty_episode_scores_zero(self):
        result = score_pick_and_place_episode([])

        self.assertEqual(result.total, 0.0)

    def test_reaching_only_gets_small_partial_credit(self):
        observation = PickAndPlaceObservation(
            gripper_position=(0.01, 0.0, 0.0),
            object_position=(0.0, 0.0, 0.0),
            target_position=(1.0, 0.0, 0.0),
            is_grasped=False,
            table_height=0.0,
        )

        result = score_pick_and_place_episode([observation])

        self.assertGreater(result.stages.reaching, 0.0)
        self.assertEqual(result.stages.grasping, 0.0)
        self.assertLess(result.total, 0.05)

    def test_grasping_and_lifting_are_gated_by_real_grasp(self):
        observation = PickAndPlaceObservation(
            gripper_position=(0.0, 0.0, 0.2),
            object_position=(0.0, 0.0, 0.2),
            target_position=(1.0, 0.0, 0.0),
            is_grasped=False,
            table_height=0.0,
        )

        result = score_pick_and_place_episode([observation])

        self.assertEqual(result.stages.grasping, 0.0)
        self.assertEqual(result.stages.lifting, 0.0)

    def test_complete_stable_pick_and_place_scores_high(self):
        episode = [
            PickAndPlaceObservation(
                gripper_position=(0.50, 0.0, 0.0),
                object_position=(0.0, 0.0, 0.0),
                target_position=(1.0, 0.0, 0.0),
                is_grasped=False,
                table_height=0.0,
            ),
            PickAndPlaceObservation(
                gripper_position=(0.0, 0.0, 0.0),
                object_position=(0.0, 0.0, 0.0),
                target_position=(1.0, 0.0, 0.0),
                is_grasped=True,
                table_height=0.0,
            ),
            PickAndPlaceObservation(
                gripper_position=(0.0, 0.0, 0.10),
                object_position=(0.0, 0.0, 0.10),
                target_position=(1.0, 0.0, 0.0),
                is_grasped=True,
                table_height=0.0,
            ),
            PickAndPlaceObservation(
                gripper_position=(1.0, 0.0, 0.10),
                object_position=(1.0, 0.0, 0.10),
                target_position=(1.0, 0.0, 0.0),
                is_grasped=True,
                table_height=0.0,
            ),
        ]

        for _ in range(10):
            episode.append(
                PickAndPlaceObservation(
                    gripper_position=(1.0, 0.0, 0.10),
                    object_position=(1.0, 0.0, 0.0),
                    target_position=(1.0, 0.0, 0.0),
                    is_grasped=False,
                    table_height=0.0,
                    object_speed=0.0,
                )
            )

        result = score_pick_and_place_episode(episode)

        self.assertGreater(result.total, 0.95)
        self.assertGreater(result.stages.placement_stability, 0.90)

    def test_unstable_placement_scores_less_than_stable_placement(self):
        unstable_episode = [
            PickAndPlaceObservation(
                gripper_position=(0.0, 0.0, 0.0),
                object_position=(0.0, 0.0, 0.0),
                target_position=(0.0, 0.0, 0.0),
                is_grasped=False,
                table_height=0.0,
                object_speed=1.0,
            )
        ]

        stable_episode = [
            PickAndPlaceObservation(
                gripper_position=(0.0, 0.0, 0.0),
                object_position=(0.0, 0.0, 0.0),
                target_position=(0.0, 0.0, 0.0),
                is_grasped=False,
                table_height=0.0,
                object_speed=0.0,
            )
            for _ in range(10)
        ]

        unstable_result = score_pick_and_place_episode(unstable_episode)
        stable_result = score_pick_and_place_episode(stable_episode)

        self.assertGreater(stable_result.total, unstable_result.total)
        self.assertEqual(unstable_result.stages.placement_stability, 0.0)
        self.assertEqual(stable_result.stages.placement_stability, 1.0)

    def test_earlier_stable_placement_scores_better_than_later_stability(self):
        early_stable_episode = [
            PickAndPlaceObservation(
                gripper_position=(0.0, 0.0, 0.0),
                object_position=(0.0, 0.0, 0.0),
                target_position=(0.0, 0.0, 0.0),
                is_grasped=False,
                table_height=0.0,
                object_speed=0.0,
            )
            for _ in range(20)
        ]

        late_stable_episode = [
            PickAndPlaceObservation(
                gripper_position=(1.0, 0.0, 0.0),
                object_position=(1.0, 0.0, 0.0),
                target_position=(0.0, 0.0, 0.0),
                is_grasped=False,
                table_height=0.0,
                object_speed=1.0,
            )
            for _ in range(10)
        ]

        for _ in range(10):
            late_stable_episode.append(
                PickAndPlaceObservation(
                    gripper_position=(0.0, 0.0, 0.0),
                    object_position=(0.0, 0.0, 0.0),
                    target_position=(0.0, 0.0, 0.0),
                    is_grasped=False,
                    table_height=0.0,
                    object_speed=0.0,
                )
            )

        early_result = score_pick_and_place_episode(early_stable_episode)
        late_result = score_pick_and_place_episode(late_stable_episode)

        self.assertGreater(
            early_result.stages.placement_stability,
            late_result.stages.placement_stability,
        )


if __name__ == "__main__":
    unittest.main()
