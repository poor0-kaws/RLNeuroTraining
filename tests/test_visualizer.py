import unittest

from rl_neuro_training.fitness import (
    PickAndPlaceFitnessResult,
    PickAndPlaceStageScores,
)
from rl_neuro_training.visualizer import (
    FitnessProgressGraphConfig,
    render_fitness_progress_svg,
    strongest_average_stage,
    summarize_generation,
    summarize_training,
)


def make_fitness_result(
    total,
    reaching=0.0,
    grasping=0.0,
    lifting=0.0,
    moving=0.0,
    placing=0.0,
):
    return PickAndPlaceFitnessResult(
        total=total,
        stages=PickAndPlaceStageScores(
            reaching=reaching,
            grasping=grasping,
            lifting=lifting,
            moving=moving,
            placing=placing,
            placement_accuracy=placing,
            placement_stability=placing,
        ),
    )


class VisualizerTest(unittest.TestCase):
    def test_summarize_generation_tracks_best_average_and_worst(self):
        fitness_results = [
            make_fitness_result(0.20),
            make_fitness_result(0.90),
            make_fitness_result(0.10),
            make_fitness_result(0.70),
        ]

        record = summarize_generation(
            generation_index=0,
            fitness_results=fitness_results,
        )

        self.assertEqual(record.best_fitness, 0.90)
        self.assertAlmostEqual(record.average_fitness, 0.475)
        self.assertEqual(record.worst_fitness, 0.10)
        self.assertEqual(record.best_robot_index, 1)
        self.assertEqual(record.worst_robot_index, 2)

    def test_summarize_generation_tracks_running_and_all_time_values(self):
        first_record = summarize_generation(
            generation_index=0,
            fitness_results=[
                make_fitness_result(0.20),
                make_fitness_result(0.40),
            ],
        )
        second_record = summarize_generation(
            generation_index=1,
            fitness_results=[
                make_fitness_result(0.10),
                make_fitness_result(0.30),
            ],
            previous_records=[first_record],
        )

        self.assertEqual(second_record.average_fitness, 0.20)
        self.assertEqual(second_record.running_average_fitness, 0.25)
        self.assertEqual(second_record.all_time_best_fitness, 0.40)
        self.assertEqual(second_record.all_time_worst_fitness, 0.10)

    def test_summarize_generation_tracks_average_behavior_stages(self):
        fitness_results = [
            make_fitness_result(
                total=0.30,
                reaching=1.0,
                grasping=0.5,
                lifting=0.0,
                moving=0.0,
                placing=0.0,
            ),
            make_fitness_result(
                total=0.50,
                reaching=0.5,
                grasping=0.5,
                lifting=1.0,
                moving=0.0,
                placing=0.0,
            ),
        ]

        record = summarize_generation(
            generation_index=0,
            fitness_results=fitness_results,
        )

        self.assertEqual(record.stage_averages.reaching, 0.75)
        self.assertEqual(record.stage_averages.grasping, 0.50)
        self.assertEqual(record.stage_averages.lifting, 0.50)
        self.assertEqual(record.stage_averages.moving, 0.00)
        self.assertEqual(record.stage_averages.placing, 0.00)
        self.assertEqual(strongest_average_stage(record), "reaching")

    def test_summarize_training_tracks_champion_and_weakest_robot(self):
        generations = [
            [
                make_fitness_result(0.20),
                make_fitness_result(0.40),
            ],
            [
                make_fitness_result(0.10),
                make_fitness_result(0.80),
            ],
        ]

        summary = summarize_training(generations)

        self.assertEqual(len(summary.records), 2)
        self.assertEqual(summary.champion.generation_index, 1)
        self.assertEqual(summary.champion.robot_index, 1)
        self.assertEqual(summary.champion.fitness, 0.80)
        self.assertEqual(summary.weakest_robot.generation_index, 1)
        self.assertEqual(summary.weakest_robot.robot_index, 0)
        self.assertEqual(summary.weakest_robot.fitness, 0.10)

    def test_render_fitness_progress_svg_contains_expected_graph_parts(self):
        summary = summarize_training(
            [
                [
                    make_fitness_result(0.20),
                    make_fitness_result(0.40),
                ],
                [
                    make_fitness_result(0.10),
                    make_fitness_result(0.80),
                ],
            ]
        )
        config = FitnessProgressGraphConfig(width=500, height=320)

        svg = render_fitness_progress_svg(summary.records, config)

        self.assertIn("<svg", svg)
        self.assertIn("Training Fitness Progress", svg)
        self.assertIn("best fitness", svg)
        self.assertIn("average fitness", svg)
        self.assertIn("worst fitness", svg)
        self.assertIn("all-time best fitness", svg)
        self.assertIn("fitness range", svg)
        self.assertIn("</svg>", svg)

    def test_empty_generation_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_generation(
                generation_index=0,
                fitness_results=[],
            )

    def test_empty_graph_records_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            render_fitness_progress_svg([])


if __name__ == "__main__":
    unittest.main()
