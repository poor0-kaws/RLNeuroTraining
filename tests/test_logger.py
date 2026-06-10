import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_neuro_training.evaluator import GenomeEvaluationResult
from rl_neuro_training.fitness import (
    PickAndPlaceFitnessResult,
    PickAndPlaceStageScores,
)
from rl_neuro_training.logger import (
    GENERATION_STATS_CSV_FIELDS,
    TrainingLogger,
    generation_stats_to_row,
    summarize_generation,
)


def make_evaluation_result(
    total,
    reaching=0.0,
    grasping=0.0,
    lifting=0.0,
    moving=0.0,
    placing=0.0,
    placement_accuracy=0.0,
    placement_stability=0.0,
):
    stages = PickAndPlaceStageScores(
        reaching=reaching,
        grasping=grasping,
        lifting=lifting,
        moving=moving,
        placing=placing,
        placement_accuracy=placement_accuracy,
        placement_stability=placement_stability,
    )
    fitness = PickAndPlaceFitnessResult(total=total, stages=stages)

    return GenomeEvaluationResult(
        fitness=fitness,
        states=(),
        actions=(),
    )


class LoggerTest(unittest.TestCase):
    def test_summarize_generation_calculates_fitness_stats(self):
        results = [
            make_evaluation_result(total=0.20),
            make_evaluation_result(
                total=0.80,
                reaching=1.0,
                grasping=0.9,
                lifting=0.8,
                moving=0.7,
                placing=0.6,
                placement_accuracy=0.5,
                placement_stability=0.4,
            ),
            make_evaluation_result(total=0.40),
        ]

        stats = summarize_generation(
            generation_number=3,
            evaluation_results=results,
        )

        self.assertEqual(stats.generation_number, 3)
        self.assertEqual(stats.population_size, 3)
        self.assertEqual(stats.best_genome_index, 1)
        self.assertEqual(stats.best_fitness, 0.80)
        self.assertEqual(stats.median_fitness, 0.40)
        self.assertEqual(stats.worst_fitness, 0.20)
        self.assertAlmostEqual(stats.average_fitness, np.mean([0.20, 0.80, 0.40]))
        self.assertAlmostEqual(stats.fitness_std, np.std([0.20, 0.80, 0.40]))
        self.assertEqual(stats.best_reaching, 1.0)
        self.assertEqual(stats.best_grasping, 0.9)
        self.assertEqual(stats.best_lifting, 0.8)
        self.assertEqual(stats.best_moving, 0.7)
        self.assertEqual(stats.best_placing, 0.6)
        self.assertEqual(stats.best_placement_accuracy, 0.5)
        self.assertEqual(stats.best_placement_stability, 0.4)

    def test_best_genome_index_uses_first_best_result_when_tied(self):
        results = [
            make_evaluation_result(total=0.90, reaching=0.1),
            make_evaluation_result(total=0.90, reaching=0.9),
            make_evaluation_result(total=0.20),
        ]

        stats = summarize_generation(
            generation_number=0,
            evaluation_results=results,
        )

        self.assertEqual(stats.best_genome_index, 0)
        self.assertEqual(stats.best_reaching, 0.1)

    def test_training_logger_records_history_and_latest(self):
        logger = TrainingLogger()
        first_results = [make_evaluation_result(total=0.10)]
        second_results = [make_evaluation_result(total=0.30)]

        first_stats = logger.record_generation(0, first_results)
        second_stats = logger.record_generation(1, second_results)

        self.assertEqual(logger.history, (first_stats, second_stats))
        self.assertEqual(logger.latest(), second_stats)

    def test_latest_returns_none_before_any_generation_is_recorded(self):
        logger = TrainingLogger()

        self.assertIsNone(logger.latest())

    def test_generation_stats_to_row_uses_csv_column_names(self):
        result = make_evaluation_result(total=0.50, reaching=1.0)
        stats = summarize_generation(2, [result])

        row = generation_stats_to_row(stats)

        self.assertEqual(list(row.keys()), GENERATION_STATS_CSV_FIELDS)
        self.assertEqual(row["generation"], 2)
        self.assertEqual(row["population_size"], 1)
        self.assertEqual(row["best_fitness"], 0.50)
        self.assertEqual(row["best_reaching"], 1.0)

    def test_write_csv_saves_header_and_rows(self):
        logger = TrainingLogger()
        logger.record_generation(0, [make_evaluation_result(total=0.10)])
        logger.record_generation(1, [make_evaluation_result(total=0.30)])

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "training_log.csv"

            logger.write_csv(csv_path)

            with csv_path.open() as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["generation"], "0")
        self.assertEqual(rows[0]["best_fitness"], "0.1")
        self.assertEqual(rows[1]["generation"], "1")
        self.assertEqual(rows[1]["best_fitness"], "0.3")

    def test_empty_generation_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            summarize_generation(0, [])

    def test_negative_generation_number_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "generation_number"):
            summarize_generation(-1, [make_evaluation_result(total=0.10)])

    def test_nan_fitness_raises_clear_error(self):
        results = [make_evaluation_result(total=np.nan)]

        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_generation(0, results)


if __name__ == "__main__":
    unittest.main()
