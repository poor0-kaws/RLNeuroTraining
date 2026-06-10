import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_neuro_training.fitness import PickAndPlaceStageScores
from rl_neuro_training.logger import GenerationStats, TrainingLogger
from rl_neuro_training.trainer import ChampionGenomeRecord, TrainingResult
from scripts.train_pick_place import (
    build_robosuite_config,
    build_trainer_config,
    champion_metadata,
    choose_survivor_count,
    load_initial_population,
    parse_args,
    progress_records_from_generation_stats,
    run_config_metadata,
    save_training_artifacts,
)


def make_generation_stats(
    generation_number,
    best_fitness,
    average_fitness,
    worst_fitness,
):
    return GenerationStats(
        generation_number=generation_number,
        population_size=4,
        best_fitness=best_fitness,
        average_fitness=average_fitness,
        median_fitness=average_fitness,
        worst_fitness=worst_fitness,
        fitness_std=0.0,
        best_genome_index=1,
        worst_genome_index=2,
        best_reaching=1.0,
        best_grasping=0.8,
        best_lifting=0.6,
        best_moving=0.4,
        best_placing=0.2,
        best_placement_accuracy=0.2,
        best_placement_stability=0.1,
        average_reaching=0.5,
        average_grasping=0.4,
        average_lifting=0.3,
        average_moving=0.2,
        average_placing=0.1,
        average_placement_accuracy=0.1,
        average_placement_stability=0.05,
    )


def make_training_result():
    stages = PickAndPlaceStageScores(
        reaching=1.0,
        grasping=0.8,
        lifting=0.6,
        moving=0.4,
        placing=0.2,
        placement_accuracy=0.2,
        placement_stability=0.1,
    )

    return TrainingResult(
        final_population=np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        ),
        history=(
            make_generation_stats(0, 0.6, 0.4, 0.2),
            make_generation_stats(1, 0.8, 0.5, 0.1),
        ),
        champion=ChampionGenomeRecord(
            generation_number=1,
            genome_index=0,
            fitness=0.8,
            stages=stages,
            genome=np.array([0.3, 0.4]),
        ),
        generation_fitness_scores=(
            np.array([0.2, 0.6]),
            np.array([0.8, 0.1]),
        ),
    )


class TrainPickPlaceScriptTest(unittest.TestCase):
    def test_choose_survivor_count_uses_top_twenty_percent_by_default(self):
        survivor_count = choose_survivor_count(
            population_size=50,
            survivor_fraction=0.20,
            survivor_count=None,
            elite_count=2,
        )

        self.assertEqual(survivor_count, 10)

    def test_choose_survivor_count_never_drops_below_elite_count(self):
        survivor_count = choose_survivor_count(
            population_size=5,
            survivor_fraction=0.20,
            survivor_count=None,
            elite_count=2,
        )

        self.assertEqual(survivor_count, 2)

    def test_build_trainer_config_matches_requested_sizes(self):
        args = parse_args(
            [
                "--generations",
                "2",
                "--population-size",
                "6",
                "--hidden-size",
                "4",
                "--max-steps",
                "3",
                "--elite-count",
                "1",
            ]
        )

        config = build_trainer_config(args)

        self.assertEqual(config.generation_count, 2)
        self.assertEqual(config.population_config.population_size, 6)
        self.assertEqual(config.evaluator_config.max_steps, 3)
        self.assertEqual(config.reproducer_config.elite_count, 1)
        self.assertEqual(
            config.population_config.genome_length,
            config.evaluator_config.network_shape.genome_length,
        )

    def test_build_robosuite_config_includes_optional_env_seed(self):
        args = parse_args(
            [
                "--max-steps",
                "9",
                "--object-type",
                "milk",
                "--env-seed",
                "123",
            ]
        )

        config = build_robosuite_config(args)

        self.assertEqual(config.env_kwargs["horizon"], 9)
        self.assertEqual(config.env_kwargs["object_type"], "milk")
        self.assertEqual(config.env_kwargs["seed"], 123)

    def test_load_initial_population_reads_2d_numpy_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            population_path = Path(temp_dir) / "final_population.npy"
            expected = np.array([[0.1, 0.2], [0.3, 0.4]])
            np.save(population_path, expected)

            population = load_initial_population(population_path)

        np.testing.assert_array_equal(population, expected)

    def test_load_initial_population_rejects_non_2d_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            population_path = Path(temp_dir) / "bad_population.npy"
            np.save(population_path, np.array([0.1, 0.2]))

            with self.assertRaisesRegex(ValueError, "2D"):
                load_initial_population(population_path)

    def test_run_config_metadata_contains_replay_settings(self):
        args = parse_args(
            [
                "--hidden-size",
                "4",
                "--max-steps",
                "7",
                "--action-mode",
                "joint",
                "--object-type",
                "milk",
                "--env-seed",
                "123",
            ]
        )
        trainer_config = build_trainer_config(args)

        metadata = run_config_metadata(args, trainer_config)

        self.assertEqual(metadata["replay"]["hidden_size"], 4)
        self.assertEqual(metadata["replay"]["max_steps"], 7)
        self.assertEqual(metadata["replay"]["action_mode"], "joint")
        self.assertEqual(metadata["replay"]["object_type"], "milk")
        self.assertEqual(metadata["replay"]["env_seed"], 123)
        self.assertEqual(
            metadata["network_shape"]["genome_length"],
            trainer_config.evaluator_config.network_shape.genome_length,
        )

    def test_progress_records_from_generation_stats_tracks_running_values(self):
        history = (
            make_generation_stats(0, 0.6, 0.4, 0.2),
            make_generation_stats(1, 0.5, 0.6, 0.1),
        )

        records = progress_records_from_generation_stats(history)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].all_time_best_fitness, 0.6)
        self.assertEqual(records[1].all_time_best_fitness, 0.6)
        self.assertEqual(records[1].all_time_worst_fitness, 0.1)
        self.assertEqual(records[1].running_average_fitness, 0.5)
        self.assertEqual(records[1].stage_averages.reaching, 0.5)

    def test_champion_metadata_is_json_safe(self):
        result = make_training_result()

        metadata = champion_metadata(result)

        self.assertEqual(metadata["generation_number"], 1)
        self.assertEqual(metadata["genome_index"], 0)
        self.assertEqual(metadata["fitness"], 0.8)
        self.assertEqual(metadata["genome_length"], 2)
        self.assertEqual(metadata["stages"]["reaching"], 1.0)

    def test_save_training_artifacts_writes_expected_files(self):
        result = make_training_result()
        logger = TrainingLogger()
        logger._history.extend(result.history)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_training_artifacts(
                result=result,
                logger=logger,
                output_dir=Path(temp_dir),
            )

            self.assertTrue(paths.csv_log.exists())
            self.assertTrue(paths.champion_genome.exists())
            self.assertTrue(paths.champion_metadata.exists())
            self.assertTrue(paths.final_population.exists())
            self.assertTrue(paths.generation_fitness_scores.exists())
            self.assertTrue(paths.fitness_progress_svg.exists())
            self.assertTrue(paths.run_config.exists())
            self.assertIn(
                "<svg",
                paths.fitness_progress_svg.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                json.loads(paths.run_config.read_text(encoding="utf-8")),
                {"format_version": 1},
            )


if __name__ == "__main__":
    unittest.main()
