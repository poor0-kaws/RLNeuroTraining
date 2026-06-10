import unittest
from unittest.mock import patch

import numpy as np

from rl_neuro_training.evaluator import (
    PickAndPlaceEvaluatorConfig,
    make_pick_and_place_network_shape,
)
from rl_neuro_training.fitness import (
    PickAndPlaceFitnessResult,
    PickAndPlaceStageScores,
)
from rl_neuro_training.evaluator import GenomeEvaluationResult
from rl_neuro_training.population import PopulationInitializerConfig
from rl_neuro_training.reproducer import ReproducerConfig
from rl_neuro_training.selector import SelectorConfig
from rl_neuro_training.trainer import (
    TrainerConfig,
    fitness_scores_from_results,
    train,
)


def make_evaluation_result(total):
    stages = PickAndPlaceStageScores(
        reaching=total,
        grasping=0.0,
        lifting=0.0,
        moving=0.0,
        placing=0.0,
        placement_accuracy=0.0,
        placement_stability=0.0,
    )

    return GenomeEvaluationResult(
        fitness=PickAndPlaceFitnessResult(total=total, stages=stages),
        states=(),
        actions=(),
    )


def make_config(population_size=4, generation_count=2):
    network_shape = make_pick_and_place_network_shape(hidden_size=2)

    return TrainerConfig(
        generation_count=generation_count,
        population_config=PopulationInitializerConfig(
            population_size=population_size,
            genome_length=network_shape.genome_length,
            seed=123,
        ),
        evaluator_config=PickAndPlaceEvaluatorConfig(
            network_shape=network_shape,
            max_steps=1,
        ),
        selector_config=SelectorConfig(survivor_count=2),
        reproducer_config=ReproducerConfig(
            population_size=population_size,
            elite_count=1,
            mutation_rate=0.0,
            mutation_strength=0.0,
            seed=123,
        ),
    )


class TrainerTest(unittest.TestCase):
    def test_train_evaluates_logs_selects_and_reproduces(self):
        config = make_config()
        genome_length = config.population_config.genome_length
        initial_population = np.array(
            [
                np.full(genome_length, 0.1),
                np.full(genome_length, 0.2),
                np.full(genome_length, 0.3),
                np.full(genome_length, 0.4),
            ]
        )
        first_generation_results = [
            make_evaluation_result(0.20),
            make_evaluation_result(0.90),
            make_evaluation_result(0.10),
            make_evaluation_result(0.70),
        ]
        second_generation_results = [
            make_evaluation_result(0.30),
            make_evaluation_result(0.40),
            make_evaluation_result(0.50),
            make_evaluation_result(0.60),
        ]

        with patch(
            "rl_neuro_training.trainer.evaluate_population",
            side_effect=[first_generation_results, second_generation_results],
        ) as evaluate_population_mock:
            result = train(
                simulator_factory=lambda: None,
                config=config,
                initial_population=initial_population,
            )

        self.assertEqual(evaluate_population_mock.call_count, 2)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.history[0].best_fitness, 0.90)
        self.assertEqual(result.history[1].best_fitness, 0.60)
        self.assertEqual(result.champion.generation_number, 0)
        self.assertEqual(result.champion.genome_index, 1)
        self.assertEqual(result.champion.fitness, 0.90)
        np.testing.assert_array_equal(
            result.champion.genome,
            initial_population[1],
        )
        np.testing.assert_array_equal(
            result.final_population[0],
            initial_population[1],
        )
        self.assertEqual(result.final_population.shape, initial_population.shape)

    def test_train_can_create_initial_population_from_config(self):
        config = make_config(population_size=3, generation_count=1)
        evaluation_results = [
            make_evaluation_result(0.10),
            make_evaluation_result(0.20),
            make_evaluation_result(0.30),
        ]

        with patch(
            "rl_neuro_training.trainer.evaluate_population",
            return_value=evaluation_results,
        ) as evaluate_population_mock:
            result = train(
                simulator_factory=lambda: None,
                config=config,
            )

        first_call_population = evaluate_population_mock.call_args.kwargs[
            "population"
        ]

        self.assertEqual(first_call_population.shape[0], 3)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.champion.fitness, 0.30)

    def test_fitness_scores_from_results_returns_one_score_per_result(self):
        results = [
            make_evaluation_result(0.10),
            make_evaluation_result(0.80),
        ]

        scores = fitness_scores_from_results(results)

        np.testing.assert_array_equal(scores, np.array([0.10, 0.80]))

    def test_config_rejects_mismatched_genome_length(self):
        network_shape = make_pick_and_place_network_shape(hidden_size=2)

        with self.assertRaisesRegex(ValueError, "genome_length"):
            TrainerConfig(
                generation_count=1,
                population_config=PopulationInitializerConfig(
                    population_size=4,
                    genome_length=network_shape.genome_length - 1,
                ),
                evaluator_config=PickAndPlaceEvaluatorConfig(
                    network_shape=network_shape,
                ),
                selector_config=SelectorConfig(survivor_count=2),
                reproducer_config=ReproducerConfig(population_size=4),
            )

    def test_config_rejects_elite_count_larger_than_survivor_count(self):
        network_shape = make_pick_and_place_network_shape(hidden_size=2)

        with self.assertRaisesRegex(ValueError, "elite_count"):
            TrainerConfig(
                generation_count=1,
                population_config=PopulationInitializerConfig(
                    population_size=4,
                    genome_length=network_shape.genome_length,
                ),
                evaluator_config=PickAndPlaceEvaluatorConfig(
                    network_shape=network_shape,
                ),
                selector_config=SelectorConfig(survivor_count=1),
                reproducer_config=ReproducerConfig(
                    population_size=4,
                    elite_count=2,
                ),
            )

    def test_empty_evaluation_results_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            fitness_scores_from_results([])


if __name__ == "__main__":
    unittest.main()
