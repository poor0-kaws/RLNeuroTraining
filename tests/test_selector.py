import unittest

import numpy as np

from rl_neuro_training.selector import (
    SelectorConfig,
    select_survivors,
)


class SelectorTest(unittest.TestCase):
    def test_select_survivors_keeps_highest_fitness_genomes(self):
        population = np.array(
            [
                [10.0, 10.0],
                [20.0, 20.0],
                [30.0, 30.0],
                [40.0, 40.0],
            ]
        )
        fitness_scores = np.array([0.20, 0.90, 0.10, 0.70])
        config = SelectorConfig(survivor_count=2)

        result = select_survivors(population, fitness_scores, config)

        np.testing.assert_array_equal(
            result.survivors,
            np.array(
                [
                    [20.0, 20.0],
                    [40.0, 40.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            result.survivor_fitness_scores,
            np.array([0.90, 0.70]),
        )
        np.testing.assert_array_equal(
            result.survivor_indices,
            np.array([1, 3]),
        )

    def test_equal_fitness_keeps_original_order(self):
        population = np.array(
            [
                [10.0],
                [20.0],
                [30.0],
            ]
        )
        fitness_scores = np.array([0.50, 0.50, 0.50])
        config = SelectorConfig(survivor_count=2)

        result = select_survivors(population, fitness_scores, config)

        np.testing.assert_array_equal(result.survivor_indices, np.array([0, 1]))

    def test_survivors_are_copied_from_population(self):
        population = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        fitness_scores = np.array([1.0, 0.0])
        config = SelectorConfig(survivor_count=1)

        result = select_survivors(population, fitness_scores, config)
        population[0, 0] = 999.0

        np.testing.assert_array_equal(result.survivors, np.array([[1.0, 2.0]]))

    def test_invalid_survivor_count_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "survivor_count"):
            SelectorConfig(survivor_count=0)

    def test_survivor_count_cannot_exceed_population_size(self):
        population = np.array(
            [
                [1.0],
                [2.0],
            ]
        )
        fitness_scores = np.array([0.1, 0.2])
        config = SelectorConfig(survivor_count=3)

        with self.assertRaisesRegex(ValueError, "larger than population size"):
            select_survivors(population, fitness_scores, config)

    def test_rejects_population_that_is_not_2d(self):
        population = np.array([1.0, 2.0])
        fitness_scores = np.array([0.1, 0.2])
        config = SelectorConfig(survivor_count=1)

        with self.assertRaisesRegex(ValueError, "2D"):
            select_survivors(population, fitness_scores, config)

    def test_rejects_fitness_scores_that_are_not_1d(self):
        population = np.array(
            [
                [1.0],
                [2.0],
            ]
        )
        fitness_scores = np.array([[0.1], [0.2]])
        config = SelectorConfig(survivor_count=1)

        with self.assertRaisesRegex(ValueError, "1D"):
            select_survivors(population, fitness_scores, config)

    def test_rejects_score_count_that_does_not_match_population(self):
        population = np.array(
            [
                [1.0],
                [2.0],
                [3.0],
            ]
        )
        fitness_scores = np.array([0.1, 0.2])
        config = SelectorConfig(survivor_count=1)

        with self.assertRaisesRegex(ValueError, "one score per genome"):
            select_survivors(population, fitness_scores, config)

    def test_rejects_nan_fitness_scores(self):
        population = np.array(
            [
                [1.0],
                [2.0],
            ]
        )
        fitness_scores = np.array([0.1, np.nan])
        config = SelectorConfig(survivor_count=1)

        with self.assertRaisesRegex(ValueError, "finite"):
            select_survivors(population, fitness_scores, config)


if __name__ == "__main__":
    unittest.main()
