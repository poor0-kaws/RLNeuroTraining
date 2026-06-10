import unittest

import numpy as np

from rl_neuro_training.population import (
    PopulationInitializerConfig,
    create_initial_population,
    validate_population,
)


class PopulationInitializerTest(unittest.TestCase):
    def test_initial_population_has_expected_shape(self):
        config = PopulationInitializerConfig(
            population_size=50,
            genome_length=12,
        )

        population = create_initial_population(config)

        self.assertEqual(population.shape, (50, 12))

    def test_initial_population_uses_configured_value_range(self):
        config = PopulationInitializerConfig(
            population_size=20,
            genome_length=8,
            min_gene_value=-1.0,
            max_gene_value=1.0,
            seed=123,
        )

        population = create_initial_population(config)

        self.assertTrue(np.all(population >= -1.0))
        self.assertTrue(np.all(population <= 1.0))

    def test_same_seed_creates_same_population(self):
        config = PopulationInitializerConfig(
            population_size=5,
            genome_length=4,
            seed=123,
        )

        first_population = create_initial_population(config)
        second_population = create_initial_population(config)

        np.testing.assert_array_equal(first_population, second_population)

    def test_different_seed_creates_different_population(self):
        first_config = PopulationInitializerConfig(
            population_size=5,
            genome_length=4,
            seed=123,
        )
        second_config = PopulationInitializerConfig(
            population_size=5,
            genome_length=4,
            seed=456,
        )

        first_population = create_initial_population(first_config)
        second_population = create_initial_population(second_config)

        self.assertFalse(np.array_equal(first_population, second_population))

    def test_invalid_population_size_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "population_size"):
            PopulationInitializerConfig(
                population_size=0,
                genome_length=4,
            )

    def test_invalid_genome_length_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "genome_length"):
            PopulationInitializerConfig(
                population_size=5,
                genome_length=0,
            )

    def test_invalid_value_range_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "less than"):
            PopulationInitializerConfig(
                population_size=5,
                genome_length=4,
                min_gene_value=1.0,
                max_gene_value=1.0,
            )

    def test_validate_population_rejects_wrong_shape(self):
        config = PopulationInitializerConfig(
            population_size=5,
            genome_length=4,
        )
        population = np.zeros((5, 3))

        with self.assertRaisesRegex(ValueError, "wrong shape"):
            validate_population(population, config)

    def test_validate_population_rejects_nan_values(self):
        config = PopulationInitializerConfig(
            population_size=2,
            genome_length=2,
        )
        population = np.array(
            [
                [0.0, np.nan],
                [0.5, -0.5],
            ]
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            validate_population(population, config)

    def test_validate_population_rejects_values_outside_range(self):
        config = PopulationInitializerConfig(
            population_size=2,
            genome_length=2,
            min_gene_value=-1.0,
            max_gene_value=1.0,
        )
        population = np.array(
            [
                [0.0, 1.1],
                [0.5, -0.5],
            ]
        )

        with self.assertRaisesRegex(ValueError, "above"):
            validate_population(population, config)


if __name__ == "__main__":
    unittest.main()
