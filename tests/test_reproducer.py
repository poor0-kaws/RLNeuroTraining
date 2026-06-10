import unittest

import numpy as np

from rl_neuro_training.reproducer import (
    ReproducerConfig,
    create_next_generation,
    mutate_genome,
)


class ReproducerTest(unittest.TestCase):
    def test_create_next_generation_keeps_elites_unchanged(self):
        survivors = np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )
        config = ReproducerConfig(
            population_size=6,
            elite_count=2,
            mutation_rate=1.0,
            mutation_strength=0.50,
            seed=123,
        )

        result = create_next_generation(survivors, config)

        np.testing.assert_array_equal(
            result.elite_generation,
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            result.next_population[:2],
            result.elite_generation,
        )

    def test_create_next_generation_fills_population_with_mutated_children(self):
        survivors = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        config = ReproducerConfig(
            population_size=5,
            elite_count=1,
            mutation_rate=1.0,
            mutation_strength=0.50,
            seed=123,
        )

        result = create_next_generation(survivors, config)

        self.assertEqual(result.next_population.shape, (5, 3))
        self.assertEqual(result.elite_generation.shape, (1, 3))
        self.assertEqual(result.mutated_generation.shape, (4, 3))
        self.assertFalse(np.all(result.mutated_generation == 0.0))

    def test_same_seed_creates_same_next_generation(self):
        survivors = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )
        config = ReproducerConfig(
            population_size=5,
            elite_count=1,
            seed=123,
        )

        first_result = create_next_generation(survivors, config)
        second_result = create_next_generation(survivors, config)

        np.testing.assert_array_equal(
            first_result.next_population,
            second_result.next_population,
        )

    def test_mutate_genome_keeps_values_inside_gene_limits(self):
        config = ReproducerConfig(
            population_size=2,
            elite_count=1,
            mutation_rate=1.0,
            mutation_strength=100.0,
            min_gene_value=-1.0,
            max_gene_value=1.0,
            seed=123,
        )

        child = mutate_genome([0.0, 0.0, 0.0], config)

        self.assertTrue(np.all(child >= -1.0))
        self.assertTrue(np.all(child <= 1.0))

    def test_zero_mutation_rate_copies_parent_for_mutated_child(self):
        survivors = np.array(
            [
                [1.0, 2.0, 3.0],
            ]
        )
        config = ReproducerConfig(
            population_size=3,
            elite_count=1,
            mutation_rate=0.0,
            mutation_strength=0.50,
            seed=123,
        )

        result = create_next_generation(survivors, config)

        np.testing.assert_array_equal(
            result.next_population,
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0],
                ]
            ),
        )

    def test_invalid_population_size_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "population_size"):
            ReproducerConfig(population_size=0)

    def test_invalid_mutation_rate_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "mutation_rate"):
            ReproducerConfig(
                population_size=5,
                mutation_rate=1.5,
            )

    def test_elite_count_cannot_exceed_survivor_count(self):
        survivors = np.array(
            [
                [1.0, 2.0],
            ]
        )
        config = ReproducerConfig(
            population_size=5,
            elite_count=2,
        )

        with self.assertRaisesRegex(ValueError, "survivor count"):
            create_next_generation(survivors, config)

    def test_rejects_empty_survivors(self):
        survivors = np.empty((0, 3))
        config = ReproducerConfig(
            population_size=5,
            elite_count=0,
        )

        with self.assertRaisesRegex(ValueError, "at least one"):
            create_next_generation(survivors, config)

    def test_rejects_nan_survivor_values(self):
        survivors = np.array(
            [
                [1.0, np.nan],
                [2.0, 3.0],
            ]
        )
        config = ReproducerConfig(
            population_size=5,
            elite_count=1,
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            create_next_generation(survivors, config)


if __name__ == "__main__":
    unittest.main()
