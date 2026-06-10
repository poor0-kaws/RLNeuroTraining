import unittest

import numpy as np

from rl_neuro_training.genome import (
    NetworkShape,
    NeuralNetworkWeights,
    create_random_genome,
    decode_genome,
    encode_genome,
    run_network,
)


class GenomeEncodingTest(unittest.TestCase):
    def test_network_shape_calculates_genome_length(self):
        shape = NetworkShape(input_size=20, hidden_size=32, output_size=7)

        self.assertEqual(shape.input_to_hidden_count, 640)
        self.assertEqual(shape.hidden_bias_count, 32)
        self.assertEqual(shape.hidden_to_output_count, 224)
        self.assertEqual(shape.output_bias_count, 7)
        self.assertEqual(shape.genome_length, 903)

    def test_create_random_genome_has_expected_length(self):
        shape = NetworkShape(input_size=3, hidden_size=4, output_size=2)
        rng = np.random.default_rng(123)

        genome = create_random_genome(shape, rng=rng)

        self.assertEqual(genome.shape, (shape.genome_length,))
        self.assertFalse(np.all(genome == 0.0))

    def test_decode_genome_splits_numbers_into_expected_shapes(self):
        shape = NetworkShape(input_size=2, hidden_size=3, output_size=1)
        genome = np.arange(shape.genome_length, dtype=float)

        weights = decode_genome(genome, shape)

        np.testing.assert_array_equal(
            weights.input_to_hidden,
            np.array(
                [
                    [0.0, 1.0],
                    [2.0, 3.0],
                    [4.0, 5.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            weights.hidden_bias,
            np.array([6.0, 7.0, 8.0]),
        )
        np.testing.assert_array_equal(
            weights.hidden_to_output,
            np.array([[9.0, 10.0, 11.0]]),
        )
        np.testing.assert_array_equal(
            weights.output_bias,
            np.array([12.0]),
        )

    def test_encode_genome_flattens_weights_in_decode_order(self):
        weights = NeuralNetworkWeights(
            input_to_hidden=np.array([[1.0, 2.0], [3.0, 4.0]]),
            hidden_bias=np.array([5.0, 6.0]),
            hidden_to_output=np.array([[7.0, 8.0]]),
            output_bias=np.array([9.0]),
        )

        genome = encode_genome(weights)

        np.testing.assert_array_equal(
            genome,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        )

    def test_zero_weights_return_zero_action(self):
        shape = NetworkShape(input_size=2, hidden_size=3, output_size=1)
        genome = np.zeros(shape.genome_length)
        weights = decode_genome(genome, shape)

        action = run_network(observation=[10.0, -10.0], weights=weights)

        np.testing.assert_array_equal(action, np.array([0.0]))

    def test_action_values_are_squashed_between_minus_one_and_one(self):
        weights = NeuralNetworkWeights(
            input_to_hidden=np.array([[100.0]]),
            hidden_bias=np.array([100.0]),
            hidden_to_output=np.array([[100.0]]),
            output_bias=np.array([100.0]),
        )

        action = run_network(observation=[1.0], weights=weights)

        self.assertGreaterEqual(action[0], -1.0)
        self.assertLessEqual(action[0], 1.0)

    def test_wrong_genome_length_raises_clear_error(self):
        shape = NetworkShape(input_size=2, hidden_size=3, output_size=1)

        with self.assertRaisesRegex(ValueError, "wrong length"):
            decode_genome([1.0, 2.0], shape)


if __name__ == "__main__":
    unittest.main()
