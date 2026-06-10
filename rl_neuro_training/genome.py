"""Genome encoding for a simple neural network controller.

In this project, a genome is one flat NumPy array.

The genome is flat because evolution works best with a simple list of numbers:

    copy this list
    mutate some numbers
    compare the resulting robot's fitness

The robot controller still needs normal neural network shapes, so this file
also knows how to decode the flat list back into weight matrices and biases.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True, kw_only=True)
class NetworkShape:
    """The size of the fixed neural network.

    input_size:
        How many observation numbers the robot sees.

    output_size:
        How many action numbers the robot sends to the arm.

    hidden_size:
        How many neurons sit between inputs and outputs.

    For version one, we use one hidden layer because it is simple but still more
    powerful than a direct input-to-output controller.
    """

    input_size: int
    output_size: int
    hidden_size: int = 32

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be greater than zero")

        if self.output_size <= 0:
            raise ValueError("output_size must be greater than zero")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")

    @property
    def input_to_hidden_count(self) -> int:
        return self.input_size * self.hidden_size

    @property
    def hidden_bias_count(self) -> int:
        return self.hidden_size

    @property
    def hidden_to_output_count(self) -> int:
        return self.hidden_size * self.output_size

    @property
    def output_bias_count(self) -> int:
        return self.output_size

    @property
    def genome_length(self) -> int:
        return (
            self.input_to_hidden_count
            + self.hidden_bias_count
            + self.hidden_to_output_count
            + self.output_bias_count
        )


@dataclass(frozen=True)
class NeuralNetworkWeights:
    """Decoded neural network weights.

    These are the same numbers as the genome, just reshaped into the forms that
    neural network math expects.
    """

    input_to_hidden: np.ndarray
    hidden_bias: np.ndarray
    hidden_to_output: np.ndarray
    output_bias: np.ndarray


def create_random_genome(
    shape: NetworkShape,
    rng: Optional[np.random.Generator] = None,
    weight_scale: float = 0.5,
) -> np.ndarray:
    """Create one random genome for generation zero.

    The numbers start near zero. That is important because very large weights
    can make the network outputs slam straight to -1 or 1 before evolution has
    learned anything useful.
    """

    if weight_scale <= 0.0:
        raise ValueError("weight_scale must be greater than zero")

    if rng is None:
        rng = np.random.default_rng()

    return rng.normal(
        loc=0.0,
        scale=weight_scale,
        size=shape.genome_length,
    )


def decode_genome(
    genome: Sequence[float],
    shape: NetworkShape,
) -> NeuralNetworkWeights:
    """Turn one flat genome into neural network weights."""

    genome_array = np.asarray(genome, dtype=float)

    if genome_array.ndim != 1:
        raise ValueError("genome must be one flat array")

    if genome_array.size != shape.genome_length:
        raise ValueError(
            "genome has the wrong length: "
            f"expected {shape.genome_length}, got {genome_array.size}"
        )

    next_index = 0

    input_to_hidden_values, next_index = _take_values(
        genome_array,
        start_index=next_index,
        count=shape.input_to_hidden_count,
    )
    input_to_hidden = input_to_hidden_values.reshape(
        shape.hidden_size,
        shape.input_size,
    )

    hidden_bias, next_index = _take_values(
        genome_array,
        start_index=next_index,
        count=shape.hidden_bias_count,
    )

    hidden_to_output_values, next_index = _take_values(
        genome_array,
        start_index=next_index,
        count=shape.hidden_to_output_count,
    )
    hidden_to_output = hidden_to_output_values.reshape(
        shape.output_size,
        shape.hidden_size,
    )

    output_bias, next_index = _take_values(
        genome_array,
        start_index=next_index,
        count=shape.output_bias_count,
    )

    if next_index != shape.genome_length:
        raise ValueError("genome decoding did not consume the full genome")

    return NeuralNetworkWeights(
        input_to_hidden=input_to_hidden,
        hidden_bias=hidden_bias,
        hidden_to_output=hidden_to_output,
        output_bias=output_bias,
    )


def encode_genome(weights: NeuralNetworkWeights) -> np.ndarray:
    """Turn neural network weights back into one flat genome."""

    return np.concatenate(
        [
            weights.input_to_hidden.ravel(),
            weights.hidden_bias.ravel(),
            weights.hidden_to_output.ravel(),
            weights.output_bias.ravel(),
        ]
    )


def run_network(
    observation: Sequence[float],
    weights: NeuralNetworkWeights,
) -> np.ndarray:
    """Run one observation through the decoded neural network.

    The returned action values are between -1 and 1 because tanh squashes any
    number into that range. Later, the evaluator can map those normalized action
    values to Robosuite's actual action format.
    """

    observation_array = np.asarray(observation, dtype=float)

    if observation_array.ndim != 1:
        raise ValueError("observation must be one flat array")

    expected_input_size = weights.input_to_hidden.shape[1]

    if observation_array.size != expected_input_size:
        raise ValueError(
            "observation has the wrong length: "
            f"expected {expected_input_size}, got {observation_array.size}"
        )

    hidden_raw = weights.input_to_hidden @ observation_array
    hidden_raw = hidden_raw + weights.hidden_bias
    hidden_activated = np.tanh(hidden_raw)

    output_raw = weights.hidden_to_output @ hidden_activated
    output_raw = output_raw + weights.output_bias

    return np.tanh(output_raw)


def _take_values(
    genome: np.ndarray,
    start_index: int,
    count: int,
) -> tuple[np.ndarray, int]:
    end_index = start_index + count
    values = genome[start_index:end_index].copy()

    return values, end_index
