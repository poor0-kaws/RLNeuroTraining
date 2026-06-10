"""Selection for choosing which genomes survive.

The evaluator gives every genome a fitness score.
The selector uses those scores to choose the winners.

This file does not know about simulators, robots, mutation, or crossover.
It only answers one question:

    "Which genomes did best?"
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, kw_only=True)
class SelectorConfig:
    """Settings for survivor selection.

    survivor_count:
        How many genomes should survive into the parent group.
    """

    survivor_count: int

    def __post_init__(self) -> None:
        if self.survivor_count <= 0:
            raise ValueError("survivor_count must be greater than zero")


@dataclass(frozen=True)
class SelectionResult:
    """The output from selecting survivors.

    survivors:
        The genomes that survived.

    survivor_fitness_scores:
        The fitness score for each survivor.

    survivor_indices:
        Where each survivor came from in the original population.
        This is useful for logging and debugging.
    """

    survivors: np.ndarray
    survivor_fitness_scores: np.ndarray
    survivor_indices: np.ndarray


def select_survivors(
    population: Sequence[Sequence[float]],
    fitness_scores: Sequence[float],
    config: SelectorConfig,
) -> SelectionResult:
    """Choose the best genomes from a scored population.

    Higher fitness is better.

    Example:

        population has 50 genomes
        fitness_scores has 50 numbers
        survivor_count is 10

    This function returns the 10 genomes with the highest fitness.
    """

    population_array = _as_population_array(population)
    fitness_array = _as_fitness_array(fitness_scores)

    _validate_selection_inputs(
        population=population_array,
        fitness_scores=fitness_array,
        config=config,
    )

    ranked_indices = _rank_indices_by_fitness(fitness_array)
    survivor_indices = ranked_indices[: config.survivor_count]

    return SelectionResult(
        survivors=population_array[survivor_indices].copy(),
        survivor_fitness_scores=fitness_array[survivor_indices].copy(),
        survivor_indices=survivor_indices.copy(),
    )


def _as_population_array(population: Sequence[Sequence[float]]) -> np.ndarray:
    population_array = np.asarray(population, dtype=float)

    if population_array.ndim != 2:
        raise ValueError("population must be a 2D array")

    if not np.all(np.isfinite(population_array)):
        raise ValueError("population must only contain finite numbers")

    return population_array


def _as_fitness_array(fitness_scores: Sequence[float]) -> np.ndarray:
    fitness_array = np.asarray(fitness_scores, dtype=float)

    if fitness_array.ndim != 1:
        raise ValueError("fitness_scores must be a 1D array")

    if not np.all(np.isfinite(fitness_array)):
        raise ValueError("fitness_scores must only contain finite numbers")

    return fitness_array


def _validate_selection_inputs(
    population: np.ndarray,
    fitness_scores: np.ndarray,
    config: SelectorConfig,
) -> None:
    population_size = population.shape[0]

    if fitness_scores.size != population_size:
        raise ValueError(
            "fitness_scores must have one score per genome: "
            f"expected {population_size}, got {fitness_scores.size}"
        )

    if config.survivor_count > population_size:
        raise ValueError("survivor_count cannot be larger than population size")


def _rank_indices_by_fitness(fitness_scores: np.ndarray) -> np.ndarray:
    original_indices = np.arange(fitness_scores.size)

    return np.lexsort(
        (
            original_indices,
            -fitness_scores,
        )
    )

