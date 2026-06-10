"""Population initialization for generation zero.

A population is a group of genomes.

One genome is one robot brain.
One population is many robot brains that can be tested in the simulator later.

This file only creates the first random population. It does not know about the
robot, the simulator, the fitness score, selection, mutation, or crossover.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True, kw_only=True)
class PopulationInitializerConfig:
    """Settings for creating generation zero.

    population_size:
        How many genomes to create.

    genome_length:
        How many numbers each genome needs.

    min_gene_value and max_gene_value:
        The starting range for each number inside each genome.

    seed:
        Optional number that makes the random population repeatable.
        This is useful when we want to debug the same experiment twice.
    """

    population_size: int = 50
    genome_length: int
    min_gene_value: float = -1.0
    max_gene_value: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be greater than zero")

        if self.genome_length <= 0:
            raise ValueError("genome_length must be greater than zero")

        if not np.isfinite(self.min_gene_value):
            raise ValueError("min_gene_value must be finite")

        if not np.isfinite(self.max_gene_value):
            raise ValueError("max_gene_value must be finite")

        if self.min_gene_value >= self.max_gene_value:
            raise ValueError("min_gene_value must be less than max_gene_value")


def create_initial_population(config: PopulationInitializerConfig) -> np.ndarray:
    """Create the first generation of random genomes.

    The result is a 2D NumPy array.

    Each row is one genome:

        population[0] is robot brain 1
        population[1] is robot brain 2
        population[2] is robot brain 3

    Each column is one number inside that genome.
    """

    rng = np.random.default_rng(config.seed)

    population = rng.uniform(
        low=config.min_gene_value,
        high=config.max_gene_value,
        size=(config.population_size, config.genome_length),
    )

    validate_population(population, config)

    return population


def validate_population(
    population: Sequence[Sequence[float]],
    config: PopulationInitializerConfig,
) -> None:
    """Check that a population has the shape and values we expect."""

    population_array = np.asarray(population, dtype=float)

    if population_array.ndim != 2:
        raise ValueError("population must be a 2D array")

    expected_shape = (config.population_size, config.genome_length)

    if population_array.shape != expected_shape:
        raise ValueError(
            "population has the wrong shape: "
            f"expected {expected_shape}, got {population_array.shape}"
        )

    if not np.all(np.isfinite(population_array)):
        raise ValueError("population must only contain finite numbers")

    if np.any(population_array < config.min_gene_value):
        raise ValueError("population contains a value below min_gene_value")

    if np.any(population_array > config.max_gene_value):
        raise ValueError("population contains a value above max_gene_value")
