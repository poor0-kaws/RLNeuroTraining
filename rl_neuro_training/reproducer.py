"""Reproduction for creating the next generation.

The selector gives us the best genomes from the old generation.
The reproducer uses those genomes to create the new generation.

In simple terms:

    keep a few winners unchanged
    make mutated copies until the population is full again
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True, kw_only=True)
class ReproducerConfig:
    """Settings for creating the next generation.

    population_size:
        How many genomes the next generation should contain.

    elite_count:
        How many top survivors are copied exactly.

    mutation_rate:
        Chance that each gene is changed.
        For example, 0.05 means each genome number has a 5% chance to mutate.

    mutation_strength:
        Typical size of a mutation.
        A value of 0.10 means mutated genes usually move by about 0.10.

    min_gene_value and max_gene_value:
        Safety limits for neural network weights.
        These keep weights from growing forever.

    seed:
        Optional number that makes reproduction repeatable for debugging.
    """

    population_size: int
    elite_count: int = 2
    mutation_rate: float = 0.05
    mutation_strength: float = 0.10
    min_gene_value: float = -3.0
    max_gene_value: float = 3.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be greater than zero")

        if self.elite_count < 0:
            raise ValueError("elite_count cannot be negative")

        if self.elite_count > self.population_size:
            raise ValueError("elite_count cannot be larger than population_size")

        if self.mutation_rate < 0.0 or self.mutation_rate > 1.0:
            raise ValueError("mutation_rate must be between 0.0 and 1.0")

        if self.mutation_strength < 0.0:
            raise ValueError("mutation_strength cannot be negative")

        if not np.isfinite(self.min_gene_value):
            raise ValueError("min_gene_value must be finite")

        if not np.isfinite(self.max_gene_value):
            raise ValueError("max_gene_value must be finite")

        if self.min_gene_value >= self.max_gene_value:
            raise ValueError("min_gene_value must be less than max_gene_value")


@dataclass(frozen=True)
class ReproductionResult:
    """The result from creating a new generation.

    next_population:
        The full new population.

    elite_generation:
        The exact copied winners.

    mutated_generation:
        The new children made from mutated survivor copies.
    """

    next_population: np.ndarray
    elite_generation: np.ndarray
    mutated_generation: np.ndarray


def create_next_generation(
    survivors: Sequence[Sequence[float]],
    config: ReproducerConfig,
) -> ReproductionResult:
    """Create the next generation from sorted survivors.

    The survivors must already be sorted from best to worst.

    That means:

        survivors[0] is the best genome
        survivors[1] is the second-best genome
        survivors[2] is the third-best genome

    The selector is responsible for that sorting.
    """

    survivor_array = _as_survivor_array(survivors)
    _validate_reproduction_inputs(survivor_array, config)

    rng = np.random.default_rng(config.seed)

    elite_generation = survivor_array[: config.elite_count].copy()
    mutated_generation = _create_mutated_generation(
        survivors=survivor_array,
        child_count=config.population_size - config.elite_count,
        config=config,
        rng=rng,
    )
    next_population = np.vstack([elite_generation, mutated_generation])

    return ReproductionResult(
        next_population=next_population,
        elite_generation=elite_generation,
        mutated_generation=mutated_generation,
    )


def mutate_genome(
    genome: Sequence[float],
    config: ReproducerConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Create one mutated copy of one genome."""

    genome_array = np.asarray(genome, dtype=float)

    if genome_array.ndim != 1:
        raise ValueError("genome must be a 1D array")

    if not np.all(np.isfinite(genome_array)):
        raise ValueError("genome must only contain finite numbers")

    if rng is None:
        rng = np.random.default_rng(config.seed)

    child = genome_array.copy()
    mutation_mask = rng.random(size=child.shape) < config.mutation_rate
    mutation_noise = rng.normal(
        loc=0.0,
        scale=config.mutation_strength,
        size=child.shape,
    )

    child[mutation_mask] = child[mutation_mask] + mutation_noise[mutation_mask]

    return np.clip(
        child,
        config.min_gene_value,
        config.max_gene_value,
    )


def _create_mutated_generation(
    survivors: np.ndarray,
    child_count: int,
    config: ReproducerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    genome_length = survivors.shape[1]
    mutated_generation = np.empty((child_count, genome_length), dtype=float)

    for child_index in range(child_count):
        parent_index = rng.integers(0, survivors.shape[0])
        parent = survivors[parent_index]
        child = mutate_genome(parent, config, rng)
        mutated_generation[child_index] = child

    return mutated_generation


def _as_survivor_array(survivors: Sequence[Sequence[float]]) -> np.ndarray:
    survivor_array = np.asarray(survivors, dtype=float)

    if survivor_array.ndim != 2:
        raise ValueError("survivors must be a 2D array")

    if survivor_array.shape[0] == 0:
        raise ValueError("survivors must contain at least one genome")

    if survivor_array.shape[1] == 0:
        raise ValueError("survivor genomes must contain at least one gene")

    if not np.all(np.isfinite(survivor_array)):
        raise ValueError("survivors must only contain finite numbers")

    return survivor_array


def _validate_reproduction_inputs(
    survivors: np.ndarray,
    config: ReproducerConfig,
) -> None:
    survivor_count = survivors.shape[0]

    if config.elite_count > survivor_count:
        raise ValueError("elite_count cannot be larger than survivor count")

