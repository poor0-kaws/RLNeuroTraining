"""Training loop for the neuroevolution system.

The trainer connects the parts we already built.

It does not invent a new learning rule.
It simply runs this loop:

    create population
    evaluate every genome
    log the generation
    select the best genomes
    reproduce the next generation
    repeat
"""

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np

from rl_neuro_training.evaluator import (
    GenomeEvaluationResult,
    PickAndPlaceEvaluatorConfig,
    PickAndPlaceSimulator,
    evaluate_population,
)
from rl_neuro_training.fitness import PickAndPlaceStageScores
from rl_neuro_training.logger import GenerationStats, TrainingLogger
from rl_neuro_training.population import (
    PopulationInitializerConfig,
    create_initial_population,
    validate_population,
)
from rl_neuro_training.reproducer import (
    ReproducerConfig,
    create_next_generation,
)
from rl_neuro_training.selector import SelectorConfig, select_survivors


@dataclass(frozen=True, kw_only=True)
class TrainerConfig:
    """Settings for one full training run.

    generation_count:
        How many generations should be evaluated.

    population_config:
        How generation zero is created.

    evaluator_config:
        How each genome is tested in the simulator.

    selector_config:
        How many top genomes survive as parents.

    reproducer_config:
        How parents create the next generation.
    """

    generation_count: int
    population_config: PopulationInitializerConfig
    evaluator_config: PickAndPlaceEvaluatorConfig
    selector_config: SelectorConfig
    reproducer_config: ReproducerConfig

    def __post_init__(self) -> None:
        if self.generation_count <= 0:
            raise ValueError("generation_count must be greater than zero")

        genome_length = self.evaluator_config.network_shape.genome_length

        if self.population_config.genome_length != genome_length:
            raise ValueError(
                "population_config genome_length must match network genome length"
            )

        population_size = self.population_config.population_size

        if self.selector_config.survivor_count > population_size:
            raise ValueError("survivor_count cannot exceed population_size")

        if self.reproducer_config.population_size != population_size:
            raise ValueError(
                "reproducer_config population_size must match population_size"
            )

        if self.reproducer_config.elite_count > self.selector_config.survivor_count:
            raise ValueError("elite_count cannot exceed survivor_count")


@dataclass(frozen=True)
class ChampionGenomeRecord:
    """The best genome seen during the whole training run."""

    generation_number: int
    genome_index: int
    fitness: float
    stages: PickAndPlaceStageScores
    genome: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    """The final report from training."""

    final_population: np.ndarray
    history: tuple[GenerationStats, ...]
    champion: ChampionGenomeRecord
    generation_fitness_scores: tuple[np.ndarray, ...]


def train(
    simulator_factory: Callable[[], PickAndPlaceSimulator],
    config: TrainerConfig,
    initial_population: Optional[Sequence[Sequence[float]]] = None,
    logger: Optional[TrainingLogger] = None,
) -> TrainingResult:
    """Run the full neuroevolution training loop.

    The simulator_factory creates a fresh simulator for each genome evaluation.
    That keeps one robot's attempt from leaking into another robot's attempt.
    """

    if logger is None:
        logger = TrainingLogger()

    population = _starting_population(
        initial_population=initial_population,
        config=config,
    )
    generation_fitness_scores = []
    champion = None

    for generation_number in range(config.generation_count):
        evaluation_results = evaluate_population(
            population=population,
            simulator_factory=simulator_factory,
            config=config.evaluator_config,
        )
        logger.record_generation(
            generation_number=generation_number,
            evaluation_results=evaluation_results,
        )

        fitness_scores = fitness_scores_from_results(evaluation_results)
        generation_fitness_scores.append(fitness_scores.copy())
        champion = _update_champion(
            current_champion=champion,
            generation_number=generation_number,
            population=population,
            evaluation_results=evaluation_results,
            fitness_scores=fitness_scores,
        )

        if generation_number == config.generation_count - 1:
            continue

        selection = select_survivors(
            population=population,
            fitness_scores=fitness_scores,
            config=config.selector_config,
        )
        reproduction_config = _reproducer_config_for_generation(
            config.reproducer_config,
            generation_number,
        )
        reproduction = create_next_generation(
            survivors=selection.survivors,
            config=reproduction_config,
        )
        population = reproduction.next_population

    if champion is None:
        raise ValueError("training could not find a champion")

    return TrainingResult(
        final_population=population.copy(),
        history=logger.history,
        champion=champion,
        generation_fitness_scores=tuple(generation_fitness_scores),
    )


def fitness_scores_from_results(
    evaluation_results: Sequence[GenomeEvaluationResult],
) -> np.ndarray:
    """Pull one fitness number out of each evaluation result."""

    results = list(evaluation_results)

    if not results:
        raise ValueError("evaluation_results must contain at least one result")

    fitness_scores = np.array(
        [result.fitness.total for result in results],
        dtype=float,
    )

    if not np.all(np.isfinite(fitness_scores)):
        raise ValueError("fitness scores must only contain finite numbers")

    return fitness_scores


def _starting_population(
    initial_population: Optional[Sequence[Sequence[float]]],
    config: TrainerConfig,
) -> np.ndarray:
    if initial_population is None:
        return create_initial_population(config.population_config)

    population = np.asarray(initial_population, dtype=float)
    validate_population(population, config.population_config)

    return population.copy()


def _update_champion(
    current_champion: Optional[ChampionGenomeRecord],
    generation_number: int,
    population: np.ndarray,
    evaluation_results: Sequence[GenomeEvaluationResult],
    fitness_scores: np.ndarray,
) -> ChampionGenomeRecord:
    best_index = int(np.argmax(fitness_scores))
    best_fitness = float(fitness_scores[best_index])

    if current_champion is not None and best_fitness <= current_champion.fitness:
        return current_champion

    best_result = evaluation_results[best_index]

    return ChampionGenomeRecord(
        generation_number=generation_number,
        genome_index=best_index,
        fitness=best_fitness,
        stages=best_result.fitness.stages,
        genome=population[best_index].copy(),
    )


def _reproducer_config_for_generation(
    config: ReproducerConfig,
    generation_number: int,
) -> ReproducerConfig:
    if config.seed is None:
        return config

    return replace(config, seed=config.seed + generation_number)

