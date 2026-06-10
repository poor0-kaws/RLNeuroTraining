"""Training logger for generation-level progress.

The logger remembers what happened during training.

It does not run the robot.
It does not score behavior.
It does not select winners.
It does not create children.

It only turns many evaluation results into simple numbers we can inspect later.
"""

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from rl_neuro_training.evaluator import GenomeEvaluationResult


GENERATION_STATS_CSV_FIELDS = [
    "generation",
    "population_size",
    "best_fitness",
    "average_fitness",
    "median_fitness",
    "worst_fitness",
    "fitness_std",
    "best_genome_index",
    "best_reaching",
    "best_grasping",
    "best_lifting",
    "best_moving",
    "best_placing",
    "best_placement_accuracy",
    "best_placement_stability",
]


@dataclass(frozen=True)
class GenerationStats:
    """A small report for one generation.

    Fitness numbers tell us if the population is improving overall.
    Stage numbers tell us what the best robot actually learned to do.
    """

    generation_number: int
    population_size: int
    best_fitness: float
    average_fitness: float
    median_fitness: float
    worst_fitness: float
    fitness_std: float
    best_genome_index: int
    best_reaching: float
    best_grasping: float
    best_lifting: float
    best_moving: float
    best_placing: float
    best_placement_accuracy: float
    best_placement_stability: float


class TrainingLogger:
    """In-memory history of generation summaries."""

    def __init__(self) -> None:
        self._history: list[GenerationStats] = []

    @property
    def history(self) -> tuple[GenerationStats, ...]:
        """Return all recorded generation summaries."""

        return tuple(self._history)

    def record_generation(
        self,
        generation_number: int,
        evaluation_results: Sequence[GenomeEvaluationResult],
    ) -> GenerationStats:
        """Summarize one generation and remember it."""

        stats = summarize_generation(
            generation_number=generation_number,
            evaluation_results=evaluation_results,
        )

        self._history.append(stats)

        return stats

    def latest(self) -> GenerationStats | None:
        """Return the most recent generation summary."""

        if not self._history:
            return None

        return self._history[-1]

    def to_rows(self) -> list[dict[str, float | int]]:
        """Return logger history as simple CSV-ready rows."""

        return [generation_stats_to_row(stats) for stats in self._history]

    def write_csv(self, path: str | Path) -> None:
        """Save all recorded generation summaries to a CSV file."""

        csv_path = Path(path)

        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=GENERATION_STATS_CSV_FIELDS,
            )
            writer.writeheader()
            writer.writerows(self.to_rows())


def summarize_generation(
    generation_number: int,
    evaluation_results: Sequence[GenomeEvaluationResult],
) -> GenerationStats:
    """Create one readable summary from many genome evaluations."""

    if generation_number < 0:
        raise ValueError("generation_number cannot be negative")

    results = list(evaluation_results)

    if not results:
        raise ValueError("evaluation_results must contain at least one result")

    fitness_values = _fitness_values_from_results(results)
    best_genome_index = int(np.argmax(fitness_values))
    best_result = results[best_genome_index]
    best_stages = best_result.fitness.stages

    return GenerationStats(
        generation_number=generation_number,
        population_size=len(results),
        best_fitness=float(np.max(fitness_values)),
        average_fitness=float(np.mean(fitness_values)),
        median_fitness=float(np.median(fitness_values)),
        worst_fitness=float(np.min(fitness_values)),
        fitness_std=float(np.std(fitness_values)),
        best_genome_index=best_genome_index,
        best_reaching=float(best_stages.reaching),
        best_grasping=float(best_stages.grasping),
        best_lifting=float(best_stages.lifting),
        best_moving=float(best_stages.moving),
        best_placing=float(best_stages.placing),
        best_placement_accuracy=float(best_stages.placement_accuracy),
        best_placement_stability=float(best_stages.placement_stability),
    )


def generation_stats_to_row(stats: GenerationStats) -> dict[str, float | int]:
    """Turn GenerationStats into one flat CSV row."""

    stats_dict = asdict(stats)

    return {
        "generation": stats_dict["generation_number"],
        "population_size": stats_dict["population_size"],
        "best_fitness": stats_dict["best_fitness"],
        "average_fitness": stats_dict["average_fitness"],
        "median_fitness": stats_dict["median_fitness"],
        "worst_fitness": stats_dict["worst_fitness"],
        "fitness_std": stats_dict["fitness_std"],
        "best_genome_index": stats_dict["best_genome_index"],
        "best_reaching": stats_dict["best_reaching"],
        "best_grasping": stats_dict["best_grasping"],
        "best_lifting": stats_dict["best_lifting"],
        "best_moving": stats_dict["best_moving"],
        "best_placing": stats_dict["best_placing"],
        "best_placement_accuracy": stats_dict["best_placement_accuracy"],
        "best_placement_stability": stats_dict["best_placement_stability"],
    }


def _fitness_values_from_results(
    evaluation_results: Sequence[GenomeEvaluationResult],
) -> np.ndarray:
    fitness_values = np.array(
        [result.fitness.total for result in evaluation_results],
        dtype=float,
    )

    if not np.all(np.isfinite(fitness_values)):
        raise ValueError("fitness values must only contain finite numbers")

    return fitness_values
