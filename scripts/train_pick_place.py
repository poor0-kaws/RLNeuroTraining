"""Run Panda pick-and-place neuroevolution training.

Use normal Python for headless training:

    python scripts/train_pick_place.py --generations 3 --population-size 10

Use mjpython for live MuJoCo rendering on macOS:

    mjpython scripts/train_pick_place.py --render
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rl_neuro_training.evaluator import (  # noqa: E402
    END_EFFECTOR_ACTION_MODE,
    JOINT_ACTION_MODE,
    PickAndPlaceEvaluatorConfig,
    make_pick_and_place_network_shape,
)
from rl_neuro_training.logger import GenerationStats, TrainingLogger  # noqa: E402
from rl_neuro_training.population import PopulationInitializerConfig  # noqa: E402
from rl_neuro_training.reproducer import ReproducerConfig  # noqa: E402
from rl_neuro_training.robosuite_adapter import (  # noqa: E402
    RobosuitePickAndPlaceConfig,
    make_robosuite_pick_and_place_simulator_factory,
)
from rl_neuro_training.selector import SelectorConfig  # noqa: E402
from rl_neuro_training.trainer import TrainerConfig, TrainingResult, train  # noqa: E402
from rl_neuro_training.visualizer import (  # noqa: E402
    FitnessProgressGraphConfig,
    GenerationProgressRecord,
    StageAverages,
    render_fitness_progress_svg,
)


@dataclass(frozen=True)
class TrainingArtifactPaths:
    """Paths written after one training run."""

    output_dir: Path
    csv_log: Path
    champion_genome: Path
    champion_metadata: Path
    final_population: Path
    generation_fitness_scores: Path
    fitness_progress_svg: Path
    run_config: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Panda pick-and-place controller with neuroevolution.",
    )
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--action-mode",
        choices=(END_EFFECTOR_ACTION_MODE, JOINT_ACTION_MODE),
        default=END_EFFECTOR_ACTION_MODE,
    )
    parser.add_argument("--survivor-fraction", type=float, default=0.20)
    parser.add_argument("--survivor-count", type=int, default=None)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--mutation-strength", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--env-seed",
        type=int,
        default=None,
        help="Optional Robosuite environment seed for repeatable resets.",
    )
    parser.add_argument("--object-type", default="can")
    parser.add_argument("--table-height", type=float, default=0.8)
    parser.add_argument("--output-dir", default="runs/pick_place")
    parser.add_argument(
        "--resume-population",
        default=None,
        help="Optional final_population.npy file to continue training from.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render live. On macOS, run this script with mjpython.",
    )

    return parser.parse_args(argv)


def build_trainer_config(args: argparse.Namespace) -> TrainerConfig:
    """Build the trainer config from command-line values."""

    network_shape = make_pick_and_place_network_shape(
        hidden_size=args.hidden_size,
        action_mode=args.action_mode,
    )
    survivor_count = choose_survivor_count(
        population_size=args.population_size,
        survivor_fraction=args.survivor_fraction,
        survivor_count=args.survivor_count,
        elite_count=args.elite_count,
    )

    return TrainerConfig(
        generation_count=args.generations,
        population_config=PopulationInitializerConfig(
            population_size=args.population_size,
            genome_length=network_shape.genome_length,
            seed=args.seed,
        ),
        evaluator_config=PickAndPlaceEvaluatorConfig(
            network_shape=network_shape,
            max_steps=args.max_steps,
            action_mode=args.action_mode,
        ),
        selector_config=SelectorConfig(
            survivor_count=survivor_count,
        ),
        reproducer_config=ReproducerConfig(
            population_size=args.population_size,
            elite_count=args.elite_count,
            mutation_rate=args.mutation_rate,
            mutation_strength=args.mutation_strength,
            seed=args.seed,
        ),
    )


def build_robosuite_config(args: argparse.Namespace) -> RobosuitePickAndPlaceConfig:
    """Build the real Panda PickPlace simulator config."""

    env_kwargs = {
        "use_object_obs": True,
        "single_object_mode": 2,
        "object_type": args.object_type,
        "horizon": args.max_steps,
    }

    if args.env_seed is not None:
        env_kwargs["seed"] = args.env_seed

    return RobosuitePickAndPlaceConfig(
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_each_step=args.render,
        table_height=args.table_height,
        env_kwargs=env_kwargs,
    )


def choose_survivor_count(
    population_size: int,
    survivor_fraction: float,
    survivor_count: int | None,
    elite_count: int,
) -> int:
    """Choose how many genomes survive as the parent pool.

    If survivor_count is provided, we use it directly.
    Otherwise, we use the top survivor_fraction of the population.
    """

    if population_size <= 0:
        raise ValueError("population_size must be greater than zero")

    if elite_count <= 0:
        raise ValueError("elite_count must be greater than zero")

    if survivor_count is None:
        if survivor_fraction <= 0.0 or survivor_fraction > 1.0:
            raise ValueError("survivor_fraction must be between 0.0 and 1.0")

        survivor_count = int(round(population_size * survivor_fraction))
        survivor_count = max(survivor_count, elite_count)

    if survivor_count <= 0:
        raise ValueError("survivor_count must be greater than zero")

    if survivor_count > population_size:
        raise ValueError("survivor_count cannot exceed population_size")

    if elite_count > survivor_count:
        raise ValueError("elite_count cannot exceed survivor_count")

    return survivor_count


def save_training_artifacts(
    result: TrainingResult,
    logger: TrainingLogger,
    output_dir: Path,
    run_config: dict[str, object] | None = None,
) -> TrainingArtifactPaths:
    """Save the useful files from one training run."""

    output_dir.mkdir(parents=True, exist_ok=True)

    paths = TrainingArtifactPaths(
        output_dir=output_dir,
        csv_log=output_dir / "training_log.csv",
        champion_genome=output_dir / "champion_genome.npy",
        champion_metadata=output_dir / "champion_metadata.json",
        final_population=output_dir / "final_population.npy",
        generation_fitness_scores=output_dir / "generation_fitness_scores.npy",
        fitness_progress_svg=output_dir / "fitness_progress.svg",
        run_config=output_dir / "run_config.json",
    )

    logger.write_csv(paths.csv_log)
    np.save(paths.champion_genome, result.champion.genome)
    np.save(paths.final_population, result.final_population)
    np.save(
        paths.generation_fitness_scores,
        np.vstack(result.generation_fitness_scores),
    )

    metadata = champion_metadata(result)
    paths.champion_metadata.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    graph_records = progress_records_from_generation_stats(result.history)
    svg = render_fitness_progress_svg(
        graph_records,
        FitnessProgressGraphConfig(),
    )
    paths.fitness_progress_svg.write_text(svg, encoding="utf-8")
    paths.run_config.write_text(
        json.dumps(run_config or {"format_version": 1}, indent=2),
        encoding="utf-8",
    )

    return paths


def load_initial_population(path: str | Path) -> np.ndarray:
    """Load a saved population for resume training."""

    population_path = Path(path)

    if not population_path.exists():
        raise FileNotFoundError(
            f"resume population file does not exist: {population_path}"
        )

    population = np.asarray(np.load(population_path), dtype=float)

    if population.ndim != 2:
        raise ValueError("resume population must be a 2D array")

    if not np.all(np.isfinite(population)):
        raise ValueError("resume population must only contain finite numbers")

    return population


def run_config_metadata(
    args: argparse.Namespace,
    trainer_config: TrainerConfig,
) -> dict[str, object]:
    """Create the JSON config that lets replay know how training was run."""

    network_shape = trainer_config.evaluator_config.network_shape

    return {
        "format_version": 1,
        "script": "scripts/train_pick_place.py",
        "training": {
            "generations": args.generations,
            "population_size": args.population_size,
            "survivor_fraction": args.survivor_fraction,
            "survivor_count": trainer_config.selector_config.survivor_count,
            "elite_count": args.elite_count,
            "mutation_rate": args.mutation_rate,
            "mutation_strength": args.mutation_strength,
            "seed": args.seed,
            "resume_population": args.resume_population,
        },
        "replay": {
            "hidden_size": args.hidden_size,
            "max_steps": args.max_steps,
            "action_mode": args.action_mode,
            "object_type": args.object_type,
            "table_height": args.table_height,
            "env_seed": args.env_seed,
        },
        "network_shape": {
            "input_size": network_shape.input_size,
            "hidden_size": network_shape.hidden_size,
            "output_size": network_shape.output_size,
            "genome_length": network_shape.genome_length,
        },
    }


def progress_records_from_generation_stats(
    history: Sequence[GenerationStats],
) -> tuple[GenerationProgressRecord, ...]:
    """Convert logger stats into visualizer records."""

    records = []
    average_fitness_total = 0.0
    all_time_best_fitness = None
    all_time_worst_fitness = None

    for stats in history:
        average_fitness_total += stats.average_fitness

        if all_time_best_fitness is None:
            all_time_best_fitness = stats.best_fitness
        else:
            all_time_best_fitness = max(all_time_best_fitness, stats.best_fitness)

        if all_time_worst_fitness is None:
            all_time_worst_fitness = stats.worst_fitness
        else:
            all_time_worst_fitness = min(all_time_worst_fitness, stats.worst_fitness)

        running_average = average_fitness_total / (len(records) + 1)

        records.append(
            GenerationProgressRecord(
                generation_index=stats.generation_number,
                best_fitness=stats.best_fitness,
                average_fitness=stats.average_fitness,
                worst_fitness=stats.worst_fitness,
                running_average_fitness=running_average,
                all_time_best_fitness=all_time_best_fitness,
                all_time_worst_fitness=all_time_worst_fitness,
                best_robot_index=stats.best_genome_index,
                worst_robot_index=stats.worst_genome_index,
                stage_averages=StageAverages(
                    reaching=stats.average_reaching,
                    grasping=stats.average_grasping,
                    lifting=stats.average_lifting,
                    moving=stats.average_moving,
                    placing=stats.average_placing,
                    placement_accuracy=stats.average_placement_accuracy,
                    placement_stability=stats.average_placement_stability,
                ),
            )
        )

    return tuple(records)


def champion_metadata(result: TrainingResult) -> dict[str, object]:
    """Create a small JSON-safe champion report."""

    champion = result.champion

    return {
        "generation_number": champion.generation_number,
        "genome_index": champion.genome_index,
        "fitness": champion.fitness,
        "stages": {
            "reaching": champion.stages.reaching,
            "grasping": champion.stages.grasping,
            "lifting": champion.stages.lifting,
            "moving": champion.stages.moving,
            "placing": champion.stages.placing,
            "placement_accuracy": champion.stages.placement_accuracy,
            "placement_stability": champion.stages.placement_stability,
        },
        "genome_length": int(champion.genome.size),
    }


def print_summary(result: TrainingResult, paths: TrainingArtifactPaths) -> None:
    """Print the important result paths after training."""

    latest_stats = result.history[-1]

    print("Training complete")
    print(f"Generations evaluated: {len(result.history)}")
    print(f"Final generation best fitness: {latest_stats.best_fitness:.6f}")
    print(f"Final generation average fitness: {latest_stats.average_fitness:.6f}")
    print(f"Champion fitness: {result.champion.fitness:.6f}")
    print(f"Champion generation: {result.champion.generation_number}")
    print(f"Output directory: {paths.output_dir}")
    print(f"CSV log: {paths.csv_log}")
    print(f"Champion genome: {paths.champion_genome}")
    print(f"Progress graph: {paths.fitness_progress_svg}")
    print(f"Run config: {paths.run_config}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    logger = TrainingLogger()
    trainer_config = build_trainer_config(args)
    robosuite_config = build_robosuite_config(args)
    simulator_factory = make_robosuite_pick_and_place_simulator_factory(
        robosuite_config,
    )
    initial_population = None

    if args.resume_population is not None:
        initial_population = load_initial_population(args.resume_population)

    result = train(
        simulator_factory=simulator_factory,
        config=trainer_config,
        initial_population=initial_population,
        logger=logger,
    )
    paths = save_training_artifacts(
        result=result,
        logger=logger,
        output_dir=output_dir,
        run_config=run_config_metadata(args, trainer_config),
    )
    print_summary(result, paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
