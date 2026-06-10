"""Visualization helpers for understanding training progress.

The visualizer is read-only.

It does not change genomes.
It does not run the simulator.
It does not choose survivors.
It does not mutate anything.

It only turns training results into information humans can inspect.
"""

from dataclasses import dataclass
from html import escape
from typing import Sequence

import numpy as np

from rl_neuro_training.fitness import (
    PickAndPlaceFitnessResult,
    PickAndPlaceStageScores,
)


MAIN_STAGE_NAMES = (
    "reaching",
    "grasping",
    "lifting",
    "moving",
    "placing",
)


@dataclass(frozen=True)
class StageAverages:
    """Average behavior scores for one generation.

    These numbers explain what the average robot is learning.

    Example:

        reaching is high
        grasping is medium
        placing is low

    That means the average robot has learned to get near the object, but has
    not learned to place it well yet.
    """

    reaching: float
    grasping: float
    lifting: float
    moving: float
    placing: float
    placement_accuracy: float
    placement_stability: float


@dataclass(frozen=True)
class RobotPerformanceRecord:
    """A saved snapshot of one important robot's score."""

    generation_index: int
    robot_index: int
    fitness: float
    stages: PickAndPlaceStageScores


@dataclass(frozen=True)
class GenerationProgressRecord:
    """Human-readable progress numbers for one generation."""

    generation_index: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    running_average_fitness: float
    all_time_best_fitness: float
    all_time_worst_fitness: float
    best_robot_index: int
    worst_robot_index: int
    stage_averages: StageAverages


@dataclass(frozen=True)
class TrainingVisualizationSummary:
    """A complete visual summary for all generations so far."""

    records: tuple[GenerationProgressRecord, ...]
    champion: RobotPerformanceRecord
    weakest_robot: RobotPerformanceRecord


@dataclass(frozen=True, kw_only=True)
class FitnessProgressGraphConfig:
    """Layout settings for the SVG fitness graph."""

    width: int = 900
    height: int = 520
    padding_left: int = 70
    padding_right: int = 30
    padding_top: int = 40
    padding_bottom: int = 70
    title: str = "Training Fitness Progress"

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be greater than zero")

        if self.height <= 0:
            raise ValueError("height must be greater than zero")

        horizontal_padding = self.padding_left + self.padding_right
        vertical_padding = self.padding_top + self.padding_bottom

        if horizontal_padding >= self.width:
            raise ValueError("horizontal padding must be smaller than width")

        if vertical_padding >= self.height:
            raise ValueError("vertical padding must be smaller than height")


def summarize_generation(
    generation_index: int,
    fitness_results: Sequence[PickAndPlaceFitnessResult],
    previous_records: Sequence[GenerationProgressRecord] = (),
) -> GenerationProgressRecord:
    """Summarize one generation's fitness results.

    This is the main bridge from learning data to visualization data.
    """

    if generation_index < 0:
        raise ValueError("generation_index cannot be negative")

    results = list(fitness_results)

    if not results:
        raise ValueError("fitness_results must contain at least one result")

    fitness_values = _fitness_values(results)
    best_robot_index = int(np.argmax(fitness_values))
    worst_robot_index = int(np.argmin(fitness_values))

    best_fitness = float(fitness_values[best_robot_index])
    average_fitness = float(np.mean(fitness_values))
    worst_fitness = float(fitness_values[worst_robot_index])

    running_average_fitness = _running_average_fitness(
        current_average=average_fitness,
        previous_records=previous_records,
    )
    all_time_best_fitness = _all_time_best_fitness(
        current_best=best_fitness,
        previous_records=previous_records,
    )
    all_time_worst_fitness = _all_time_worst_fitness(
        current_worst=worst_fitness,
        previous_records=previous_records,
    )

    return GenerationProgressRecord(
        generation_index=generation_index,
        best_fitness=best_fitness,
        average_fitness=average_fitness,
        worst_fitness=worst_fitness,
        running_average_fitness=running_average_fitness,
        all_time_best_fitness=all_time_best_fitness,
        all_time_worst_fitness=all_time_worst_fitness,
        best_robot_index=best_robot_index,
        worst_robot_index=worst_robot_index,
        stage_averages=_stage_averages(results),
    )


def summarize_training(
    generations: Sequence[Sequence[PickAndPlaceFitnessResult]],
) -> TrainingVisualizationSummary:
    """Summarize every generation from the start until now."""

    if not generations:
        raise ValueError("generations must contain at least one generation")

    records: list[GenerationProgressRecord] = []
    champion: RobotPerformanceRecord | None = None
    weakest_robot: RobotPerformanceRecord | None = None

    for generation_index, fitness_results in enumerate(generations):
        generation_results = list(fitness_results)
        record = summarize_generation(
            generation_index=generation_index,
            fitness_results=generation_results,
            previous_records=records,
        )
        records.append(record)

        generation_champion = _make_robot_record(
            generation_index=generation_index,
            robot_index=record.best_robot_index,
            fitness_result=generation_results[record.best_robot_index],
        )
        generation_weakest_robot = _make_robot_record(
            generation_index=generation_index,
            robot_index=record.worst_robot_index,
            fitness_result=generation_results[record.worst_robot_index],
        )

        if champion is None or generation_champion.fitness > champion.fitness:
            champion = generation_champion

        if weakest_robot is None:
            weakest_robot = generation_weakest_robot
            continue

        if generation_weakest_robot.fitness < weakest_robot.fitness:
            weakest_robot = generation_weakest_robot

    if champion is None or weakest_robot is None:
        raise ValueError("training summary could not find any robots")

    return TrainingVisualizationSummary(
        records=tuple(records),
        champion=champion,
        weakest_robot=weakest_robot,
    )


def strongest_average_stage(record: GenerationProgressRecord) -> str:
    """Return the behavior stage the average robot is best at."""

    stage_values = {
        "reaching": record.stage_averages.reaching,
        "grasping": record.stage_averages.grasping,
        "lifting": record.stage_averages.lifting,
        "moving": record.stage_averages.moving,
        "placing": record.stage_averages.placing,
    }

    return max(stage_values, key=stage_values.get)


def render_fitness_progress_svg(
    records: Sequence[GenerationProgressRecord],
    config: FitnessProgressGraphConfig | None = None,
) -> str:
    """Render a simple SVG graph of training progress.

    The graph shows:

        best fitness each generation
        average fitness each generation
        worst fitness each generation
        running average fitness
        all-time best fitness
        all-time worst fitness
        shaded range between worst and best
    """

    record_list = list(records)

    if not record_list:
        raise ValueError("records must contain at least one record")

    if config is None:
        config = FitnessProgressGraphConfig()

    plot = _PlotArea(config)
    best_points = _points_for_values(
        record_list,
        [record.best_fitness for record in record_list],
        plot,
    )
    average_points = _points_for_values(
        record_list,
        [record.average_fitness for record in record_list],
        plot,
    )
    worst_points = _points_for_values(
        record_list,
        [record.worst_fitness for record in record_list],
        plot,
    )
    running_average_points = _points_for_values(
        record_list,
        [record.running_average_fitness for record in record_list],
        plot,
    )
    all_time_best_points = _points_for_values(
        record_list,
        [record.all_time_best_fitness for record in record_list],
        plot,
    )
    all_time_worst_points = _points_for_values(
        record_list,
        [record.all_time_worst_fitness for record in record_list],
        plot,
    )

    range_polygon = _range_polygon_points(best_points, worst_points)
    generation_labels = _generation_axis_labels(record_list, plot)

    return "\n".join(
        [
            _svg_header(config),
            _svg_background(config),
            _svg_title(config),
            _svg_axes(plot),
            _svg_y_axis_labels(plot),
            generation_labels,
            _svg_polygon(range_polygon, "#dbeafe", "fitness range"),
            _svg_polyline(worst_points, "#dc2626", "worst fitness"),
            _svg_polyline(average_points, "#2563eb", "average fitness"),
            _svg_polyline(best_points, "#16a34a", "best fitness"),
            _svg_polyline(
                running_average_points,
                "#7c3aed",
                "running average fitness",
            ),
            _svg_polyline(
                all_time_best_points,
                "#111827",
                "all-time best fitness",
                dash_pattern="6 4",
            ),
            _svg_polyline(
                all_time_worst_points,
                "#6b7280",
                "all-time worst fitness",
                dash_pattern="6 4",
            ),
            _svg_legend(config),
            "</svg>",
        ]
    )


def _fitness_values(results: Sequence[PickAndPlaceFitnessResult]) -> np.ndarray:
    values = np.array([result.total for result in results], dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError("fitness values must be finite")

    return values


def _running_average_fitness(
    current_average: float,
    previous_records: Sequence[GenerationProgressRecord],
) -> float:
    if not previous_records:
        return current_average

    previous_average_total = sum(
        record.average_fitness for record in previous_records
    )

    return (previous_average_total + current_average) / (
        len(previous_records) + 1
    )


def _all_time_best_fitness(
    current_best: float,
    previous_records: Sequence[GenerationProgressRecord],
) -> float:
    if not previous_records:
        return current_best

    previous_best = max(
        record.all_time_best_fitness for record in previous_records
    )

    return max(previous_best, current_best)


def _all_time_worst_fitness(
    current_worst: float,
    previous_records: Sequence[GenerationProgressRecord],
) -> float:
    if not previous_records:
        return current_worst

    previous_worst = min(
        record.all_time_worst_fitness for record in previous_records
    )

    return min(previous_worst, current_worst)


def _stage_averages(
    results: Sequence[PickAndPlaceFitnessResult],
) -> StageAverages:
    reaching = []
    grasping = []
    lifting = []
    moving = []
    placing = []
    placement_accuracy = []
    placement_stability = []

    for result in results:
        stages = result.stages
        reaching.append(stages.reaching)
        grasping.append(stages.grasping)
        lifting.append(stages.lifting)
        moving.append(stages.moving)
        placing.append(stages.placing)
        placement_accuracy.append(stages.placement_accuracy)
        placement_stability.append(stages.placement_stability)

    return StageAverages(
        reaching=_mean(reaching, "reaching"),
        grasping=_mean(grasping, "grasping"),
        lifting=_mean(lifting, "lifting"),
        moving=_mean(moving, "moving"),
        placing=_mean(placing, "placing"),
        placement_accuracy=_mean(placement_accuracy, "placement_accuracy"),
        placement_stability=_mean(
            placement_stability,
            "placement_stability",
        ),
    )


def _mean(values: Sequence[float], name: str) -> float:
    value_array = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(value_array)):
        raise ValueError(f"{name} values must be finite")

    return float(np.mean(value_array))


def _make_robot_record(
    generation_index: int,
    robot_index: int,
    fitness_result: PickAndPlaceFitnessResult,
) -> RobotPerformanceRecord:
    return RobotPerformanceRecord(
        generation_index=generation_index,
        robot_index=robot_index,
        fitness=fitness_result.total,
        stages=fitness_result.stages,
    )


@dataclass(frozen=True)
class _PlotArea:
    config: FitnessProgressGraphConfig

    @property
    def left(self) -> float:
        return float(self.config.padding_left)

    @property
    def right(self) -> float:
        return float(self.config.width - self.config.padding_right)

    @property
    def top(self) -> float:
        return float(self.config.padding_top)

    @property
    def bottom(self) -> float:
        return float(self.config.height - self.config.padding_bottom)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def x_for_generation(
        self,
        record: GenerationProgressRecord,
        records: Sequence[GenerationProgressRecord],
    ) -> float:
        if len(records) == 1:
            return self.left + (self.width / 2.0)

        first_generation = records[0].generation_index
        last_generation = records[-1].generation_index
        generation_span = last_generation - first_generation

        if generation_span <= 0:
            return self.left

        progress = (record.generation_index - first_generation) / generation_span

        return self.left + (progress * self.width)

    def y_for_fitness(self, fitness: float) -> float:
        clamped_fitness = _clamp_01(fitness)

        return self.bottom - (clamped_fitness * self.height)


def _points_for_values(
    records: Sequence[GenerationProgressRecord],
    values: Sequence[float],
    plot: _PlotArea,
) -> list[tuple[float, float]]:
    points = []

    for record, value in zip(records, values):
        points.append(
            (
                plot.x_for_generation(record, records),
                plot.y_for_fitness(value),
            )
        )

    return points


def _range_polygon_points(
    best_points: Sequence[tuple[float, float]],
    worst_points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    reversed_worst_points = list(reversed(worst_points))

    return list(best_points) + reversed_worst_points


def _svg_header(config: FitnessProgressGraphConfig) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{config.width}" height="{config.height}" '
        f'viewBox="0 0 {config.width} {config.height}" '
        f'role="img" aria-label="{escape(config.title)}">'
    )


def _svg_background(config: FitnessProgressGraphConfig) -> str:
    return (
        f'<rect x="0" y="0" width="{config.width}" height="{config.height}" '
        'fill="#ffffff"/>'
    )


def _svg_title(config: FitnessProgressGraphConfig) -> str:
    title = escape(config.title)

    return (
        f'<text x="{config.width / 2}" y="24" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="18" fill="#111827">'
        f"{title}</text>"
    )


def _svg_axes(plot: _PlotArea) -> str:
    return "\n".join(
        [
            (
                f'<line x1="{plot.left:.2f}" y1="{plot.bottom:.2f}" '
                f'x2="{plot.right:.2f}" y2="{plot.bottom:.2f}" '
                'stroke="#111827" stroke-width="1"/>'
            ),
            (
                f'<line x1="{plot.left:.2f}" y1="{plot.top:.2f}" '
                f'x2="{plot.left:.2f}" y2="{plot.bottom:.2f}" '
                'stroke="#111827" stroke-width="1"/>'
            ),
            (
                f'<text x="{plot.left - 45:.2f}" y="{plot.top - 12:.2f}" '
                'font-family="Arial, sans-serif" font-size="12" '
                'fill="#374151">fitness</text>'
            ),
            (
                f'<text x="{(plot.left + plot.right) / 2:.2f}" '
                f'y="{plot.bottom + 45:.2f}" text-anchor="middle" '
                'font-family="Arial, sans-serif" font-size="12" '
                'fill="#374151">generation</text>'
            ),
        ]
    )


def _svg_y_axis_labels(plot: _PlotArea) -> str:
    labels = []

    for value in (0.0, 0.25, 0.50, 0.75, 1.0):
        y = plot.y_for_fitness(value)
        labels.append(
            (
                f'<line x1="{plot.left - 5:.2f}" y1="{y:.2f}" '
                f'x2="{plot.left:.2f}" y2="{y:.2f}" '
                'stroke="#111827" stroke-width="1"/>'
            )
        )
        labels.append(
            (
                f'<text x="{plot.left - 10:.2f}" y="{y + 4:.2f}" '
                'text-anchor="end" font-family="Arial, sans-serif" '
                f'font-size="11" fill="#374151">{value:.2f}</text>'
            )
        )

    return "\n".join(labels)


def _generation_axis_labels(
    records: Sequence[GenerationProgressRecord],
    plot: _PlotArea,
) -> str:
    labels = []

    for record in records:
        x = plot.x_for_generation(record, records)
        labels.append(
            (
                f'<line x1="{x:.2f}" y1="{plot.bottom:.2f}" '
                f'x2="{x:.2f}" y2="{plot.bottom + 5:.2f}" '
                'stroke="#111827" stroke-width="1"/>'
            )
        )
        labels.append(
            (
                f'<text x="{x:.2f}" y="{plot.bottom + 22:.2f}" '
                'text-anchor="middle" font-family="Arial, sans-serif" '
                f'font-size="11" fill="#374151">{record.generation_index}</text>'
            )
        )

    return "\n".join(labels)


def _svg_polygon(
    points: Sequence[tuple[float, float]],
    color: str,
    label: str,
) -> str:
    point_text = _format_points(points)

    return (
        f'<polygon points="{point_text}" fill="{color}" opacity="0.45">'
        f"<title>{escape(label)}</title></polygon>"
    )


def _svg_polyline(
    points: Sequence[tuple[float, float]],
    color: str,
    label: str,
    dash_pattern: str | None = None,
) -> str:
    point_text = _format_points(points)
    dash_attribute = ""

    if dash_pattern is not None:
        dash_attribute = f' stroke-dasharray="{dash_pattern}"'

    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="2.5" stroke-linecap="round" '
        f'stroke-linejoin="round"{dash_attribute}>'
        f"<title>{escape(label)}</title></polyline>"
    )


def _svg_legend(config: FitnessProgressGraphConfig) -> str:
    legend_items = [
        ("#16a34a", "best"),
        ("#2563eb", "average"),
        ("#dc2626", "worst"),
        ("#7c3aed", "running average"),
        ("#111827", "all-time best"),
        ("#6b7280", "all-time worst"),
    ]
    item_width = 125
    start_x = config.padding_left
    y = config.height - 20
    parts = []

    for index, (color, label) in enumerate(legend_items):
        x = start_x + (index * item_width)
        parts.append(
            (
                f'<line x1="{x}" y1="{y - 4}" x2="{x + 18}" y2="{y - 4}" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
        )
        parts.append(
            (
                f'<text x="{x + 24}" y="{y}" '
                'font-family="Arial, sans-serif" font-size="11" '
                f'fill="#374151">{escape(label)}</text>'
            )
        )

    return "\n".join(parts)


def _format_points(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0

    if value > 1.0:
        return 1.0

    return value

