"""Replay a saved champion genome in Panda pick-and-place.

Use normal Python for headless replay:

    python scripts/replay_champion.py --champion-genome runs/pick_place/champion_genome.npy

Use mjpython for live MuJoCo rendering on macOS:

    mjpython scripts/replay_champion.py --render --champion-genome runs/pick_place/champion_genome.npy
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rl_neuro_training.evaluator import (  # noqa: E402
    END_EFFECTOR_ACTION_MODE,
    JOINT_ACTION_MODE,
    PickAndPlaceEvaluatorConfig,
    close_simulator,
    evaluate_genome,
    make_pick_and_place_network_shape,
)
from rl_neuro_training.evaluator import GenomeEvaluationResult  # noqa: E402
from rl_neuro_training.robosuite_adapter import (  # noqa: E402
    RobosuitePickAndPlaceConfig,
    make_robosuite_pick_and_place_simulator,
)


DEFAULT_CHAMPION_GENOME = "runs/pick_place/champion_genome.npy"
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_MAX_STEPS = 50
DEFAULT_ACTION_MODE = END_EFFECTOR_ACTION_MODE
DEFAULT_OBJECT_TYPE = "can"
DEFAULT_TABLE_HEIGHT = 0.8


@dataclass(frozen=True)
class ReplaySettings:
    """Fully resolved replay settings.

    The command line can leave some values blank.
    This object holds the final values after reading run_config.json.
    """

    champion_genome: Path
    run_config: Path | None
    hidden_size: int
    max_steps: int
    action_mode: str
    object_type: str
    table_height: float
    env_seed: int | None
    render: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a saved Panda pick-and-place champion genome.",
    )
    parser.add_argument(
        "--champion-genome",
        default=DEFAULT_CHAMPION_GENOME,
        help="Path to champion_genome.npy from training.",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help=(
            "Optional run_config.json from training. "
            "By default, replay looks next to the genome."
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--action-mode",
        choices=(END_EFFECTOR_ACTION_MODE, JOINT_ACTION_MODE),
        default=None,
    )
    parser.add_argument("--object-type", default=None)
    parser.add_argument("--table-height", type=float, default=None)
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render live. On macOS, run this script with mjpython.",
    )

    return parser.parse_args(argv)


def default_run_config_path(champion_genome: str | Path) -> Path:
    """Return the run_config.json path next to a champion genome."""

    return Path(champion_genome).parent / "run_config.json"


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load the training config JSON saved by train_pick_place.py."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"run config file does not exist: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("run config must be a JSON object")

    return data


def resolve_replay_settings(args: argparse.Namespace) -> ReplaySettings:
    """Combine command-line values with run_config.json values.

    Think of it like this:

        command line wins
        run_config.json is the backup
        hard-coded defaults are the last backup
    """

    champion_genome = Path(args.champion_genome)
    run_config_path = _resolve_run_config_path(args, champion_genome)
    run_config = _load_optional_run_config(args, run_config_path)
    replay_config = _replay_section(run_config)

    return ReplaySettings(
        champion_genome=champion_genome,
        run_config=run_config_path,
        hidden_size=int(
            _setting_value(
                args.hidden_size,
                replay_config,
                "hidden_size",
                DEFAULT_HIDDEN_SIZE,
            )
        ),
        max_steps=int(
            _setting_value(
                args.max_steps,
                replay_config,
                "max_steps",
                DEFAULT_MAX_STEPS,
            )
        ),
        action_mode=str(
            _setting_value(
                args.action_mode,
                replay_config,
                "action_mode",
                DEFAULT_ACTION_MODE,
            )
        ),
        object_type=str(
            _setting_value(
                args.object_type,
                replay_config,
                "object_type",
                DEFAULT_OBJECT_TYPE,
            )
        ),
        table_height=float(
            _setting_value(
                args.table_height,
                replay_config,
                "table_height",
                DEFAULT_TABLE_HEIGHT,
            )
        ),
        env_seed=_optional_int(
            _setting_value(args.env_seed, replay_config, "env_seed", None)
        ),
        render=args.render,
    )


def load_genome(path: str | Path) -> np.ndarray:
    """Load one flat genome from disk."""

    genome_path = Path(path)

    if not genome_path.exists():
        raise FileNotFoundError(f"genome file does not exist: {genome_path}")

    genome = np.asarray(np.load(genome_path), dtype=float)

    if genome.ndim != 1:
        raise ValueError("champion genome must be one flat array")

    if not np.all(np.isfinite(genome)):
        raise ValueError("champion genome must only contain finite numbers")

    return genome


def build_evaluator_config(settings: ReplaySettings) -> PickAndPlaceEvaluatorConfig:
    """Build the evaluator config used for replay."""

    network_shape = make_pick_and_place_network_shape(
        hidden_size=settings.hidden_size,
        action_mode=settings.action_mode,
    )

    return PickAndPlaceEvaluatorConfig(
        network_shape=network_shape,
        max_steps=settings.max_steps,
        action_mode=settings.action_mode,
    )


def build_robosuite_config(settings: ReplaySettings) -> RobosuitePickAndPlaceConfig:
    """Build the real Panda PickPlace simulator config."""

    env_kwargs = {
        "use_object_obs": True,
        "single_object_mode": 2,
        "object_type": settings.object_type,
        "horizon": settings.max_steps,
    }

    if settings.env_seed is not None:
        env_kwargs["seed"] = settings.env_seed

    return RobosuitePickAndPlaceConfig(
        has_renderer=settings.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_each_step=settings.render,
        table_height=settings.table_height,
        env_kwargs=env_kwargs,
    )


def validate_genome_length(
    genome: np.ndarray,
    evaluator_config: PickAndPlaceEvaluatorConfig,
) -> None:
    """Make sure the saved genome matches the requested network shape."""

    expected_length = evaluator_config.network_shape.genome_length

    if genome.size == expected_length:
        return

    raise ValueError(
        "champion genome length does not match replay network shape: "
        f"expected {expected_length}, got {genome.size}. "
        "Use the saved run_config.json, or pass the same --hidden-size and "
        "--action-mode as training."
    )


def format_replay_report(
    result: GenomeEvaluationResult,
    genome_path: str | Path,
) -> str:
    """Build a readable text report for one replay."""

    stages = result.fitness.stages

    return "\n".join(
        [
            "Champion replay complete",
            f"Genome: {Path(genome_path)}",
            f"Fitness: {result.fitness.total:.6f}",
            f"States: {len(result.states)}",
            f"Actions: {len(result.actions)}",
            "Stage scores:",
            f"  reaching: {stages.reaching:.6f}",
            f"  grasping: {stages.grasping:.6f}",
            f"  lifting: {stages.lifting:.6f}",
            f"  moving: {stages.moving:.6f}",
            f"  placing: {stages.placing:.6f}",
            f"  placement_accuracy: {stages.placement_accuracy:.6f}",
            f"  placement_stability: {stages.placement_stability:.6f}",
        ]
    )


def replay_champion(settings: ReplaySettings) -> GenomeEvaluationResult:
    """Load the champion genome and run one replay episode."""

    genome = load_genome(settings.champion_genome)
    evaluator_config = build_evaluator_config(settings)
    validate_genome_length(genome, evaluator_config)
    robosuite_config = build_robosuite_config(settings)
    simulator = make_robosuite_pick_and_place_simulator(robosuite_config)

    try:
        return evaluate_genome(
            genome=genome,
            simulator=simulator,
            config=evaluator_config,
        )
    finally:
        close_simulator(simulator)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = resolve_replay_settings(args)
    result = replay_champion(settings)
    report = format_replay_report(
        result=result,
        genome_path=settings.champion_genome,
    )

    print(report)

    return 0


def _resolve_run_config_path(
    args: argparse.Namespace,
    champion_genome: Path,
) -> Path | None:
    if args.run_config is not None:
        return Path(args.run_config)

    default_path = default_run_config_path(champion_genome)

    if default_path.exists():
        return default_path

    return None


def _load_optional_run_config(
    args: argparse.Namespace,
    run_config_path: Path | None,
) -> dict[str, Any]:
    if run_config_path is None:
        return {}

    if args.run_config is None and not run_config_path.exists():
        return {}

    return load_run_config(run_config_path)


def _replay_section(run_config: Mapping[str, Any]) -> Mapping[str, Any]:
    replay_config = run_config.get("replay", {})

    if not isinstance(replay_config, Mapping):
        raise ValueError("run config replay section must be a JSON object")

    return replay_config


def _setting_value(
    command_line_value: Any,
    replay_config: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    if command_line_value is not None:
        return command_line_value

    if key in replay_config:
        return replay_config[key]

    return default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
