import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_neuro_training.evaluator import GenomeEvaluationResult
from rl_neuro_training.fitness import (
    PickAndPlaceFitnessResult,
    PickAndPlaceStageScores,
)
from scripts.replay_champion import (
    ReplaySettings,
    build_evaluator_config,
    build_robosuite_config,
    default_run_config_path,
    format_replay_report,
    load_genome,
    load_run_config,
    parse_args,
    resolve_replay_settings,
    validate_genome_length,
)


def make_replay_result():
    stages = PickAndPlaceStageScores(
        reaching=1.0,
        grasping=0.8,
        lifting=0.6,
        moving=0.4,
        placing=0.2,
        placement_accuracy=0.2,
        placement_stability=0.1,
    )

    return GenomeEvaluationResult(
        fitness=PickAndPlaceFitnessResult(total=0.7, stages=stages),
        states=(object(), object()),
        actions=(np.array([0.0]),),
    )


class ReplayChampionScriptTest(unittest.TestCase):
    def make_settings(self, **overrides):
        values = {
            "champion_genome": Path("champion_genome.npy"),
            "run_config": None,
            "hidden_size": 32,
            "max_steps": 50,
            "action_mode": "end_effector",
            "object_type": "can",
            "table_height": 0.8,
            "env_seed": None,
            "render": False,
        }
        values.update(overrides)

        return ReplaySettings(**values)

    def test_load_genome_reads_flat_numpy_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            genome_path = Path(temp_dir) / "champion_genome.npy"
            np.save(genome_path, np.array([0.1, 0.2, 0.3]))

            genome = load_genome(genome_path)

        np.testing.assert_array_equal(genome, np.array([0.1, 0.2, 0.3]))

    def test_load_genome_rejects_non_flat_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            genome_path = Path(temp_dir) / "bad_genome.npy"
            np.save(genome_path, np.zeros((2, 2)))

            with self.assertRaisesRegex(ValueError, "flat"):
                load_genome(genome_path)

    def test_default_run_config_path_lives_next_to_champion_genome(self):
        path = default_run_config_path("runs/pick_place/champion_genome.npy")

        self.assertEqual(path, Path("runs/pick_place/run_config.json"))

    def test_load_run_config_reads_json_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "run_config.json"
            config_path.write_text(
                json.dumps({"replay": {"hidden_size": 4}}),
                encoding="utf-8",
            )

            config = load_run_config(config_path)

        self.assertEqual(config["replay"]["hidden_size"], 4)

    def test_resolve_replay_settings_uses_sibling_run_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            champion_path = Path(temp_dir) / "champion_genome.npy"
            champion_path.write_bytes(b"placeholder")
            config_path = Path(temp_dir) / "run_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "replay": {
                            "hidden_size": 4,
                            "max_steps": 7,
                            "action_mode": "joint",
                            "object_type": "milk",
                            "table_height": 0.9,
                            "env_seed": 123,
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(["--champion-genome", str(champion_path)])

            settings = resolve_replay_settings(args)

        self.assertEqual(settings.run_config, config_path)
        self.assertEqual(settings.hidden_size, 4)
        self.assertEqual(settings.max_steps, 7)
        self.assertEqual(settings.action_mode, "joint")
        self.assertEqual(settings.object_type, "milk")
        self.assertEqual(settings.table_height, 0.9)
        self.assertEqual(settings.env_seed, 123)

    def test_resolve_replay_settings_prefers_command_line_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            champion_path = Path(temp_dir) / "champion_genome.npy"
            champion_path.write_bytes(b"placeholder")
            config_path = Path(temp_dir) / "run_config.json"
            config_path.write_text(
                json.dumps({"replay": {"hidden_size": 4, "max_steps": 7}}),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--champion-genome",
                    str(champion_path),
                    "--hidden-size",
                    "8",
                    "--max-steps",
                    "3",
                ]
            )

            settings = resolve_replay_settings(args)

        self.assertEqual(settings.hidden_size, 8)
        self.assertEqual(settings.max_steps, 3)

    def test_resolve_replay_settings_rejects_missing_explicit_run_config(self):
        args = parse_args(
            [
                "--champion-genome",
                "champion_genome.npy",
                "--run-config",
                "missing_run_config.json",
            ]
        )

        with self.assertRaisesRegex(FileNotFoundError, "run config"):
            resolve_replay_settings(args)

    def test_build_evaluator_config_uses_requested_network_shape(self):
        settings = self.make_settings(
            hidden_size=4,
            max_steps=7,
        )

        config = build_evaluator_config(settings)

        self.assertEqual(config.network_shape.hidden_size, 4)
        self.assertEqual(config.max_steps, 7)

    def test_build_robosuite_config_uses_replay_options(self):
        settings = self.make_settings(
            object_type="milk",
            max_steps=9,
            env_seed=123,
            render=True,
        )

        config = build_robosuite_config(settings)

        self.assertTrue(config.has_renderer)
        self.assertTrue(config.render_each_step)
        self.assertEqual(config.env_kwargs["object_type"], "milk")
        self.assertEqual(config.env_kwargs["horizon"], 9)
        self.assertEqual(config.env_kwargs["seed"], 123)

    def test_validate_genome_length_accepts_matching_length(self):
        settings = self.make_settings(hidden_size=2)
        config = build_evaluator_config(settings)
        genome = np.zeros(config.network_shape.genome_length)

        validate_genome_length(genome, config)

    def test_validate_genome_length_rejects_wrong_length(self):
        settings = self.make_settings(hidden_size=2)
        config = build_evaluator_config(settings)

        with self.assertRaisesRegex(ValueError, "run_config"):
            validate_genome_length(np.zeros(3), config)

    def test_format_replay_report_includes_fitness_and_stage_scores(self):
        report = format_replay_report(
            result=make_replay_result(),
            genome_path="champion_genome.npy",
        )

        self.assertIn("Champion replay complete", report)
        self.assertIn("Fitness: 0.700000", report)
        self.assertIn("reaching: 1.000000", report)
        self.assertIn("placement_stability: 0.100000", report)


if __name__ == "__main__":
    unittest.main()
