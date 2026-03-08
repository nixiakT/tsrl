from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tsrl_lite.optimizer as optimizer_module
from tsrl_lite.optimizer import run_overnight_optimizer, run_overnight_watchdog


class OvernightOptimizerTest(unittest.TestCase):
    def test_run_overnight_optimizer_writes_state_and_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "optimizer_spec.json"
            output_dir = tmp_path / "optimizer_run"

            base_config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "optimizer_base",
                        "seed": 3,
                        "data": {
                            "source": "synthetic",
                            "window_size": 16,
                            "train_ratio": 0.75,
                            "val_ratio": 0.1,
                            "synthetic": {
                                "steps": 700,
                                "drift": 0.001,
                                "volatility": 0.007,
                                "seasonality": 0.002,
                            },
                        },
                        "env": {
                            "name": "trading-v0",
                            "reward_scale": 1.0,
                            "episode_horizon": 64,
                            "random_reset": True,
                            "params": {
                                "positions": [-1.0, 0.0, 1.0],
                                "trading_cost": 0.0005,
                            },
                        },
                        "encoder": {
                            "name": "returns-context-v0",
                            "params": {},
                        },
                        "agent": {
                            "name": "linear-ppo",
                            "gamma": 0.99,
                            "gae_lambda": 0.95,
                            "policy_lr": 0.03,
                            "value_lr": 0.05,
                            "gradient_clip": 5.0,
                            "params": {
                                "clip_epsilon": 0.2,
                                "update_epochs": 2,
                            },
                        },
                        "trainer": {
                            "episodes": 2,
                            "val_episodes": 1,
                            "eval_episodes": 1,
                            "log_interval": 2,
                            "eval_interval": 1,
                            "checkpoint_dir": str(tmp_path / "base_run"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec_path.write_text(
                json.dumps(
                    {
                        "name": "optimizer_smoke",
                        "base_config": "base_config.json",
                        "mode": "train",
                        "selection_metric": "mean_terminal_score",
                        "selection_mode": "max",
                        "seed": 13,
                        "population_size": 2,
                        "elite_size": 1,
                        "mutation_rate": 0.5,
                        "max_generations": 2,
                        "search_space": {
                            "encoder.name": [
                                "returns-context-v0",
                                "price-context-v0",
                            ],
                            "agent.params.clip_epsilon": [
                                0.1,
                                0.2,
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            artifacts, state = run_overnight_optimizer(spec_path=spec_path, output_dir=output_dir)

            self.assertTrue(artifacts.state_path.exists())
            self.assertTrue(artifacts.heartbeat_path.exists())
            self.assertTrue(artifacts.spec_copy_path.exists())
            self.assertTrue(artifacts.progress_csv_path.exists())
            self.assertTrue(artifacts.best_config_path.exists())
            self.assertTrue(artifacts.best_candidate_path.exists())
            self.assertEqual(state["status"], "max_generations_reached")
            self.assertEqual(state["generation"], 2)
            self.assertEqual(len(state["completed_generations"]), 2)
            self.assertIsNotNone(state["best_candidate"])
            self.assertEqual(state["active_generation"], None)
            self.assertEqual(state["active_candidates_total"], 0)
            with artifacts.progress_csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(state["evaluated_candidates"]))
            best_candidate = json.loads(artifacts.best_candidate_path.read_text(encoding="utf-8"))
            best_config = json.loads(artifacts.best_config_path.read_text(encoding="utf-8"))
            self.assertEqual(best_config["experiment_name"], best_candidate["experiment"])

    def test_run_overnight_optimizer_resumes_partial_generation_after_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "optimizer_spec.json"
            output_dir = tmp_path / "optimizer_resume_run"

            base_config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "optimizer_resume",
                        "seed": 5,
                        "data": {
                            "source": "synthetic",
                            "window_size": 16,
                            "train_ratio": 0.75,
                            "val_ratio": 0.1,
                            "synthetic": {
                                "steps": 700,
                                "drift": 0.001,
                                "volatility": 0.007,
                                "seasonality": 0.002,
                            },
                        },
                        "env": {
                            "name": "trading-v0",
                            "reward_scale": 1.0,
                            "episode_horizon": 64,
                            "random_reset": True,
                            "params": {
                                "positions": [-1.0, 0.0, 1.0],
                                "trading_cost": 0.0005,
                            },
                        },
                        "encoder": {
                            "name": "returns-context-v0",
                            "params": {},
                        },
                        "agent": {
                            "name": "linear-ppo",
                            "gamma": 0.99,
                            "gae_lambda": 0.95,
                            "policy_lr": 0.03,
                            "value_lr": 0.05,
                            "gradient_clip": 5.0,
                            "params": {
                                "clip_epsilon": 0.2,
                                "update_epochs": 2,
                            },
                        },
                        "trainer": {
                            "episodes": 2,
                            "val_episodes": 1,
                            "eval_episodes": 1,
                            "log_interval": 2,
                            "eval_interval": 1,
                            "checkpoint_dir": str(tmp_path / "resume_base_run"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec_path.write_text(
                json.dumps(
                    {
                        "name": "optimizer_resume_smoke",
                        "base_config": "base_config.json",
                        "mode": "train",
                        "selection_metric": "mean_terminal_score",
                        "selection_mode": "max",
                        "seed": 17,
                        "population_size": 2,
                        "elite_size": 1,
                        "mutation_rate": 0.5,
                        "max_generations": 1,
                        "search_space": {
                            "encoder.name": [
                                "returns-context-v0",
                                "price-context-v0",
                            ],
                            "agent.params.clip_epsilon": [
                                0.1,
                                0.2,
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            real_evaluate = optimizer_module._evaluate_study_item
            call_count = {"value": 0}

            def flaky_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
                call_count["value"] += 1
                if call_count["value"] == 2:
                    raise RuntimeError("simulated crash")
                return real_evaluate(*args, **kwargs)

            with patch("tsrl_lite.optimizer._evaluate_study_item", side_effect=flaky_evaluate):
                with self.assertRaises(RuntimeError):
                    run_overnight_optimizer(spec_path=spec_path, output_dir=output_dir)

            crashed_state = json.loads((output_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(crashed_state["status"], "running_generation")
            self.assertEqual(crashed_state["active_generation"], 1)
            self.assertEqual(crashed_state["active_candidates_completed"], 1)
            self.assertEqual(crashed_state["active_candidates_total"], 2)
            self.assertEqual(len(crashed_state["active_population"]), 2)
            self.assertEqual(len(crashed_state["active_generation_rows"]), 1)

            artifacts, resumed_state = run_overnight_optimizer(spec_path=spec_path, output_dir=output_dir)

            self.assertEqual(resumed_state["status"], "max_generations_reached")
            self.assertEqual(resumed_state["generation"], 1)
            self.assertEqual(len(resumed_state["completed_generations"]), 1)
            self.assertEqual(len(resumed_state["evaluated_candidates"]), 2)
            self.assertEqual(resumed_state["active_generation"], None)
            candidate_slots = sorted(
                (int(entry["generation"]), int(entry["candidate_index"]))
                for entry in resumed_state["evaluated_candidates"]
            )
            self.assertEqual(candidate_slots, [(1, 1), (1, 2)])
            with artifacts.progress_csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)

    def test_run_overnight_watchdog_restarts_after_optimizer_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "optimizer_spec.json"
            output_dir = tmp_path / "watchdog_run"

            base_config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "watchdog_base",
                        "seed": 7,
                        "data": {
                            "source": "synthetic",
                            "window_size": 16,
                            "train_ratio": 0.75,
                            "val_ratio": 0.1,
                            "synthetic": {
                                "steps": 700,
                                "drift": 0.001,
                                "volatility": 0.007,
                                "seasonality": 0.002,
                            },
                        },
                        "env": {
                            "name": "trading-v0",
                            "reward_scale": 1.0,
                            "episode_horizon": 64,
                            "random_reset": True,
                            "params": {
                                "positions": [-1.0, 0.0, 1.0],
                                "trading_cost": 0.0005,
                            },
                        },
                        "encoder": {
                            "name": "returns-context-v0",
                            "params": {},
                        },
                        "agent": {
                            "name": "linear-ppo",
                            "gamma": 0.99,
                            "gae_lambda": 0.95,
                            "policy_lr": 0.03,
                            "value_lr": 0.05,
                            "gradient_clip": 5.0,
                            "params": {
                                "clip_epsilon": 0.2,
                                "update_epochs": 2,
                            },
                        },
                        "trainer": {
                            "episodes": 2,
                            "val_episodes": 1,
                            "eval_episodes": 1,
                            "log_interval": 2,
                            "eval_interval": 1,
                            "checkpoint_dir": str(tmp_path / "watchdog_base_run"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec_path.write_text(
                json.dumps(
                    {
                        "name": "watchdog_smoke",
                        "base_config": "base_config.json",
                        "mode": "train",
                        "selection_metric": "mean_terminal_score",
                        "selection_mode": "max",
                        "seed": 19,
                        "population_size": 1,
                        "elite_size": 1,
                        "mutation_rate": 0.5,
                        "max_generations": 1,
                        "search_space": {
                            "encoder.name": ["returns-context-v0"],
                            "agent.params.clip_epsilon": [0.2],
                        },
                    }
                ),
                encoding="utf-8",
            )

            real_run = optimizer_module.run_overnight_optimizer
            call_count = {"value": 0}

            def flaky_run(*args: object, **kwargs: object) -> tuple[object, dict[str, object]]:
                call_count["value"] += 1
                if call_count["value"] == 1:
                    raise RuntimeError("simulated watchdog crash")
                return real_run(*args, **kwargs)

            with patch("tsrl_lite.optimizer.run_overnight_optimizer", side_effect=flaky_run):
                artifacts, state = run_overnight_watchdog(
                    spec_path=spec_path,
                    output_dir=output_dir,
                    max_restarts=2,
                    restart_delay_seconds=0.0,
                )

            self.assertTrue(artifacts.state_path.exists())
            self.assertTrue(artifacts.heartbeat_path.exists())
            self.assertEqual(state["status"], "max_generations_reached")
            self.assertEqual(state["restart_count"], 1)
            self.assertEqual(state["launch_count"], 2)
            self.assertIsNotNone(state["best_candidate"])

    def test_run_overnight_optimizer_prioritizes_feasible_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "optimizer_constraints.json"
            output_dir = tmp_path / "optimizer_constraints_run"

            base_config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "optimizer_constraints",
                        "seed": 3,
                        "data": {
                            "source": "synthetic",
                            "window_size": 16,
                            "train_ratio": 0.75,
                            "val_ratio": 0.1,
                            "synthetic": {
                                "steps": 700,
                                "drift": 0.001,
                                "volatility": 0.007,
                                "seasonality": 0.002,
                            },
                        },
                        "env": {
                            "name": "trading-v0",
                            "reward_scale": 1.0,
                            "episode_horizon": 64,
                            "random_reset": True,
                            "params": {
                                "positions": [-1.0, 0.0, 1.0],
                                "trading_cost": 0.0005,
                            },
                        },
                        "encoder": {
                            "name": "returns-context-v0",
                            "params": {},
                        },
                        "agent": {
                            "name": "linear-ppo",
                            "gamma": 0.99,
                            "gae_lambda": 0.95,
                            "policy_lr": 0.03,
                            "value_lr": 0.05,
                            "gradient_clip": 5.0,
                            "params": {
                                "clip_epsilon": 0.2,
                                "update_epochs": 2,
                            },
                        },
                        "trainer": {
                            "episodes": 2,
                            "val_episodes": 1,
                            "eval_episodes": 1,
                            "log_interval": 2,
                            "eval_interval": 1,
                            "checkpoint_dir": str(tmp_path / "constraints_base_run"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec_path.write_text(
                json.dumps(
                    {
                        "name": "optimizer_constraints_smoke",
                        "base_config": "base_config.json",
                        "mode": "train",
                        "selection_metric": "mean_terminal_score",
                        "selection_mode": "max",
                        "metric_constraints": {
                            "mean_max_drawdown": {"max": 0.3}
                        },
                        "seed": 17,
                        "population_size": 2,
                        "elite_size": 1,
                        "mutation_rate": 0.5,
                        "max_generations": 1,
                        "search_space": {
                            "encoder.name": [
                                "returns-context-v0",
                                "price-context-v0",
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_evaluate(*args: object, **kwargs: object) -> dict[str, object]:
                candidate_index = int(kwargs["index"])
                if candidate_index == 1:
                    return {
                        "generation": 1,
                        "candidate_index": candidate_index,
                        "experiment": "candidate_infeasible",
                        "config_path": str(kwargs["item"]["config_path"]),
                        "mode": "train",
                        "output_dir": str(Path(kwargs["study_root"]) / "candidate_infeasible"),
                        "summary_path": str(Path(kwargs["study_root"]) / "candidate_infeasible" / "summary.json"),
                        "env": "trading-v0",
                        "encoder": "returns-context-v0",
                        "agent": "linear-ppo",
                        "selection_metric": "mean_terminal_score",
                        "selection_value": 2.0,
                        "constraint_feasible": False,
                        "constraint_violation_score": 1.0,
                        "constraint_violation_count": 1,
                        "constraint_violations": ["mean_max_drawdown>0.3"],
                        "report_mean_terminal_score": 2.0,
                        "report_mean_max_drawdown": 0.6,
                    }
                return {
                    "generation": 1,
                    "candidate_index": candidate_index,
                    "experiment": "candidate_feasible",
                    "config_path": str(kwargs["item"]["config_path"]),
                    "mode": "train",
                    "output_dir": str(Path(kwargs["study_root"]) / "candidate_feasible"),
                    "summary_path": str(Path(kwargs["study_root"]) / "candidate_feasible" / "summary.json"),
                    "env": "trading-v0",
                    "encoder": "price-context-v0",
                    "agent": "linear-ppo",
                    "selection_metric": "mean_terminal_score",
                    "selection_value": 1.0,
                    "constraint_feasible": True,
                    "constraint_violation_score": 0.0,
                    "constraint_violation_count": 0,
                    "constraint_violations": [],
                    "report_mean_terminal_score": 1.0,
                    "report_mean_max_drawdown": 0.2,
                }

            with patch("tsrl_lite.optimizer._evaluate_study_item", side_effect=fake_evaluate):
                artifacts, state = run_overnight_optimizer(spec_path=spec_path, output_dir=output_dir)

            self.assertEqual(state["status"], "max_generations_reached")
            self.assertIsNotNone(state["best_candidate"])
            self.assertTrue(state["best_candidate"]["constraint_feasible"])
            self.assertEqual(state["best_candidate"]["experiment"], "candidate_feasible")
            with artifacts.progress_csv_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("True", {row["constraint_feasible"] for row in rows})

    def test_run_overnight_optimizer_supports_matrix_method_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            matrix_spec_path = tmp_path / "matrix_spec.json"
            optimizer_spec_path = tmp_path / "matrix_optimizer_spec.json"
            output_dir = tmp_path / "matrix_optimizer_run"

            matrix_spec_path.write_text(
                json.dumps(
                    {
                        "matrix_selection_metric": "train.approx_kl",
                        "matrix_selection_mode": "min",
                        "tasks": [{"name": "task_alpha", "config": "placeholder.json", "report_metrics": ["train.approx_kl"]}],
                        "methods": [
                            {"name": "linear_default"},
                            {
                                "name": "gru_sequence",
                                "overrides": {
                                    "agent.policy_lr": 0.001,
                                    "agent.params.hidden_size": 32,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            optimizer_spec_path.write_text(
                json.dumps(
                    {
                        "name": "matrix_optimizer_smoke",
                        "base_matrix_spec": "matrix_spec.json",
                        "target_method": "gru_sequence",
                        "selection_metric": "train.approx_kl",
                        "selection_mode": "min",
                        "population_size": 2,
                        "elite_size": 1,
                        "mutation_rate": 0.5,
                        "max_generations": 1,
                        "seed": 5,
                        "search_space": {
                            "agent.params.hidden_size": [32, 64],
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_matrix(
                spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
                resume: bool = True,
            ) -> tuple[object, dict]:
                generated_spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
                gru_method = next(
                    method for method in generated_spec["methods"] if method["name"] == "gru_sequence"
                )
                overrides = gru_method["overrides"]
                hidden_size = int(overrides["agent"]["params"]["hidden_size"])
                metric_value = 0.01 if hidden_size == 64 else 0.08
                run_dir = Path(output_dir or tmp_path / "matrix_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "matrix_summary.json"
                summary_path.write_text("{}", encoding="utf-8")
                summary = {
                    "leaderboard": [
                        {
                            "rank": 1,
                            "method": "gru_sequence",
                            "task_weight_total": 1.0,
                            "feasible_task_weight": 1.0,
                            "mean_rank": 1.0,
                            "mean_normalized_rank_score": 1.0,
                            "wins": 1,
                            "matrix_selection_value": metric_value,
                            "matrix_constraint_feasible": True,
                            "matrix_constraint_violation_score": 0.0,
                            "matrix_constraint_violation_count": 0,
                        },
                        {
                            "rank": 2,
                            "method": "linear_default",
                            "task_weight_total": 1.0,
                            "feasible_task_weight": 1.0,
                            "mean_rank": 2.0,
                            "mean_normalized_rank_score": 0.0,
                            "wins": 0,
                            "matrix_selection_value": 0.5,
                            "matrix_constraint_feasible": True,
                            "matrix_constraint_violation_score": 0.0,
                            "matrix_constraint_violation_count": 0,
                        },
                    ],
                    "matrix_report_metrics": ["train.approx_kl"],
                    "method_metric_report": [
                        {
                            "method": "gru_sequence",
                            "metric_train_approx_kl_weighted_mean": metric_value,
                            "metric_train_approx_kl_task_weight": 1.0,
                            "metric_train_approx_kl_weighted_coverage_ratio": 1.0,
                        },
                        {
                            "method": "linear_default",
                            "metric_train_approx_kl_weighted_mean": 0.2,
                            "metric_train_approx_kl_task_weight": 1.0,
                            "metric_train_approx_kl_weighted_coverage_ratio": 1.0,
                        },
                    ],
                }
                return SimpleNamespace(summary_path=summary_path), summary

            with patch("tsrl_lite.optimizer.run_benchmark_matrix_spec", side_effect=fake_run_matrix):
                artifacts, state = run_overnight_optimizer(spec_path=optimizer_spec_path, output_dir=output_dir)

            self.assertEqual(state["optimizer_kind"], "matrix_method")
            self.assertEqual(state["status"], "max_generations_reached")
            self.assertIsNotNone(state["best_candidate"])
            self.assertTrue(artifacts.best_config_path.exists())
            best_spec = json.loads(artifacts.best_config_path.read_text(encoding="utf-8"))
            best_method = next(method for method in best_spec["methods"] if method["name"] == "gru_sequence")
            self.assertEqual(best_method["overrides"]["agent"]["params"]["hidden_size"], 64)
            self.assertAlmostEqual(float(state["best_candidate"]["selection_value"]), 0.01)

    def test_run_overnight_optimizer_matrix_mode_prioritizes_feasible_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            matrix_spec_path = tmp_path / "matrix_spec.json"
            optimizer_spec_path = tmp_path / "matrix_optimizer_spec.json"
            output_dir = tmp_path / "matrix_optimizer_constraints_run"

            matrix_spec_path.write_text(
                json.dumps(
                    {
                        "tasks": [{"name": "task_alpha", "config": "placeholder.json", "report_metrics": ["train.approx_kl"]}],
                        "methods": [
                            {"name": "linear_default"},
                            {"name": "gru_sequence", "overrides": {"agent.params.hidden_size": 32}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            optimizer_spec_path.write_text(
                json.dumps(
                    {
                        "name": "matrix_optimizer_constraints_smoke",
                        "base_matrix_spec": "matrix_spec.json",
                        "target_method": "gru_sequence",
                        "selection_metric": "train.approx_kl",
                        "selection_mode": "min",
                        "population_size": 2,
                        "elite_size": 1,
                        "mutation_rate": 0.5,
                        "max_generations": 1,
                        "seed": 11,
                        "metric_constraints": {
                            "train.approx_kl": {"max": 0.05}
                        },
                        "search_space": {
                            "agent.params.hidden_size": [32, 64]
                        },
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_matrix(
                spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
                resume: bool = True,
            ) -> tuple[object, dict]:
                generated_spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
                gru_method = next(
                    method for method in generated_spec["methods"] if method["name"] == "gru_sequence"
                )
                hidden_size = int(gru_method["overrides"]["agent"]["params"]["hidden_size"])
                metric_value = 0.08 if hidden_size == 32 else 0.02
                feasible = hidden_size != 32
                run_dir = Path(output_dir or tmp_path / "matrix_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "matrix_summary.json"
                summary_path.write_text("{}", encoding="utf-8")
                summary = {
                    "leaderboard": [
                        {
                            "rank": 1,
                            "method": "gru_sequence",
                            "task_weight_total": 1.0,
                            "feasible_task_weight": 1.0,
                            "mean_rank": 1.0,
                            "mean_normalized_rank_score": 1.0,
                            "wins": 1,
                            "matrix_selection_value": metric_value,
                            "matrix_constraint_feasible": feasible,
                            "matrix_constraint_violation_score": 1.0 if not feasible else 0.0,
                            "matrix_constraint_violation_count": 1 if not feasible else 0,
                            "matrix_constraint_violations": ["train.approx_kl>0.05"] if not feasible else [],
                        }
                    ],
                    "matrix_report_metrics": ["train.approx_kl"],
                    "method_metric_report": [
                        {
                            "method": "gru_sequence",
                            "metric_train_approx_kl_weighted_mean": metric_value,
                            "metric_train_approx_kl_task_weight": 1.0,
                            "metric_train_approx_kl_weighted_coverage_ratio": 1.0,
                        }
                    ],
                }
                return SimpleNamespace(summary_path=summary_path), summary

            with patch("tsrl_lite.optimizer.run_benchmark_matrix_spec", side_effect=fake_run_matrix):
                _, state = run_overnight_optimizer(spec_path=optimizer_spec_path, output_dir=output_dir)

            self.assertEqual(state["status"], "max_generations_reached")
            self.assertTrue(state["best_candidate"]["constraint_feasible"])
            self.assertEqual(state["best_candidate"]["params"]["agent.params.hidden_size"], 64)


if __name__ == "__main__":
    unittest.main()
