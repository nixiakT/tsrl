from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.matrix import run_benchmark_matrix_spec
from tsrl_lite.study import StudyArtifacts


class MatrixTest(unittest.TestCase):
    def test_run_benchmark_matrix_spec_resumes_unchanged_task_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "task.json"
            spec_path = tmp_path / "matrix_spec.json"
            matrix_dir = tmp_path / "matrix_run"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "task_base",
                        "seed": 3,
                        "data": {
                            "source": "synthetic",
                            "window_size": 16,
                            "train_ratio": 0.75,
                            "val_ratio": 0.1,
                            "synthetic": {
                                "steps": 500,
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
                        "encoder": {"name": "returns-context-v0", "params": {}},
                        "agent": {
                            "name": "linear-ppo",
                            "gamma": 0.99,
                            "gae_lambda": 0.95,
                            "policy_lr": 0.03,
                            "value_lr": 0.05,
                            "gradient_clip": 5.0,
                            "params": {"clip_epsilon": 0.2, "update_epochs": 2},
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
                        "tasks": [{"name": "resume_task", "config": "task.json"}],
                        "methods": [{"name": "method_alpha", "overrides": {"encoder.name": "returns-context-v0"}}],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_study_spec(
                generated_spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
            ) -> tuple[StudyArtifacts, dict]:
                run_dir = Path(output_dir or tmp_path / "study_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "study_summary.json"
                leaderboard_csv_path = run_dir / "leaderboard.csv"
                leaderboard_md_path = run_dir / "leaderboard.md"
                summary = {
                    "mode": "train",
                    "selection_metric": "mean_terminal_score",
                    "selection_mode": "max",
                    "run_count": 1,
                    "top_experiment": "method_alpha",
                    "top_feasible_experiment": "method_alpha",
                    "feasible_run_count": 1,
                    "infeasible_run_count": 0,
                    "leaderboard": [
                        {
                            "rank": 1,
                            "experiment": "method_alpha",
                            "selection_value": 1.0,
                            "constraint_feasible": True,
                            "summary_path": str(run_dir / "method_alpha" / "summary.json"),
                        }
                    ],
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                leaderboard_csv_path.write_text("", encoding="utf-8")
                leaderboard_md_path.write_text("", encoding="utf-8")
                return (
                    StudyArtifacts(
                        root_dir=run_dir,
                        summary_path=summary_path,
                        leaderboard_csv_path=leaderboard_csv_path,
                        leaderboard_md_path=leaderboard_md_path,
                    ),
                    summary,
                )

            with patch("tsrl_lite.matrix.run_study_spec", side_effect=fake_run_study_spec):
                _, first_summary = run_benchmark_matrix_spec(spec_path, output_dir=matrix_dir)
            self.assertEqual(first_summary["executed_task_count"], 1)
            self.assertEqual(first_summary["resumed_task_count"], 0)

            with patch("tsrl_lite.matrix.run_study_spec", side_effect=AssertionError("should reuse previous task")):
                _, second_summary = run_benchmark_matrix_spec(spec_path, output_dir=matrix_dir)

            self.assertEqual(second_summary["executed_task_count"], 0)
            self.assertEqual(second_summary["resumed_task_count"], 1)
            self.assertEqual(second_summary["tasks"][0]["task_status"], "resumed")
            self.assertTrue((matrix_dir / "pairwise_wins.csv").exists())
            self.assertTrue((matrix_dir / "pairwise_wins.md").exists())

    def test_run_benchmark_matrix_spec_respects_task_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trading_config = tmp_path / "trading.json"
            regime_config = tmp_path / "regime.json"
            spec_path = tmp_path / "matrix_spec.json"
            matrix_dir = tmp_path / "matrix_run"

            base_payload = {
                "experiment_name": "base_task",
                "seed": 7,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.75,
                    "val_ratio": 0.1,
                    "synthetic": {
                        "steps": 600,
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
                "encoder": {"name": "returns-context-v0", "params": {}},
                "agent": {
                    "name": "linear-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.03,
                    "value_lr": 0.05,
                    "gradient_clip": 5.0,
                    "params": {"clip_epsilon": 0.2, "update_epochs": 2},
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
            trading_config.write_text(json.dumps({**base_payload, "experiment_name": "trading_task"}), encoding="utf-8")
            regime_config.write_text(json.dumps({**base_payload, "experiment_name": "regime_task"}), encoding="utf-8")
            spec_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {"name": "trading_suite", "config": "trading.json", "task_weight": 3.0},
                            {"name": "regime_suite", "config": "regime.json", "task_weight": 1.0},
                        ],
                        "methods": [
                            {"name": "method_alpha", "overrides": {"encoder.name": "returns-context-v0"}},
                            {"name": "method_beta", "overrides": {"encoder.name": "price-context-v0"}},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_study_spec(
                generated_spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
            ) -> tuple[StudyArtifacts, dict]:
                spec_payload = json.loads(Path(generated_spec_path).read_text(encoding="utf-8"))
                run_dir = Path(output_dir or tmp_path / "study_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "study_summary.json"
                leaderboard_csv_path = run_dir / "leaderboard.csv"
                leaderboard_md_path = run_dir / "leaderboard.md"
                summary_path.write_text("{}", encoding="utf-8")
                leaderboard_csv_path.write_text("", encoding="utf-8")
                leaderboard_md_path.write_text("", encoding="utf-8")

                task_base = Path(spec_payload["base_config"]).stem
                if "trading_suite" in task_base:
                    ordered = [("method_alpha", 1.0), ("method_beta", 0.4)]
                else:
                    ordered = [("method_beta", 1.0), ("method_alpha", 0.4)]

                summary = {
                    "mode": "train",
                    "selection_metric": "mean_terminal_score",
                    "selection_mode": "max",
                    "run_count": 2,
                    "top_experiment": ordered[0][0],
                    "top_feasible_experiment": ordered[0][0],
                    "feasible_run_count": 2,
                    "infeasible_run_count": 0,
                    "leaderboard": [
                        {
                            "rank": index,
                            "experiment": method_name,
                            "selection_value": value,
                            "constraint_feasible": True,
                            "summary_path": str(run_dir / method_name / "summary.json"),
                        }
                        for index, (method_name, value) in enumerate(ordered, start=1)
                    ],
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return (
                    StudyArtifacts(
                        root_dir=run_dir,
                        summary_path=summary_path,
                        leaderboard_csv_path=leaderboard_csv_path,
                        leaderboard_md_path=leaderboard_md_path,
                    ),
                    summary,
                )

            with patch("tsrl_lite.matrix.run_study_spec", side_effect=fake_run_study_spec):
                _, summary = run_benchmark_matrix_spec(spec_path, output_dir=matrix_dir)

            self.assertEqual(summary["total_task_weight"], 4.0)
            self.assertEqual(summary["top_method"], "method_alpha")
            self.assertAlmostEqual(summary["leaderboard"][0]["task_weight_total"], 4.0)
            self.assertGreater(
                float(summary["leaderboard"][0]["mean_normalized_rank_score"]),
                float(summary["leaderboard"][1]["mean_normalized_rank_score"]),
            )
            pairwise = [
                row for row in summary["pairwise_comparisons"]
                if row["method"] == "method_alpha" and row["opponent"] == "method_beta"
            ][0]
            self.assertEqual(pairwise["wins"], 1)
            self.assertEqual(pairwise["losses"], 1)
            self.assertAlmostEqual(float(pairwise["weighted_net"]), 2.0)

    def test_run_benchmark_matrix_spec_writes_reports_and_aggregates_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trading_config = tmp_path / "trading.json"
            regime_config = tmp_path / "regime.json"
            spec_path = tmp_path / "matrix_spec.json"
            matrix_dir = tmp_path / "matrix_run"

            base_payload = {
                "experiment_name": "base_task",
                "seed": 7,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.75,
                    "val_ratio": 0.1,
                    "synthetic": {
                        "steps": 600,
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
                "encoder": {"name": "returns-context-v0", "params": {}},
                "agent": {
                    "name": "linear-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.03,
                    "value_lr": 0.05,
                    "gradient_clip": 5.0,
                    "params": {"clip_epsilon": 0.2, "update_epochs": 2},
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
            trading_config.write_text(json.dumps({**base_payload, "experiment_name": "trading_task"}), encoding="utf-8")
            regime_config.write_text(
                json.dumps(
                    {
                        **base_payload,
                        "experiment_name": "regime_task",
                        "env": {
                            "name": "regime-v0",
                            "reward_scale": 1.0,
                            "episode_horizon": 64,
                            "random_reset": True,
                            "params": {
                                "forecast_horizon": 4,
                                "regime_threshold": 0.002,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "name": "trading_suite",
                                "config": "trading.json",
                                "selection_metric": "mean_sharpe_ratio",
                                "report_metrics": ["train.approx_kl"],
                            },
                            {
                                "name": "regime_suite",
                                "config": "regime.json",
                                "selection_metric": "mean_info_accuracy",
                                "report_metrics": ["train.approx_kl"],
                            },
                        ],
                        "methods": [
                            {
                                "name": "method_alpha",
                                "tags": ["alpha"],
                                "overrides": {"encoder.name": "returns-context-v0"},
                            },
                            {
                                "name": "method_beta",
                                "tags": ["beta"],
                                "overrides": {"encoder.name": "price-context-v0"},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_study_spec(
                generated_spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
            ) -> tuple[StudyArtifacts, dict]:
                spec_payload = json.loads(Path(generated_spec_path).read_text(encoding="utf-8"))
                run_dir = Path(output_dir or tmp_path / "study_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "study_summary.json"
                leaderboard_csv_path = run_dir / "leaderboard.csv"
                leaderboard_md_path = run_dir / "leaderboard.md"
                metrics_csv_path = run_dir / "metrics_report.csv"
                metrics_md_path = run_dir / "metrics_report.md"
                pareto_frontier_path = run_dir / "pareto_frontier.json"
                pareto_frontier_md_path = run_dir / "pareto_frontier.md"
                for path in (
                    leaderboard_csv_path,
                    leaderboard_md_path,
                    metrics_csv_path,
                    metrics_md_path,
                    pareto_frontier_path,
                    pareto_frontier_md_path,
                ):
                    path.write_text("", encoding="utf-8")

                task_base = Path(spec_payload["base_config"]).stem
                experiments = [experiment["name"] for experiment in spec_payload["experiments"]]
                self.assertEqual(experiments, ["method_alpha", "method_beta"])
                self.assertTrue(any(tag.startswith("task:") for tag in spec_payload["experiments"][0]["tags"]))

                if "trading_suite" in task_base:
                    ordered = [("method_alpha", 1.4), ("method_beta", 0.8)]
                    report_values = {"method_alpha": 0.03, "method_beta": 0.08}
                else:
                    ordered = [("method_alpha", 0.72), ("method_beta", 0.65)]
                    report_values = {"method_alpha": 0.05, "method_beta": 0.02}

                leaderboard = [
                    {
                        "rank": index,
                        "experiment": method_name,
                        "selection_value": metric_value,
                        "constraint_feasible": True,
                        "report_train_approx_kl": report_values[method_name],
                        "summary_path": str(run_dir / method_name / "summary.json"),
                    }
                    for index, (method_name, metric_value) in enumerate(ordered, start=1)
                ]
                summary = {
                    "mode": spec_payload["mode"],
                    "selection_metric": spec_payload["selection_metric"],
                    "selection_mode": spec_payload["selection_mode"],
                    "report_metrics": spec_payload.get("report_metrics", []),
                    "run_count": len(leaderboard),
                    "top_experiment": ordered[0][0],
                    "top_feasible_experiment": ordered[0][0],
                    "feasible_run_count": len(leaderboard),
                    "infeasible_run_count": 0,
                    "leaderboard": leaderboard,
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return (
                    StudyArtifacts(
                        root_dir=run_dir,
                        summary_path=summary_path,
                        leaderboard_csv_path=leaderboard_csv_path,
                        leaderboard_md_path=leaderboard_md_path,
                        metrics_csv_path=metrics_csv_path,
                        metrics_md_path=metrics_md_path,
                        pareto_frontier_path=pareto_frontier_path,
                        pareto_frontier_md_path=pareto_frontier_md_path,
                    ),
                    summary,
                )

            with patch("tsrl_lite.matrix.run_study_spec", side_effect=fake_run_study_spec):
                artifacts, summary = run_benchmark_matrix_spec(spec_path, output_dir=matrix_dir)

            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.leaderboard_csv_path.exists())
            self.assertTrue(artifacts.leaderboard_md_path.exists())
            self.assertTrue(artifacts.task_matrix_csv_path.exists())
            self.assertTrue(artifacts.task_matrix_md_path.exists())
            self.assertTrue(artifacts.pairwise_csv_path.exists())
            self.assertTrue(artifacts.pairwise_md_path.exists())
            self.assertIsNotNone(artifacts.method_metrics_csv_path)
            self.assertIsNotNone(artifacts.method_metrics_md_path)
            self.assertTrue(artifacts.method_metrics_csv_path.exists())
            self.assertTrue(artifacts.method_metrics_md_path.exists())
            self.assertEqual(summary["task_count"], 2)
            self.assertEqual(summary["generated_task_variants"], 2)
            self.assertEqual(summary["top_method"], "method_alpha")
            self.assertEqual(summary["matrix_report_metrics"], ["train.approx_kl"])
            self.assertEqual(summary["leaderboard"][0]["rank"], 1)
            self.assertEqual(summary["leaderboard"][0]["method"], "method_alpha")
            self.assertEqual(summary["leaderboard"][0]["wins"], 2)
            self.assertEqual(len(summary["tasks"]), 2)
            task_matrix_md = artifacts.task_matrix_md_path.read_text(encoding="utf-8")
            self.assertIn("trading_suite", task_matrix_md)
            self.assertIn("regime_suite", task_matrix_md)
            pairwise_md = artifacts.pairwise_md_path.read_text(encoding="utf-8")
            self.assertIn("method_alpha", pairwise_md)
            self.assertIn("2-0-0", pairwise_md)
            method_metrics_md = artifacts.method_metrics_md_path.read_text(encoding="utf-8")
            self.assertIn("train.approx_kl", method_metrics_md)

    def test_run_benchmark_matrix_spec_aggregates_shared_report_metrics_by_task_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trading_config = tmp_path / "trading.json"
            regime_config = tmp_path / "regime.json"
            spec_path = tmp_path / "matrix_spec.json"
            matrix_dir = tmp_path / "matrix_run"

            base_payload = {
                "experiment_name": "base_task",
                "seed": 7,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.75,
                    "val_ratio": 0.1,
                    "synthetic": {
                        "steps": 600,
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
                "encoder": {"name": "returns-context-v0", "params": {}},
                "agent": {
                    "name": "linear-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.03,
                    "value_lr": 0.05,
                    "gradient_clip": 5.0,
                    "params": {"clip_epsilon": 0.2, "update_epochs": 2},
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
            trading_config.write_text(json.dumps({**base_payload, "experiment_name": "trading_task"}), encoding="utf-8")
            regime_config.write_text(json.dumps({**base_payload, "experiment_name": "regime_task"}), encoding="utf-8")
            spec_path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "name": "trading_suite",
                                "config": "trading.json",
                                "task_weight": 3.0,
                                "report_metrics": ["train.approx_kl"],
                            },
                            {
                                "name": "regime_suite",
                                "config": "regime.json",
                                "task_weight": 1.0,
                                "report_metrics": ["train.approx_kl"],
                            },
                        ],
                        "methods": [
                            {"name": "method_alpha", "overrides": {"encoder.name": "returns-context-v0"}},
                            {"name": "method_beta", "overrides": {"encoder.name": "price-context-v0"}},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_study_spec(
                generated_spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
            ) -> tuple[StudyArtifacts, dict]:
                spec_payload = json.loads(Path(generated_spec_path).read_text(encoding="utf-8"))
                run_dir = Path(output_dir or tmp_path / "study_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "study_summary.json"
                leaderboard_csv_path = run_dir / "leaderboard.csv"
                leaderboard_md_path = run_dir / "leaderboard.md"
                summary_path.write_text("{}", encoding="utf-8")
                leaderboard_csv_path.write_text("", encoding="utf-8")
                leaderboard_md_path.write_text("", encoding="utf-8")

                task_base = Path(spec_payload["base_config"]).stem
                if "trading_suite" in task_base:
                    ordered = [("method_alpha", 1.0), ("method_beta", 0.5)]
                    report_values = {"method_alpha": 0.03, "method_beta": 0.08}
                else:
                    ordered = [("method_beta", 1.0), ("method_alpha", 0.5)]
                    report_values = {"method_alpha": 0.05, "method_beta": 0.02}

                summary = {
                    "mode": "train",
                    "selection_metric": "mean_terminal_score",
                    "selection_mode": "max",
                    "report_metrics": spec_payload.get("report_metrics", []),
                    "run_count": 2,
                    "top_experiment": ordered[0][0],
                    "top_feasible_experiment": ordered[0][0],
                    "feasible_run_count": 2,
                    "infeasible_run_count": 0,
                    "leaderboard": [
                        {
                            "rank": index,
                            "experiment": method_name,
                            "selection_value": value,
                            "constraint_feasible": True,
                            "report_train_approx_kl": report_values[method_name],
                            "summary_path": str(run_dir / method_name / "summary.json"),
                        }
                        for index, (method_name, value) in enumerate(ordered, start=1)
                    ],
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return (
                    StudyArtifacts(
                        root_dir=run_dir,
                        summary_path=summary_path,
                        leaderboard_csv_path=leaderboard_csv_path,
                        leaderboard_md_path=leaderboard_md_path,
                    ),
                    summary,
                )

            with patch("tsrl_lite.matrix.run_study_spec", side_effect=fake_run_study_spec):
                artifacts, summary = run_benchmark_matrix_spec(spec_path, output_dir=matrix_dir)

            self.assertIsNotNone(artifacts.method_metrics_csv_path)
            self.assertTrue(artifacts.method_metrics_csv_path.exists())
            alpha_row = [row for row in summary["method_metric_report"] if row["method"] == "method_alpha"][0]
            beta_row = [row for row in summary["method_metric_report"] if row["method"] == "method_beta"][0]
            self.assertAlmostEqual(float(alpha_row["metric_train_approx_kl_weighted_mean"]), 0.035)
            self.assertAlmostEqual(float(beta_row["metric_train_approx_kl_weighted_mean"]), 0.065)
            self.assertAlmostEqual(float(alpha_row["metric_train_approx_kl_task_weight"]), 4.0)
            self.assertAlmostEqual(float(beta_row["metric_train_approx_kl_weighted_coverage_ratio"]), 1.0)

    def test_run_benchmark_matrix_spec_supports_matrix_selection_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trading_config = tmp_path / "trading.json"
            regime_config = tmp_path / "regime.json"
            spec_path = tmp_path / "matrix_spec.json"
            matrix_dir = tmp_path / "matrix_run"

            base_payload = {
                "experiment_name": "base_task",
                "seed": 7,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.75,
                    "val_ratio": 0.1,
                    "synthetic": {
                        "steps": 600,
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
                "encoder": {"name": "returns-context-v0", "params": {}},
                "agent": {
                    "name": "linear-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.03,
                    "value_lr": 0.05,
                    "gradient_clip": 5.0,
                    "params": {"clip_epsilon": 0.2, "update_epochs": 2},
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
            trading_config.write_text(json.dumps({**base_payload, "experiment_name": "trading_task"}), encoding="utf-8")
            regime_config.write_text(json.dumps({**base_payload, "experiment_name": "regime_task"}), encoding="utf-8")
            spec_path.write_text(
                json.dumps(
                    {
                        "matrix_selection_metric": "train.approx_kl",
                        "matrix_selection_mode": "min",
                        "tasks": [
                            {
                                "name": "trading_suite",
                                "config": "trading.json",
                                "task_weight": 1.0,
                                "report_metrics": ["train.approx_kl"],
                            },
                            {
                                "name": "regime_suite",
                                "config": "regime.json",
                                "task_weight": 1.0,
                                "report_metrics": ["train.approx_kl"],
                            },
                        ],
                        "methods": [
                            {"name": "method_alpha", "overrides": {"encoder.name": "returns-context-v0"}},
                            {"name": "method_beta", "overrides": {"encoder.name": "price-context-v0"}},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_study_spec(
                generated_spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
            ) -> tuple[StudyArtifacts, dict]:
                spec_payload = json.loads(Path(generated_spec_path).read_text(encoding="utf-8"))
                run_dir = Path(output_dir or tmp_path / "study_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "study_summary.json"
                leaderboard_csv_path = run_dir / "leaderboard.csv"
                leaderboard_md_path = run_dir / "leaderboard.md"
                summary_path.write_text("{}", encoding="utf-8")
                leaderboard_csv_path.write_text("", encoding="utf-8")
                leaderboard_md_path.write_text("", encoding="utf-8")

                task_base = Path(spec_payload["base_config"]).stem
                if "trading_suite" in task_base:
                    ordered = [("method_alpha", 1.0), ("method_beta", 0.5)]
                    report_values = {"method_alpha": 0.08, "method_beta": 0.02}
                else:
                    ordered = [("method_alpha", 1.0), ("method_beta", 0.5)]
                    report_values = {"method_alpha": 0.06, "method_beta": 0.01}

                summary = {
                    "mode": "train",
                    "selection_metric": "mean_terminal_score",
                    "selection_mode": "max",
                    "report_metrics": spec_payload.get("report_metrics", []),
                    "run_count": 2,
                    "top_experiment": ordered[0][0],
                    "top_feasible_experiment": ordered[0][0],
                    "feasible_run_count": 2,
                    "infeasible_run_count": 0,
                    "leaderboard": [
                        {
                            "rank": index,
                            "experiment": method_name,
                            "selection_value": value,
                            "constraint_feasible": True,
                            "report_train_approx_kl": report_values[method_name],
                            "summary_path": str(run_dir / method_name / "summary.json"),
                        }
                        for index, (method_name, value) in enumerate(ordered, start=1)
                    ],
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return (
                    StudyArtifacts(
                        root_dir=run_dir,
                        summary_path=summary_path,
                        leaderboard_csv_path=leaderboard_csv_path,
                        leaderboard_md_path=leaderboard_md_path,
                    ),
                    summary,
                )

            with patch("tsrl_lite.matrix.run_study_spec", side_effect=fake_run_study_spec):
                _, summary = run_benchmark_matrix_spec(spec_path, output_dir=matrix_dir)

            self.assertEqual(summary["matrix_selection_metric"], "train.approx_kl")
            self.assertEqual(summary["matrix_selection_mode"], "min")
            self.assertEqual(summary["top_method"], "method_beta")
            self.assertAlmostEqual(float(summary["leaderboard"][0]["matrix_selection_value"]), 0.015)
            self.assertTrue(summary["leaderboard"][0]["matrix_constraint_feasible"])

    def test_run_benchmark_matrix_spec_respects_matrix_metric_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            trading_config = tmp_path / "trading.json"
            regime_config = tmp_path / "regime.json"
            spec_path = tmp_path / "matrix_spec.json"
            matrix_dir = tmp_path / "matrix_run"

            base_payload = {
                "experiment_name": "base_task",
                "seed": 7,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.75,
                    "val_ratio": 0.1,
                    "synthetic": {
                        "steps": 600,
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
                "encoder": {"name": "returns-context-v0", "params": {}},
                "agent": {
                    "name": "linear-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.03,
                    "value_lr": 0.05,
                    "gradient_clip": 5.0,
                    "params": {"clip_epsilon": 0.2, "update_epochs": 2},
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
            trading_config.write_text(json.dumps({**base_payload, "experiment_name": "trading_task"}), encoding="utf-8")
            regime_config.write_text(json.dumps({**base_payload, "experiment_name": "regime_task"}), encoding="utf-8")
            spec_path.write_text(
                json.dumps(
                    {
                        "matrix_selection_metric": "train.approx_kl",
                        "matrix_selection_mode": "min",
                        "matrix_metric_constraints": {
                            "train.approx_kl": {"max": 0.03}
                        },
                        "tasks": [
                            {
                                "name": "trading_suite",
                                "config": "trading.json",
                                "task_weight": 1.0,
                                "report_metrics": ["train.approx_kl"],
                            },
                            {
                                "name": "regime_suite",
                                "config": "regime.json",
                                "task_weight": 1.0,
                                "report_metrics": ["train.approx_kl"],
                            },
                        ],
                        "methods": [
                            {"name": "method_alpha", "overrides": {"encoder.name": "returns-context-v0"}},
                            {"name": "method_beta", "overrides": {"encoder.name": "price-context-v0"}},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_study_spec(
                generated_spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
            ) -> tuple[StudyArtifacts, dict]:
                spec_payload = json.loads(Path(generated_spec_path).read_text(encoding="utf-8"))
                run_dir = Path(output_dir or tmp_path / "study_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "study_summary.json"
                leaderboard_csv_path = run_dir / "leaderboard.csv"
                leaderboard_md_path = run_dir / "leaderboard.md"
                summary_path.write_text("{}", encoding="utf-8")
                leaderboard_csv_path.write_text("", encoding="utf-8")
                leaderboard_md_path.write_text("", encoding="utf-8")

                task_base = Path(spec_payload["base_config"]).stem
                if "trading_suite" in task_base:
                    report_values = {"method_alpha": 0.02, "method_beta": 0.01}
                else:
                    report_values = {"method_alpha": 0.02, "method_beta": 0.08}

                summary = {
                    "mode": "train",
                    "selection_metric": "mean_terminal_score",
                    "selection_mode": "max",
                    "report_metrics": spec_payload.get("report_metrics", []),
                    "run_count": 2,
                    "top_experiment": "method_alpha",
                    "top_feasible_experiment": "method_alpha",
                    "feasible_run_count": 2,
                    "infeasible_run_count": 0,
                    "leaderboard": [
                        {
                            "rank": index,
                            "experiment": method_name,
                            "selection_value": 1.0 if method_name == "method_alpha" else 0.5,
                            "constraint_feasible": True,
                            "report_train_approx_kl": report_values[method_name],
                            "summary_path": str(run_dir / method_name / "summary.json"),
                        }
                        for index, method_name in enumerate(("method_alpha", "method_beta"), start=1)
                    ],
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return (
                    StudyArtifacts(
                        root_dir=run_dir,
                        summary_path=summary_path,
                        leaderboard_csv_path=leaderboard_csv_path,
                        leaderboard_md_path=leaderboard_md_path,
                    ),
                    summary,
                )

            with patch("tsrl_lite.matrix.run_study_spec", side_effect=fake_run_study_spec):
                _, summary = run_benchmark_matrix_spec(spec_path, output_dir=matrix_dir)

            self.assertEqual(summary["top_method"], "method_alpha")
            self.assertTrue(summary["leaderboard"][0]["matrix_constraint_feasible"])
            self.assertFalse(summary["leaderboard"][1]["matrix_constraint_feasible"])
            self.assertEqual(summary["leaderboard"][1]["matrix_constraint_violation_count"], 1)

    def test_run_benchmark_matrix_spec_skips_filtered_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "task.json"
            spec_path = tmp_path / "matrix_spec.json"
            matrix_dir = tmp_path / "matrix_run"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "task_base",
                        "seed": 3,
                        "data": {
                            "source": "synthetic",
                            "window_size": 16,
                            "train_ratio": 0.75,
                            "val_ratio": 0.1,
                            "synthetic": {
                                "steps": 500,
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
                        "encoder": {"name": "returns-context-v0", "params": {}},
                        "agent": {
                            "name": "linear-ppo",
                            "gamma": 0.99,
                            "gae_lambda": 0.95,
                            "policy_lr": 0.03,
                            "value_lr": 0.05,
                            "gradient_clip": 5.0,
                            "params": {"clip_epsilon": 0.2, "update_epochs": 2},
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
                        "tasks": [
                            {"name": "kept_task", "config": "task.json"},
                            {"name": "skipped_task", "config": "task.json"},
                        ],
                        "methods": [{"name": "method_alpha", "overrides": {"encoder.name": "returns-context-v0"}}],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run_study_spec(
                generated_spec_path: str | Path,
                output_dir: str | Path | None = None,
                include_tags: list[str] | None = None,
                exclude_tags: list[str] | None = None,
            ) -> tuple[StudyArtifacts, dict]:
                task_name = Path(generated_spec_path).stem
                if "skipped_task" in task_name:
                    raise ValueError("study spec did not produce any experiments after tag filtering")
                run_dir = Path(output_dir or tmp_path / "study_run")
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "study_summary.json"
                leaderboard_csv_path = run_dir / "leaderboard.csv"
                leaderboard_md_path = run_dir / "leaderboard.md"
                summary = {
                    "mode": "train",
                    "selection_metric": "mean_terminal_score",
                    "selection_mode": "max",
                    "run_count": 1,
                    "top_experiment": "method_alpha",
                    "top_feasible_experiment": "method_alpha",
                    "feasible_run_count": 1,
                    "infeasible_run_count": 0,
                    "leaderboard": [
                        {
                            "rank": 1,
                            "experiment": "method_alpha",
                            "selection_value": 1.0,
                            "constraint_feasible": True,
                            "summary_path": str(run_dir / "method_alpha" / "summary.json"),
                        }
                    ],
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                leaderboard_csv_path.write_text("", encoding="utf-8")
                leaderboard_md_path.write_text("", encoding="utf-8")
                return (
                    StudyArtifacts(
                        root_dir=run_dir,
                        summary_path=summary_path,
                        leaderboard_csv_path=leaderboard_csv_path,
                        leaderboard_md_path=leaderboard_md_path,
                    ),
                    summary,
                )

            with patch("tsrl_lite.matrix.run_study_spec", side_effect=fake_run_study_spec):
                artifacts, summary = run_benchmark_matrix_spec(spec_path, output_dir=matrix_dir)

            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(summary["task_count"], 1)
            self.assertEqual(summary["skipped_task_count"], 1)
            self.assertEqual(summary["top_method"], "method_alpha")
            self.assertEqual(summary["skipped_tasks"][0]["reason"], "filtered_out")


if __name__ == "__main__":
    unittest.main()
