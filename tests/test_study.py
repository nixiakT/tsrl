from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.study import apply_config_overrides, run_study, run_study_spec


class StudyTest(unittest.TestCase):
    def test_apply_config_overrides_supports_nested_and_dotted_keys(self) -> None:
        payload = {
            "experiment_name": "base",
            "encoder": {"name": "returns-context-v0", "params": {"short_window": 4}},
            "agent": {"name": "linear-ppo", "params": {"clip_epsilon": 0.2}},
        }

        updated = apply_config_overrides(
            payload,
            {
                "experiment_name": "override",
                "encoder.name": "multi-scale-context-v0",
                "encoder.params": {"long_window": 12},
                "agent.params.clip_epsilon": 0.1,
            },
        )

        self.assertEqual(updated["experiment_name"], "override")
        self.assertEqual(updated["encoder"]["name"], "multi-scale-context-v0")
        self.assertEqual(updated["encoder"]["params"]["short_window"], 4)
        self.assertEqual(updated["encoder"]["params"]["long_window"], 12)
        self.assertEqual(updated["agent"]["params"]["clip_epsilon"], 0.1)

    def test_run_study_ranks_multiple_configs_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            study_dir = tmp_path / "study_run"
            config_a = tmp_path / "config_a.json"
            config_b = tmp_path / "config_b.json"

            base_config = {
                "seed": 7,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.75,
                    "val_ratio": 0.1,
                    "synthetic": {
                        "steps": 800,
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
                    "episodes": 3,
                    "val_episodes": 1,
                    "eval_episodes": 1,
                    "log_interval": 3,
                    "eval_interval": 1,
                },
            }

            config_a.write_text(
                json.dumps(
                    {
                        **base_config,
                        "experiment_name": "study_alpha",
                        "trainer": {**base_config["trainer"], "checkpoint_dir": str(tmp_path / "alpha_run")},
                    }
                ),
                encoding="utf-8",
            )
            config_b.write_text(
                json.dumps(
                    {
                        **base_config,
                        "experiment_name": "study_beta",
                        "seed": 11,
                        "data": {
                            **base_config["data"],
                            "synthetic": {
                                "steps": 800,
                                "drift": 0.0005,
                                "volatility": 0.009,
                                "seasonality": 0.001,
                            },
                        },
                        "trainer": {**base_config["trainer"], "checkpoint_dir": str(tmp_path / "beta_run")},
                    }
                ),
                encoding="utf-8",
            )

            artifacts, summary = run_study(
                config_paths=[config_a, config_b],
                mode="train",
                output_dir=study_dir,
                selection_metric="mean_terminal_score",
                selection_mode="max",
                report_metrics=["train.approx_kl", "train.value_loss", "train_tail.clip_fraction"],
            )

            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.leaderboard_csv_path.exists())
            self.assertTrue(artifacts.leaderboard_md_path.exists())
            self.assertIsNotNone(artifacts.metrics_csv_path)
            self.assertIsNotNone(artifacts.metrics_md_path)
            self.assertTrue(artifacts.metrics_csv_path.exists())
            self.assertTrue(artifacts.metrics_md_path.exists())
            self.assertEqual(summary["run_count"], 2)
            self.assertEqual(summary["leaderboard"][0]["rank"], 1)
            self.assertGreaterEqual(
                float(summary["leaderboard"][0]["selection_value"]),
                float(summary["leaderboard"][1]["selection_value"]),
            )
            self.assertIn("report_train_approx_kl", summary["leaderboard"][0])
            self.assertIn("train_approx_kl_mean", summary["leaderboard"][0])
            self.assertIn("train_value_loss_mean", summary["leaderboard"][0])
            self.assertIn("train_tail_clip_fraction_mean", summary["leaderboard"][0])
            leaderboard_md = artifacts.leaderboard_md_path.read_text(encoding="utf-8")
            self.assertIn("study_alpha", leaderboard_md)
            self.assertIn("study_beta", leaderboard_md)

    def test_run_study_supports_benchmark_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            study_dir = tmp_path / "study_benchmark_run"
            config_path = tmp_path / "benchmark_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "study_benchmark",
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
                            "episodes": 3,
                            "val_episodes": 1,
                            "eval_episodes": 1,
                            "log_interval": 3,
                            "eval_interval": 1,
                            "checkpoint_dir": str(tmp_path / "benchmark_run"),
                        },
                    }
                ),
                encoding="utf-8",
            )

            artifacts, summary = run_study(
                config_paths=[config_path],
                mode="benchmark",
                output_dir=study_dir,
                selection_metric="mean_reward",
                selection_mode="max",
                report_metrics=["train.value_loss", "train_tail.approx_kl"],
                seeds=[3, 5],
            )

            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(summary["mode"], "benchmark")
            self.assertEqual(summary["benchmark_seeds"], [3, 5])
            self.assertEqual(summary["run_count"], 1)
            self.assertIn("evaluation_mean_reward_mean", summary["leaderboard"][0])
            self.assertIn("validation_mean_reward_mean", summary["leaderboard"][0])
            self.assertIn("train_value_loss_mean", summary["leaderboard"][0])
            self.assertIn("train_tail_approx_kl_mean", summary["leaderboard"][0])
            self.assertIn("report_train_value_loss", summary["leaderboard"][0])

    def test_run_study_prioritizes_feasible_rows_under_metric_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            study_dir = tmp_path / "study_constraints_run"
            config_a = tmp_path / "config_a.json"
            config_b = tmp_path / "config_b.json"
            base_config = {
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
                },
            }
            config_a.write_text(
                json.dumps(
                    {
                        **base_config,
                        "experiment_name": "constraint_alpha",
                        "trainer": {**base_config["trainer"], "checkpoint_dir": str(tmp_path / "alpha_run")},
                    }
                ),
                encoding="utf-8",
            )
            config_b.write_text(
                json.dumps(
                    {
                        **base_config,
                        "experiment_name": "constraint_beta",
                        "trainer": {**base_config["trainer"], "checkpoint_dir": str(tmp_path / "beta_run")},
                    }
                ),
                encoding="utf-8",
            )

            def fake_train_experiment(
                config_path: str | Path,
                output_dir: str | Path | None = None,
            ) -> tuple[object, dict]:
                path = Path(config_path)
                run_dir = Path(output_dir or tmp_path / path.stem)
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "summary.json"
                if path.stem == "config_a":
                    summary = {
                        "evaluation": {
                            "mean_terminal_score": 2.0,
                            "mean_max_drawdown": 0.6,
                        },
                        "best_validation": {
                            "mean_terminal_score": 2.0,
                        },
                    }
                else:
                    summary = {
                        "evaluation": {
                            "mean_terminal_score": 1.0,
                            "mean_max_drawdown": 0.2,
                        },
                        "best_validation": {
                            "mean_terminal_score": 1.0,
                        },
                    }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return SimpleNamespace(summary_path=summary_path), summary

            with patch("tsrl_lite.study.train_experiment", side_effect=fake_train_experiment):
                artifacts, summary = run_study(
                    config_paths=[config_a, config_b],
                    mode="train",
                    output_dir=study_dir,
                    selection_metric="mean_terminal_score",
                    selection_mode="max",
                    metric_constraints={"mean_max_drawdown": {"max": 0.3}},
                )

            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(summary["feasible_run_count"], 1)
            self.assertEqual(summary["infeasible_run_count"], 1)
            self.assertEqual(summary["top_feasible_experiment"], "constraint_beta")
            self.assertEqual(summary["top_experiment"], "constraint_beta")
            self.assertTrue(summary["leaderboard"][0]["constraint_feasible"])
            self.assertFalse(summary["leaderboard"][1]["constraint_feasible"])
            self.assertGreater(float(summary["leaderboard"][1]["selection_value"]), 1.0)

    def test_run_study_supports_train_update_selection_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            study_dir = tmp_path / "study_train_metric_run"
            config_a = tmp_path / "config_a.json"
            config_b = tmp_path / "config_b.json"

            base_config = {
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
                },
            }
            config_a.write_text(
                json.dumps(
                    {
                        **base_config,
                        "experiment_name": "train_metric_alpha",
                        "trainer": {**base_config["trainer"], "checkpoint_dir": str(tmp_path / "alpha_run")},
                    }
                ),
                encoding="utf-8",
            )
            config_b.write_text(
                json.dumps(
                    {
                        **base_config,
                        "experiment_name": "train_metric_beta",
                        "trainer": {**base_config["trainer"], "checkpoint_dir": str(tmp_path / "beta_run")},
                    }
                ),
                encoding="utf-8",
            )

            def fake_train_experiment(
                config_path: str | Path,
                output_dir: str | Path | None = None,
            ) -> tuple[object, dict]:
                path = Path(config_path)
                run_dir = Path(output_dir or tmp_path / path.stem)
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "summary.json"
                if path.stem == "config_a":
                    train_update_metrics = {
                        "overall": {"approx_kl": {"mean": 0.08, "min": 0.05, "max": 0.12, "last": 0.10}},
                        "tail": {"update_steps": {"mean": 8.0, "min": 8.0, "max": 8.0, "last": 8.0}},
                    }
                else:
                    train_update_metrics = {
                        "overall": {"approx_kl": {"mean": 0.02, "min": 0.01, "max": 0.03, "last": 0.02}},
                        "tail": {"update_steps": {"mean": 10.0, "min": 10.0, "max": 10.0, "last": 10.0}},
                    }
                summary = {
                    "evaluation": {"mean_terminal_score": 1.0},
                    "best_validation": {"mean_terminal_score": 1.0},
                    "train_update_metrics": train_update_metrics,
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return SimpleNamespace(summary_path=summary_path), summary

            with patch("tsrl_lite.study.train_experiment", side_effect=fake_train_experiment):
                artifacts, summary = run_study(
                    config_paths=[config_a, config_b],
                    mode="train",
                    output_dir=study_dir,
                    selection_metric="train.approx_kl",
                    selection_mode="min",
                    report_metrics=["train_tail.update_steps"],
                )

            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(summary["top_experiment"], "train_metric_beta")
            self.assertAlmostEqual(float(summary["leaderboard"][0]["selection_value"]), 0.02)
            self.assertIn("report_train_tail_update_steps", summary["leaderboard"][0])

    def test_run_study_writes_metric_report_and_pareto_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            study_dir = tmp_path / "study_report_run"
            config_a = tmp_path / "config_a.json"
            config_b = tmp_path / "config_b.json"
            config_c = tmp_path / "config_c.json"
            base_config = {
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
                },
            }
            for index, path in enumerate((config_a, config_b, config_c), start=1):
                path.write_text(
                    json.dumps(
                        {
                            **base_config,
                            "experiment_name": f"pareto_{index}",
                            "trainer": {
                                **base_config["trainer"],
                                "checkpoint_dir": str(tmp_path / f"pareto_run_{index}"),
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            metric_map = {
                "config_a": {
                    "mean_sharpe_ratio": 1.0,
                    "mean_max_drawdown": 0.10,
                    "mean_terminal_return": 0.12,
                },
                "config_b": {
                    "mean_sharpe_ratio": 1.2,
                    "mean_max_drawdown": 0.15,
                    "mean_terminal_return": 0.15,
                },
                "config_c": {
                    "mean_sharpe_ratio": 0.8,
                    "mean_max_drawdown": 0.25,
                    "mean_terminal_return": 0.10,
                },
            }

            def fake_train_experiment(
                config_path: str | Path,
                output_dir: str | Path | None = None,
            ) -> tuple[object, dict]:
                path = Path(config_path)
                run_dir = Path(output_dir or tmp_path / path.stem)
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_path = run_dir / "summary.json"
                evaluation = metric_map[path.stem]
                summary = {
                    "evaluation": evaluation,
                    "best_validation": evaluation,
                }
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                return SimpleNamespace(summary_path=summary_path), summary

            with patch("tsrl_lite.study.train_experiment", side_effect=fake_train_experiment):
                artifacts, summary = run_study(
                    config_paths=[config_a, config_b, config_c],
                    mode="train",
                    output_dir=study_dir,
                    selection_metric="mean_sharpe_ratio",
                    selection_mode="max",
                    report_metrics=["mean_terminal_return"],
                    pareto_metrics={
                        "mean_sharpe_ratio": "max",
                        "mean_max_drawdown": "min",
                    },
                )

            self.assertIsNotNone(artifacts.metrics_csv_path)
            self.assertIsNotNone(artifacts.metrics_md_path)
            self.assertIsNotNone(artifacts.pareto_frontier_path)
            self.assertIsNotNone(artifacts.pareto_frontier_md_path)
            self.assertTrue(artifacts.metrics_csv_path.exists())
            self.assertTrue(artifacts.metrics_md_path.exists())
            self.assertTrue(artifacts.pareto_frontier_path.exists())
            self.assertTrue(artifacts.pareto_frontier_md_path.exists())
            self.assertEqual(summary["pareto_pool"], "feasible")
            self.assertIn("mean_terminal_return", summary["report_metrics"])
            frontier_names = {row["experiment"] for row in summary["pareto_frontier"]}
            self.assertEqual(frontier_names, {"pareto_1", "pareto_2"})

    def test_run_study_spec_generates_resolved_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "study_spec.json"
            study_dir = tmp_path / "study_spec_run"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "spec_base",
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
                            "episodes": 3,
                            "val_episodes": 1,
                            "eval_episodes": 1,
                            "log_interval": 3,
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
                        "base_config": "base_config.json",
                        "mode": "train",
                        "selection_metric": "mean_terminal_score",
                        "experiments": [
                            {"name": "spec_alpha"},
                            {
                                "name": "spec_beta",
                                "overrides": {
                                    "encoder.name": "price-context-v0",
                                    "encoder.params": {},
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            artifacts, summary = run_study_spec(spec_path=spec_path, output_dir=study_dir)

            self.assertTrue(artifacts.summary_path.exists())
            self.assertIsNotNone(artifacts.resolved_config_dir)
            self.assertTrue(artifacts.resolved_config_dir.exists())
            self.assertEqual(summary["spec_path"], str(spec_path))
            self.assertEqual(len(summary["config_paths"]), 2)
            resolved_files = sorted(artifacts.resolved_config_dir.glob("*.json"))
            self.assertEqual(len(resolved_files), 2)
            resolved_payload = json.loads(resolved_files[1].read_text(encoding="utf-8"))
            self.assertEqual(resolved_payload["experiment_name"], "spec_beta")
            self.assertEqual(resolved_payload["encoder"]["name"], "price-context-v0")

    def test_run_study_spec_supports_grid_and_tag_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "base_config.json"
            spec_path = tmp_path / "grid_spec.json"
            study_dir = tmp_path / "grid_spec_run"
            config_path.write_text(
                json.dumps(
                    {
                        "experiment_name": "grid_base",
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
                            "name": "multi-scale-context-v0",
                            "params": {
                                "short_window": 4,
                                "long_window": 12,
                            },
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
                            "checkpoint_dir": str(tmp_path / "grid_run"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "base_config": "base_config.json",
                        "mode": "train",
                        "selection_metric": "mean_terminal_score",
                        "tags": ["trading"],
                        "experiments": [
                            {
                                "name": "encoder_grid",
                                "tags": ["fast", "encoder"],
                                "grid": {
                                    "encoder.name": [
                                        "returns-context-v0",
                                        "price-context-v0",
                                    ]
                                },
                            },
                            {
                                "name": "slow_baseline",
                                "tags": ["slow"],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            artifacts, summary = run_study_spec(
                spec_path=spec_path,
                output_dir=study_dir,
                include_tags=["trading", "fast"],
                exclude_tags=["slow"],
            )

            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(summary["generated_variants"], 3)
            self.assertEqual(summary["include_tags"], ["fast", "trading"])
            self.assertEqual(summary["exclude_tags"], ["slow"])
            self.assertEqual(summary["run_count"], 2)
            self.assertEqual(len(summary["config_paths"]), 2)
            self.assertIn("study_tags", summary["leaderboard"][0])
            self.assertIn("sweep_encoder_name", summary["leaderboard"][0])
            resolved_files = sorted(artifacts.resolved_config_dir.glob("*.json"))
            self.assertEqual(len(resolved_files), 2)
            self.assertTrue(all("encoder_grid" in path.stem for path in resolved_files))


if __name__ == "__main__":
    unittest.main()
