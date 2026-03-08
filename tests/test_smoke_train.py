from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.trainer import train_experiment


class SmokeTrainTest(unittest.TestCase):
    def test_train_experiment_produces_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "smoke_config.json"
            output_dir = tmp_path / "run"
            config = {
                "experiment_name": "smoke",
                "seed": 3,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.75,
                    "synthetic": {
                        "steps": 800,
                        "drift": 0.001,
                        "volatility": 0.007,
                        "seasonality": 0.002
                    }
                },
                "env": {
                    "name": "trading-v0",
                    "reward_scale": 1.0,
                    "episode_horizon": 96,
                    "random_reset": True,
                    "params": {
                        "positions": [-1.0, 0.0, 1.0],
                        "trading_cost": 0.0005
                    }
                },
                "encoder": {
                    "name": "returns-context-v0",
                    "params": {}
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
                        "update_epochs": 3
                    }
                },
                "trainer": {
                    "episodes": 6,
                    "eval_episodes": 1,
                    "log_interval": 3,
                    "checkpoint_dir": str(output_dir)
                }
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            artifacts, summary = train_experiment(config_path=config_path, output_dir=output_dir)

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertTrue(artifacts.best_checkpoint_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.history_path.exists())
            self.assertTrue(artifacts.history_csv_path.exists())
            self.assertIn("evaluation", summary)
            self.assertIn("best_validation", summary)
            self.assertIn("evaluation_last", summary)
            self.assertIn("evaluation_best", summary)
            self.assertIn("train_update_metrics", summary)
            self.assertTrue(summary["evaluation"]["mean_length"] > 0)
            self.assertEqual(summary["encoder"], "returns-context-v0")
            self.assertIn("mean_max_drawdown", summary["evaluation"])
            self.assertIn("mean_sharpe_ratio", summary["evaluation"])
            self.assertIn("mean_terminal_return", summary["evaluation"])
            self.assertIn("approx_kl", summary["train_update_metrics"]["overall"])
            self.assertIn("clip_fraction", summary["train_update_metrics"]["overall"])
            self.assertGreater(summary["train_update_metrics"]["overall"]["value_loss"]["mean"], 0.0)

    def test_train_regime_experiment_produces_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "regime_config.json"
            output_dir = tmp_path / "regime_run"
            config = {
                "experiment_name": "regime_smoke",
                "seed": 13,
                "data": {
                    "source": "synthetic",
                    "window_size": 20,
                    "train_ratio": 0.8,
                    "synthetic": {
                        "steps": 900,
                        "drift": 0.0012,
                        "volatility": 0.006,
                        "seasonality": 0.002
                    }
                },
                "env": {
                    "name": "regime-v0",
                    "reward_scale": 1.0,
                    "episode_horizon": 72,
                    "random_reset": True,
                    "params": {
                        "forecast_horizon": 6,
                        "regime_threshold": 0.002
                    }
                },
                "encoder": {
                    "name": "price-context-v0",
                    "params": {}
                },
                "agent": {
                    "name": "linear-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.03,
                    "value_lr": 0.05,
                    "gradient_clip": 5.0,
                    "params": {
                        "clip_epsilon": 0.15,
                        "update_epochs": 3
                    }
                },
                "trainer": {
                    "episodes": 4,
                    "eval_episodes": 1,
                    "log_interval": 2,
                    "checkpoint_dir": str(output_dir)
                }
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            artifacts, summary = train_experiment(config_path=config_path, output_dir=output_dir)

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertTrue(artifacts.best_checkpoint_path.exists())
            self.assertEqual(summary["env"], "regime-v0")
            self.assertEqual(summary["encoder"], "price-context-v0")
            self.assertIn("train_update_metrics", summary)
            self.assertNotIn("mean_max_drawdown", summary["evaluation"])

    def test_train_csv_time_split_experiment_produces_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "prices.csv"
            config_path = tmp_path / "csv_time_config.json"
            output_dir = tmp_path / "csv_time_run"
            start_date = datetime(2024, 1, 1)
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "close"])
                writer.writeheader()
                for day in range(220):
                    current = start_date + timedelta(days=day)
                    writer.writerow(
                        {
                            "timestamp": current.isoformat(),
                            "close": f"{100.0 + (0.35 * day):.6f}",
                        }
                    )

            config = {
                "experiment_name": "csv_time_smoke",
                "seed": 23,
                "data": {
                    "source": "csv",
                    "csv_path": str(csv_path),
                    "price_column": "close",
                    "timestamp_column": "timestamp",
                    "window_size": 16,
                    "val_ratio": 0.1,
                    "train_end_time": "2024-05-15T00:00:00",
                    "val_end_time": "2024-06-20T00:00:00",
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
                    "episodes": 4,
                    "eval_episodes": 1,
                    "log_interval": 2,
                    "checkpoint_dir": str(output_dir),
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            artifacts, summary = train_experiment(config_path=config_path, output_dir=output_dir)

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertTrue(artifacts.best_checkpoint_path.exists())
            self.assertEqual(summary["data_source"], "csv")
            self.assertEqual(summary["data_split_mode"], "time")
            self.assertEqual(summary["train_end_time"], "2024-05-15T00:00:00")
            self.assertEqual(summary["val_end_time"], "2024-06-20T00:00:00")
            self.assertIn("mean_max_drawdown", summary["evaluation"])

    def test_train_portfolio_experiment_produces_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "portfolio_config.json"
            output_dir = tmp_path / "portfolio_run"
            config = {
                "experiment_name": "portfolio_smoke",
                "seed": 19,
                "data": {
                    "source": "synthetic",
                    "window_size": 20,
                    "train_ratio": 0.8,
                    "synthetic": {
                        "steps": 900,
                        "assets": 3,
                        "drift": 0.0012,
                        "volatility": 0.008,
                        "seasonality": 0.002,
                        "correlation": 0.4,
                    },
                },
                "env": {
                    "name": "portfolio-v0",
                    "reward_scale": 1.0,
                    "episode_horizon": 72,
                    "random_reset": True,
                    "params": {
                        "trading_cost": 0.0005,
                        "allocation_candidates": [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.3333333333, 0.3333333333, 0.3333333333],
                        ],
                    },
                },
                "encoder": {
                    "name": "multi-asset-context-v0",
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
                        "update_epochs": 3,
                    },
                },
                "trainer": {
                    "episodes": 4,
                    "eval_episodes": 1,
                    "log_interval": 2,
                    "checkpoint_dir": str(output_dir),
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            artifacts, summary = train_experiment(config_path=config_path, output_dir=output_dir)

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertTrue(artifacts.best_checkpoint_path.exists())
            self.assertEqual(summary["env"], "portfolio-v0")
            self.assertEqual(summary["encoder"], "multi-asset-context-v0")
            self.assertIn("mean_max_drawdown", summary["evaluation"])
            self.assertIn("mean_sharpe_ratio", summary["evaluation"])


if __name__ == "__main__":
    unittest.main()
