from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.benchmark import run_walk_forward_benchmark


class WalkForwardBenchmarkTest(unittest.TestCase):
    def test_run_walk_forward_benchmark_aggregates_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "walk_forward_config.json"
            output_dir = tmp_path / "walk_forward_run"
            config = {
                "experiment_name": "walk_forward_smoke",
                "seed": 5,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.7,
                    "val_ratio": 0.1,
                    "synthetic": {
                        "steps": 1400,
                        "drift": 0.001,
                        "volatility": 0.006,
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
                    "checkpoint_dir": str(output_dir),
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            artifacts, summary = run_walk_forward_benchmark(
                config_path=config_path,
                n_folds=2,
                train_ratio_start=0.45,
                output_dir=output_dir,
            )

            self.assertTrue(artifacts.summary_path.exists())
            self.assertEqual(summary["walk_forward_folds"], 2)
            self.assertEqual(len(summary["runs"]), 2)
            self.assertIn("mean_reward", summary["aggregate_evaluation"])
            self.assertIn("mean_reward", summary["aggregate_best_validation"])
            self.assertIn("aggregate_train_update", summary)
            self.assertIn("aggregate_train_update_tail", summary)
            self.assertIn("value_loss", summary["aggregate_train_update"])
            self.assertIn("summary_path", summary["runs"][0])
            self.assertIn("best_checkpoint_path", summary["runs"][0])
            self.assertIn("train_update_metrics", summary["runs"][0])
            self.assertTrue((output_dir / "fold_0" / "summary.json").exists())
            self.assertTrue((output_dir / "fold_1" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
