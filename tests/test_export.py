from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.export import export_rollouts
from tsrl_lite.trainer import train_experiment


class ExportRolloutsTest(unittest.TestCase):
    def test_export_rollouts_writes_npz_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "export_config.json"
            output_dir = tmp_path / "run"
            rollout_path = tmp_path / "rollouts.npz"
            config = {
                "experiment_name": "export_smoke",
                "seed": 3,
                "data": {
                    "source": "synthetic",
                    "window_size": 16,
                    "train_ratio": 0.75,
                    "synthetic": {
                        "steps": 700,
                        "drift": 0.001,
                        "volatility": 0.007,
                        "seasonality": 0.002
                    }
                },
                "env": {
                    "name": "trading-v0",
                    "reward_scale": 1.0,
                    "episode_horizon": 64,
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
                        "update_epochs": 2
                    }
                },
                "trainer": {
                    "episodes": 3,
                    "eval_episodes": 1,
                    "log_interval": 3,
                    "checkpoint_dir": str(output_dir)
                }
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            artifacts, _ = train_experiment(config_path=config_path, output_dir=output_dir)
            export_artifacts, metadata = export_rollouts(
                config_path=config_path,
                checkpoint_path=artifacts.checkpoint_path,
                output_path=rollout_path,
                split="eval",
                episodes=1,
                greedy=True,
            )

            self.assertTrue(export_artifacts.rollout_path.exists())
            self.assertTrue(export_artifacts.metadata_path.exists())
            payload = np.load(export_artifacts.rollout_path)
            self.assertGreater(payload["actions"].shape[0], 0)
            self.assertEqual(int(metadata["transitions"]), int(payload["actions"].shape[0]))
            self.assertIn("info_equity", payload.files)
            self.assertIn("info_turnover", payload.files)
            self.assertIn("equity", metadata["info_keys"])
            self.assertIn("turnover", metadata["info_keys"])


if __name__ == "__main__":
    unittest.main()
