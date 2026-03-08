from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

from tsrl_lite.trainer import train_experiment


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TorchTransformerSmokeTest(unittest.TestCase):
    def test_torch_transformer_training_produces_pt_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "torch_transformer_config.json"
            output_dir = tmp_path / "run"
            config = {
                "experiment_name": "torch_transformer_smoke",
                "seed": 29,
                "data": {
                    "source": "synthetic",
                    "window_size": 20,
                    "train_ratio": 0.8,
                    "synthetic": {
                        "steps": 800,
                        "drift": 0.0012,
                        "volatility": 0.006,
                        "seasonality": 0.002
                    }
                },
                "env": {
                    "name": "regime-v0",
                    "reward_scale": 1.0,
                    "episode_horizon": 64,
                    "random_reset": True,
                    "params": {
                        "forecast_horizon": 6,
                        "regime_threshold": 0.002
                    }
                },
                "encoder": {
                    "name": "sequence-window-v0",
                    "params": {}
                },
                "agent": {
                    "name": "torch-transformer-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.001,
                    "value_lr": 0.001,
                    "gradient_clip": 1.0,
                    "params": {
                        "hidden_size": 32,
                        "num_layers": 1,
                        "num_heads": 4,
                        "dropout": 0.0,
                        "clip_epsilon": 0.2,
                        "update_epochs": 2,
                        "entropy_coef": 0.01,
                        "value_coef": 0.5,
                        "mini_batch_size": 8,
                        "target_kl": 0.05,
                        "value_clip_epsilon": 0.2,
                        "device": "cpu"
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

            artifacts, summary = train_experiment(config_path=config_path, output_dir=output_dir)

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertEqual(artifacts.checkpoint_path.suffix, ".pt")
            self.assertEqual(summary["agent"], "torch-transformer-ppo")
            self.assertIn("train_update_metrics", summary)
            self.assertIn("approx_kl", summary["train_update_metrics"]["overall"])
            self.assertIn("explained_variance", summary["train_update_metrics"]["overall"])


if __name__ == "__main__":
    unittest.main()
