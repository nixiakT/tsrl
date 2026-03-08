from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

from tsrl_lite.algorithms.common import EpisodeBatch
from tsrl_lite.encoders.sequence import SequenceWindowEncoder
from tsrl_lite.trainer import train_experiment

if TORCH_AVAILABLE:
    import torch

    from tsrl_lite.algorithms.torch_dlinear_ppo import TorchDLinearPPOAgent
    from tsrl_lite.networks.torch_dlinear import TorchDLinearActorCriticNetwork


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TorchDLinearSmokeTest(unittest.TestCase):
    def test_torch_dlinear_network_supports_multifeature_sequences(self) -> None:
        network = TorchDLinearActorCriticNetwork(
            input_dim=3,
            sequence_length=12,
            hidden_size=8,
            action_dim=4,
            moving_avg=5,
            individual=True,
            dropout=0.0,
        )
        sequence_batch = torch.randn(2, 12, 3)
        logits, value = network(sequence_batch)

        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(value.shape), (2,))

    def test_torch_dlinear_update_reports_stability_metrics(self) -> None:
        encoder = SequenceWindowEncoder(window_size=8, agent_feature_dim=1)
        agent = TorchDLinearPPOAgent(
            encoder=encoder,
            action_dim=3,
            gamma=0.99,
            gae_lambda=0.95,
            policy_lr=0.001,
            value_lr=0.001,
            gradient_clip=1.0,
            hidden_size=8,
            moving_avg=3,
            individual=False,
            dropout=0.0,
            clip_epsilon=0.2,
            update_epochs=3,
            entropy_coef=0.01,
            value_coef=0.5,
            mini_batch_size=2,
            normalize_advantages=True,
            shuffle_minibatches=False,
            target_kl=0.1,
            value_clip_epsilon=0.2,
            device="cpu",
            seed=31,
        )

        observations = np.random.default_rng(31).normal(
            size=(6, *encoder.observation_shape)
        ).astype(np.float32)
        batch = EpisodeBatch(
            observations=observations,
            actions=np.asarray([0, 0, 1, 1, 2, 2], dtype=int),
            rewards=np.asarray([0.8, 0.4, -0.2, 0.1, 0.3, -0.1], dtype=float),
            dones=np.asarray([False, False, False, False, False, True], dtype=bool),
            values=np.asarray([0.2, 0.3, 0.1, 0.0, -0.1, 0.2], dtype=float),
            action_probs=np.asarray(
                [
                    [0.99, 0.005, 0.005],
                    [0.99, 0.005, 0.005],
                    [0.005, 0.99, 0.005],
                    [0.005, 0.99, 0.005],
                    [0.005, 0.005, 0.99],
                    [0.005, 0.005, 0.99],
                ],
                dtype=float,
            ),
        )

        metrics = agent.update(batch)

        self.assertIn("approx_kl", metrics)
        self.assertIn("explained_variance", metrics)
        self.assertIn("early_stop_triggered", metrics)
        self.assertGreater(metrics["update_steps"], 0.0)
        self.assertEqual(metrics["early_stop_triggered"], 1.0)

    def test_torch_dlinear_training_produces_pt_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "torch_dlinear_config.json"
            output_dir = tmp_path / "run"
            config = {
                "experiment_name": "torch_dlinear_smoke",
                "seed": 41,
                "data": {
                    "source": "synthetic",
                    "window_size": 24,
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
                    "name": "torch-dlinear-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.001,
                    "value_lr": 0.001,
                    "gradient_clip": 1.0,
                    "params": {
                        "hidden_size": 16,
                        "moving_avg": 5,
                        "individual": False,
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
            self.assertEqual(summary["agent"], "torch-dlinear-ppo")
            self.assertIn("train_update_metrics", summary)
            self.assertIn("approx_kl", summary["train_update_metrics"]["overall"])
            self.assertIn("explained_variance", summary["train_update_metrics"]["overall"])


if __name__ == "__main__":
    unittest.main()
