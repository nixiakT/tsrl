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

    from tsrl_lite.algorithms.torch_patchtst_ppo import TorchPatchTSTPPOAgent
    from tsrl_lite.networks.torch_patchtst import TorchPatchTSTActorCriticNetwork


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class TorchPatchTSTSmokeTest(unittest.TestCase):
    def test_torch_patchtst_network_builds_temporal_patches(self) -> None:
        network = TorchPatchTSTActorCriticNetwork(
            input_dim=3,
            sequence_length=16,
            hidden_size=8,
            action_dim=4,
            patch_len=4,
            stride=2,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            use_cls_token=True,
        )
        sequence_batch = torch.randn(2, 16, 3)
        logits, value = network(sequence_batch)

        self.assertEqual(network.num_patches, 7)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(value.shape), (2,))

    def test_torch_patchtst_masked_reconstruction_loss_is_positive(self) -> None:
        network = TorchPatchTSTActorCriticNetwork(
            input_dim=2,
            sequence_length=12,
            hidden_size=8,
            action_dim=3,
            patch_len=4,
            stride=2,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            use_cls_token=True,
        )
        sequence_batch = torch.randn(3, 12, 2)
        loss, realized_mask_ratio = network.masked_patch_reconstruction_loss(sequence_batch, mask_ratio=0.5)

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(realized_mask_ratio, 0.0)
        self.assertLessEqual(realized_mask_ratio, 1.0)

    def test_torch_patchtst_update_reports_stability_metrics(self) -> None:
        encoder = SequenceWindowEncoder(window_size=8, agent_feature_dim=1)
        agent = TorchPatchTSTPPOAgent(
            encoder=encoder,
            action_dim=3,
            gamma=0.99,
            gae_lambda=0.95,
            policy_lr=0.001,
            value_lr=0.001,
            gradient_clip=1.0,
            hidden_size=16,
            patch_len=4,
            stride=2,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            use_cls_token=True,
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
            seed=43,
        )

        observations = np.random.default_rng(43).normal(
            size=(6, *encoder.observation_shape)
        ).astype(np.float32)
        batch = EpisodeBatch(
            observations=observations,
            actions=np.asarray([0, 0, 1, 1, 2, 2], dtype=int),
            rewards=np.asarray([1.0, 0.5, -0.2, 0.1, 0.3, -0.1], dtype=float),
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

    def test_torch_patchtst_update_reports_auxiliary_metrics(self) -> None:
        encoder = SequenceWindowEncoder(window_size=8, agent_feature_dim=1)
        agent = TorchPatchTSTPPOAgent(
            encoder=encoder,
            action_dim=3,
            gamma=0.99,
            gae_lambda=0.95,
            policy_lr=0.001,
            value_lr=0.001,
            gradient_clip=1.0,
            hidden_size=16,
            patch_len=4,
            stride=2,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            use_cls_token=True,
            clip_epsilon=0.2,
            update_epochs=2,
            entropy_coef=0.01,
            value_coef=0.5,
            mini_batch_size=2,
            normalize_advantages=True,
            shuffle_minibatches=False,
            target_kl=0.05,
            value_clip_epsilon=0.2,
            aux_loss_coef=0.1,
            aux_mask_ratio=0.5,
            aux_epochs=2,
            device="cpu",
            seed=53,
        )

        observations = np.random.default_rng(53).normal(
            size=(6, *encoder.observation_shape)
        ).astype(np.float32)
        batch = EpisodeBatch(
            observations=observations,
            actions=np.asarray([0, 0, 1, 1, 2, 2], dtype=int),
            rewards=np.asarray([1.0, 0.5, -0.2, 0.1, 0.3, -0.1], dtype=float),
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

        self.assertIn("aux_reconstruction_loss", metrics)
        self.assertIn("aux_mask_ratio", metrics)
        self.assertIn("aux_update_steps", metrics)
        self.assertGreater(metrics["aux_reconstruction_loss"], 0.0)
        self.assertGreater(metrics["aux_update_steps"], 0.0)

    def test_torch_patchtst_training_produces_pt_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "torch_patchtst_config.json"
            output_dir = tmp_path / "run"
            config = {
                "experiment_name": "torch_patchtst_smoke",
                "seed": 47,
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
                    "name": "torch-patchtst-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.001,
                    "value_lr": 0.001,
                    "gradient_clip": 1.0,
                    "params": {
                        "hidden_size": 16,
                        "patch_len": 4,
                        "stride": 2,
                        "num_layers": 1,
                        "num_heads": 2,
                        "dropout": 0.0,
                        "use_cls_token": True,
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
            self.assertEqual(summary["agent"], "torch-patchtst-ppo")
            self.assertIn("train_update_metrics", summary)
            self.assertIn("approx_kl", summary["train_update_metrics"]["overall"])
            self.assertIn("explained_variance", summary["train_update_metrics"]["overall"])

    def test_torch_patchtst_aux_training_reports_aux_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "torch_patchtst_aux_config.json"
            output_dir = tmp_path / "run"
            config = {
                "experiment_name": "torch_patchtst_aux_smoke",
                "seed": 59,
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
                    "name": "torch-patchtst-ppo",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "policy_lr": 0.001,
                    "value_lr": 0.001,
                    "gradient_clip": 1.0,
                    "params": {
                        "hidden_size": 16,
                        "patch_len": 4,
                        "stride": 2,
                        "num_layers": 1,
                        "num_heads": 2,
                        "dropout": 0.0,
                        "use_cls_token": True,
                        "clip_epsilon": 0.2,
                        "update_epochs": 2,
                        "entropy_coef": 0.01,
                        "value_coef": 0.5,
                        "mini_batch_size": 8,
                        "target_kl": 0.05,
                        "value_clip_epsilon": 0.2,
                        "aux_loss_coef": 0.1,
                        "aux_mask_ratio": 0.4,
                        "aux_epochs": 1,
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

            _, summary = train_experiment(config_path=config_path, output_dir=output_dir)

            self.assertIn("train_update_metrics", summary)
            self.assertIn("aux_reconstruction_loss", summary["train_update_metrics"]["overall"])
            self.assertIn("aux_mask_ratio", summary["train_update_metrics"]["overall"])


if __name__ == "__main__":
    unittest.main()
