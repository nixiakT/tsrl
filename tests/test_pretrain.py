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

if TORCH_AVAILABLE:
    import torch

    from tsrl_lite.algorithms.torch_patchtst_ppo import TorchPatchTSTPPOAgent
    from tsrl_lite.encoders.sequence import SequenceWindowEncoder
    from tsrl_lite.pretrain import pretrain_patchtst_backbone


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class PatchTSTPretrainingTest(unittest.TestCase):
    def _write_patchtst_config(self, root: Path) -> Path:
        config_path = root / "patchtst_config.json"
        config = {
            "experiment_name": "patchtst_pretrain_smoke",
            "seed": 67,
            "data": {
                "source": "synthetic",
                "window_size": 16,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "synthetic": {
                    "steps": 600,
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
                    "aux_loss_coef": 0.1,
                    "aux_mask_ratio": 0.4,
                    "device": "cpu"
                }
            },
            "trainer": {
                "episodes": 3,
                "eval_episodes": 1,
                "log_interval": 3,
                "checkpoint_dir": str(root / "run")
            }
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def _write_patchtst_portfolio_config(self, root: Path) -> Path:
        config_path = root / "patchtst_portfolio_config.json"
        config = {
            "experiment_name": "patchtst_portfolio_pretrain_smoke",
            "seed": 71,
            "data": {
                "source": "synthetic",
                "window_size": 16,
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "synthetic": {
                    "steps": 700,
                    "assets": 3,
                    "drift": 0.0011,
                    "volatility": 0.007,
                    "seasonality": 0.002,
                    "correlation": 0.35
                }
            },
            "env": {
                "name": "portfolio-v0",
                "reward_scale": 1.0,
                "episode_horizon": 64,
                "random_reset": True,
                "params": {
                    "trading_cost": 0.0005,
                    "allocation_candidates": [
                        [0.0, 0.0, 0.0],
                        [0.3333333333, 0.3333333333, 0.3333333333],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ]
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
                    "channel_independent": True,
                    "aux_loss_coef": 0.1,
                    "aux_mask_ratio": 0.4,
                    "device": "cpu"
                }
            },
            "trainer": {
                "episodes": 3,
                "eval_episodes": 1,
                "log_interval": 3,
                "checkpoint_dir": str(root / "portfolio_run")
            }
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    def test_pretrain_patchtst_backbone_produces_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = self._write_patchtst_config(tmp_path)

            artifacts, summary = pretrain_patchtst_backbone(
                config_path=config_path,
                output_dir=tmp_path / "pretrain",
                epochs=2,
                batch_size=16,
            )

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertGreaterEqual(summary["best_val_accuracy"], 0.0)
            self.assertLessEqual(summary["best_val_accuracy"], 1.0)
            self.assertEqual(summary["task"], "regime_classification")
            self.assertEqual(summary["selection_metric"], "val_accuracy")

    def test_pretrain_patchtst_supports_future_return_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = self._write_patchtst_config(tmp_path)

            artifacts, summary = pretrain_patchtst_backbone(
                config_path=config_path,
                output_dir=tmp_path / "pretrain_regression",
                epochs=2,
                batch_size=16,
                task_type="future_return_regression",
            )

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertEqual(summary["task"], "future_return_regression")
            self.assertEqual(summary["selection_metric"], "val_mae")
            self.assertGreaterEqual(summary["best_val_mae"], 0.0)
            self.assertIn("val_rmse", summary["best_metrics"])
            self.assertIn("val_correlation", summary["best_metrics"])

    def test_pretrain_patchtst_supports_joint_multitask_pretraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = self._write_patchtst_config(tmp_path)

            artifacts, summary = pretrain_patchtst_backbone(
                config_path=config_path,
                output_dir=tmp_path / "pretrain_joint",
                epochs=2,
                batch_size=16,
                task_type="joint_regime_return",
            )

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertEqual(summary["task"], "joint_regime_return")
            self.assertEqual(summary["task_heads"], ["regime_classification", "future_return_regression"])
            self.assertEqual(summary["selection_metric"], "val_joint_loss")
            self.assertGreaterEqual(summary["best_val_joint_loss"], 0.0)
            self.assertIn("val_accuracy", summary["best_metrics"])
            self.assertIn("val_mae", summary["best_metrics"])
            self.assertIn("val_rmse", summary["best_metrics"])
            self.assertIn("val_correlation", summary["best_metrics"])

            payload = torch.load(artifacts.checkpoint_path, map_location="cpu")
            self.assertEqual(payload["task"], "joint_regime_return")
            self.assertEqual(payload["task_heads"], ["regime_classification", "future_return_regression"])
            self.assertEqual(payload["best_selection_metric"], "val_joint_loss")

    def test_pretrain_patchtst_supports_multivariate_future_return_vector_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = self._write_patchtst_portfolio_config(tmp_path)

            artifacts, summary = pretrain_patchtst_backbone(
                config_path=config_path,
                output_dir=tmp_path / "pretrain_vector",
                epochs=2,
                batch_size=16,
                task_type="future_return_vector_regression",
            )

            self.assertTrue(artifacts.checkpoint_path.exists())
            self.assertEqual(summary["task"], "future_return_vector_regression")
            self.assertEqual(summary["task_heads"], ["future_return_vector_regression"])
            self.assertEqual(summary["selection_metric"], "val_mae")
            self.assertGreaterEqual(summary["best_val_mae"], 0.0)
            self.assertEqual(summary["regression_target_dim"], 3)
            self.assertIn("val_rmse", summary["best_metrics"])
            self.assertIn("val_correlation", summary["best_metrics"])

            payload = torch.load(artifacts.checkpoint_path, map_location="cpu")
            self.assertEqual(payload["task"], "future_return_vector_regression")
            self.assertEqual(payload["task_heads"], ["future_return_vector_regression"])
            self.assertEqual(payload["regression_target_dim"], 3)
            self.assertEqual(payload["label_count"], 3)

    def test_patchtst_agent_loads_and_freezes_pretrained_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = self._write_patchtst_config(tmp_path)
            artifacts, _ = pretrain_patchtst_backbone(
                config_path=config_path,
                output_dir=tmp_path / "pretrain",
                epochs=1,
                batch_size=16,
            )

            encoder = SequenceWindowEncoder(window_size=16, agent_feature_dim=1)
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
                mini_batch_size=8,
                target_kl=0.05,
                value_clip_epsilon=0.2,
                aux_loss_coef=0.1,
                aux_mask_ratio=0.4,
                aux_epochs=1,
                pretrained_backbone_path=str(artifacts.checkpoint_path),
                freeze_backbone=True,
                device="cpu",
                seed=71,
            )

            payload = torch.load(artifacts.checkpoint_path, map_location="cpu")
            loaded_backbone_state = payload["backbone_state_dict"]
            agent_backbone_state = agent.network.backbone_state_dict()
            self.assertEqual(set(loaded_backbone_state), set(agent_backbone_state))
            for key, value in loaded_backbone_state.items():
                self.assertTrue(torch.equal(value, agent_backbone_state[key]))
            self.assertTrue(all(not parameter.requires_grad for parameter in agent.network.backbone_parameters()))


if __name__ == "__main__":
    unittest.main()
