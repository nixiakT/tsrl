from __future__ import annotations

from pathlib import Path

import numpy as np

from tsrl_lite.algorithms.common import EpisodeBatch
from tsrl_lite.algorithms.torch_ppo_common import prepare_torch_ppo_batch, run_torch_ppo_update
from tsrl_lite.encoders.base import BaseStateEncoder
from tsrl_lite.registry import register_agent
from tsrl_lite.state import TimeSeriesState

try:
    import torch
    from torch import optim
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    optim = None

from tsrl_lite.networks import TorchPatchTSTActorCriticNetwork


if torch is not None and TorchPatchTSTActorCriticNetwork is not None:

    @register_agent("torch-patchtst-ppo")
    class TorchPatchTSTPPOAgent:
        checkpoint_suffix = ".pt"

        def __init__(
            self,
            encoder: BaseStateEncoder,
            action_dim: int,
            gamma: float,
            gae_lambda: float,
            policy_lr: float,
            value_lr: float,
            gradient_clip: float,
            hidden_size: int = 64,
            patch_len: int = 8,
            stride: int = 4,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
            use_cls_token: bool = True,
            channel_independent: bool = False,
            clip_epsilon: float = 0.2,
            update_epochs: int = 4,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            mini_batch_size: int | None = None,
            normalize_advantages: bool = True,
            shuffle_minibatches: bool = True,
            target_kl: float | None = None,
            value_clip_epsilon: float | None = None,
            aux_loss_coef: float = 0.0,
            aux_mask_ratio: float = 0.4,
            aux_epochs: int = 1,
            pretrained_backbone_path: str | None = None,
            freeze_backbone: bool = False,
            unfreeze_backbone_after_updates: int | None = None,
            strict_backbone_config: bool = True,
            device: str = "auto",
            seed: int = 7,
        ) -> None:
            self.encoder = encoder
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.gradient_clip = gradient_clip
            self.clip_epsilon = clip_epsilon
            self.update_epochs = max(1, int(update_epochs))
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
            self.mini_batch_size = None if mini_batch_size is None else max(1, int(mini_batch_size))
            self.normalize_advantages = bool(normalize_advantages)
            self.shuffle_minibatches = bool(shuffle_minibatches)
            self.target_kl = None if target_kl is None else float(target_kl)
            self.value_clip_epsilon = None if value_clip_epsilon is None else float(value_clip_epsilon)
            self.aux_loss_coef = max(0.0, float(aux_loss_coef))
            self.aux_mask_ratio = float(aux_mask_ratio)
            self.aux_epochs = max(1, int(aux_epochs))
            self.pretrained_backbone_path = pretrained_backbone_path
            self.freeze_backbone = bool(freeze_backbone)
            self.strict_backbone_config = bool(strict_backbone_config)
            self.unfreeze_backbone_after_updates = (
                None
                if unfreeze_backbone_after_updates is None
                else max(0, int(unfreeze_backbone_after_updates))
            )
            self.update_count = 0
            self.pretrained_backbone_metadata: dict[str, object] = {
                "loaded": False,
                "path": None,
            }
            resolved_device = device
            if device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(resolved_device)

            if len(encoder.observation_shape) != 2:
                raise ValueError("torch-patchtst-ppo requires a sequence-shaped encoder output")
            sequence_length, input_dim = encoder.observation_shape
            self.sequence_length = sequence_length
            self.input_dim = input_dim

            torch.manual_seed(seed)
            self.network = TorchPatchTSTActorCriticNetwork(
                input_dim=input_dim,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                action_dim=action_dim,
                patch_len=patch_len,
                stride=stride,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_cls_token=use_cls_token,
                channel_independent=channel_independent,
            ).to(self.device)
            self.expected_backbone_config = {
                "input_dim": input_dim,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "patch_len": patch_len,
                "stride": stride,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "use_cls_token": bool(use_cls_token),
                "channel_independent": bool(channel_independent),
            }
            if self.pretrained_backbone_path:
                payload = torch.load(self.pretrained_backbone_path, map_location=self.device)
                if isinstance(payload, dict):
                    self._validate_pretrained_backbone_payload(payload)
                backbone_state_dict = payload.get("backbone_state_dict", payload.get("state_dict", payload))
                if not isinstance(backbone_state_dict, dict):
                    raise ValueError("pretrained_backbone_path must point to a valid PatchTST backbone checkpoint")
                self.network.load_backbone_state_dict(backbone_state_dict)
                self.pretrained_backbone_metadata = {
                    "loaded": True,
                    "path": str(Path(self.pretrained_backbone_path)),
                    "task": payload.get("task") if isinstance(payload, dict) else None,
                    "task_heads": payload.get("task_heads") if isinstance(payload, dict) else None,
                    "best_selection_metric": payload.get("best_selection_metric") if isinstance(payload, dict) else None,
                    "best_selection_value": payload.get("best_selection_value") if isinstance(payload, dict) else None,
                    "regression_target_dim": payload.get("regression_target_dim") if isinstance(payload, dict) else None,
                }
            self.backbone_parameters = list(self.network.backbone_parameters())
            self.backbone_frozen = bool(self.freeze_backbone)
            self._set_backbone_trainable(not self.backbone_frozen)
            parameter_groups = []
            if self.backbone_parameters:
                parameter_groups.append({"params": self.backbone_parameters, "lr": policy_lr})
            parameter_groups.append({"params": self.network.policy_head.parameters(), "lr": policy_lr})
            parameter_groups.append({"params": self.network.value_head.parameters(), "lr": value_lr})
            self.optimizer = optim.Adam(parameter_groups)

        def _set_backbone_trainable(self, trainable: bool) -> None:
            for parameter in self.backbone_parameters:
                parameter.requires_grad_(trainable)
            self.has_trainable_backbone = bool(trainable)

        def _validate_pretrained_backbone_payload(self, payload: dict[str, object]) -> None:
            if not self.strict_backbone_config:
                return
            backbone_config = payload.get("backbone_config")
            if not isinstance(backbone_config, dict):
                return
            mismatches = []
            for key, expected_value in self.expected_backbone_config.items():
                loaded_value = backbone_config.get(key)
                if loaded_value != expected_value:
                    mismatches.append(f"{key}={loaded_value!r} (expected {expected_value!r})")
            if mismatches:
                mismatch_text = ", ".join(mismatches)
                raise ValueError(f"incompatible PatchTST backbone checkpoint: {mismatch_text}")

        def _maybe_unfreeze_backbone(self) -> bool:
            if not self.backbone_frozen:
                return False
            if self.unfreeze_backbone_after_updates is None:
                return False
            if self.update_count < self.unfreeze_backbone_after_updates:
                return False
            self.backbone_frozen = False
            self._set_backbone_trainable(True)
            return True

        def act(
            self,
            state: TimeSeriesState,
            rng: np.random.Generator,
            greedy: bool = False,
        ) -> tuple[int, np.ndarray, float, np.ndarray]:
            observation = self.encoder.encode(state).astype(np.float32)
            observation_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                logits, value = self.network(observation_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            if greedy:
                action = int(np.argmax(probs))
            else:
                action = int(rng.choice(len(probs), p=probs))
            return action, probs, float(value.item()), observation

        def update(self, batch: EpisodeBatch) -> dict[str, float]:
            backbone_unfrozen_this_update = self._maybe_unfreeze_backbone()
            prepared_batch = prepare_torch_ppo_batch(
                batch=batch,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                device=self.device,
                normalize_advantages=self.normalize_advantages,
            )
            metrics = run_torch_ppo_update(
                network=self.network,
                optimizer=self.optimizer,
                prepared_batch=prepared_batch,
                update_epochs=self.update_epochs,
                gradient_clip=self.gradient_clip,
                clip_epsilon=self.clip_epsilon,
                entropy_coef=self.entropy_coef,
                value_coef=self.value_coef,
                mini_batch_size=self.mini_batch_size,
                shuffle_minibatches=self.shuffle_minibatches,
                value_clip_epsilon=self.value_clip_epsilon,
                target_kl=self.target_kl,
            )
            self.update_count += 1
            metrics["backbone_frozen"] = 1.0 if self.backbone_frozen else 0.0
            metrics["backbone_trainable"] = 1.0 if self.has_trainable_backbone else 0.0
            metrics["backbone_unfrozen_this_update"] = 1.0 if backbone_unfrozen_this_update else 0.0
            metrics["backbone_update_count"] = float(self.update_count)
            if self.aux_loss_coef <= 0.0 or not self.has_trainable_backbone:
                return metrics

            observations = prepared_batch["observations"]
            batch_size = int(observations.shape[0])
            resolved_batch_size = (
                batch_size if self.mini_batch_size is None else max(1, min(int(self.mini_batch_size), batch_size))
            )
            aux_loss_total = 0.0
            aux_mask_ratio_total = 0.0
            aux_steps = 0

            self.network.train()
            for _ in range(self.aux_epochs):
                if self.shuffle_minibatches:
                    order = torch.randperm(batch_size, device=observations.device)
                else:
                    order = torch.arange(batch_size, device=observations.device)
                for start_index in range(0, batch_size, resolved_batch_size):
                    batch_index = order[start_index : start_index + resolved_batch_size]
                    aux_loss, realized_mask_ratio = self.network.masked_patch_reconstruction_loss(
                        observations[batch_index],
                        mask_ratio=self.aux_mask_ratio,
                    )
                    self.optimizer.zero_grad()
                    (self.aux_loss_coef * aux_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
                    self.optimizer.step()
                    aux_steps += 1
                    aux_loss_total += float(aux_loss.item())
                    aux_mask_ratio_total += float(realized_mask_ratio)

            metrics["aux_reconstruction_loss"] = aux_loss_total / max(aux_steps, 1)
            metrics["aux_mask_ratio"] = aux_mask_ratio_total / max(aux_steps, 1)
            metrics["aux_update_steps"] = float(aux_steps)
            return metrics

        def save(self, path: str) -> None:
            self.network.save(path, optimizer=self.optimizer)

        def load(self, path: str) -> None:
            network, optimizer_state = TorchPatchTSTActorCriticNetwork.load(path, map_location=self.device)
            self.network = network.to(self.device)
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

        def metadata(self) -> dict[str, object]:
            return {
                "pretrained_backbone": dict(self.pretrained_backbone_metadata),
                "backbone_config": dict(self.expected_backbone_config),
            }
