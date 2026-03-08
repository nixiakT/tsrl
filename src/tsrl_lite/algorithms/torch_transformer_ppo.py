from __future__ import annotations

import numpy as np

from tsrl_lite.algorithms.common import EpisodeBatch
from tsrl_lite.encoders.base import BaseStateEncoder
from tsrl_lite.registry import register_agent
from tsrl_lite.state import TimeSeriesState

try:
    import torch
    from torch import optim
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    optim = None

from tsrl_lite.networks import TorchTransformerActorCriticNetwork
from tsrl_lite.algorithms.torch_ppo_common import prepare_torch_ppo_batch, run_torch_ppo_update


if torch is not None and TorchTransformerActorCriticNetwork is not None:

    @register_agent("torch-transformer-ppo")
    class TorchTransformerPPOAgent:
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
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
            clip_epsilon: float = 0.2,
            update_epochs: int = 4,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            mini_batch_size: int | None = None,
            normalize_advantages: bool = True,
            shuffle_minibatches: bool = True,
            target_kl: float | None = None,
            value_clip_epsilon: float | None = None,
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
            resolved_device = device
            if device == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(resolved_device)

            if len(encoder.observation_shape) != 2:
                raise ValueError("torch-transformer-ppo requires a sequence-shaped encoder output")
            sequence_length, input_dim = encoder.observation_shape
            self.sequence_length = sequence_length
            self.input_dim = input_dim

            torch.manual_seed(seed)
            self.network = TorchTransformerActorCriticNetwork(
                input_dim=input_dim,
                hidden_size=hidden_size,
                action_dim=action_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_len=sequence_length,
            ).to(self.device)
            self.optimizer = optim.Adam(
                [
                    {"params": self.network.input_proj.parameters(), "lr": policy_lr},
                    {"params": self.network.encoder.parameters(), "lr": policy_lr},
                    {"params": self.network.policy_head.parameters(), "lr": policy_lr},
                    {"params": self.network.value_head.parameters(), "lr": value_lr},
                    {"params": self.network.norm.parameters(), "lr": value_lr},
                ]
            )

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
            prepared_batch = prepare_torch_ppo_batch(
                batch=batch,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                device=self.device,
                normalize_advantages=self.normalize_advantages,
            )
            return run_torch_ppo_update(
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

        def save(self, path: str) -> None:
            self.network.save(path, optimizer=self.optimizer)

        def load(self, path: str) -> None:
            network, optimizer_state = TorchTransformerActorCriticNetwork.load(path, map_location=self.device)
            self.network = network.to(self.device)
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)
