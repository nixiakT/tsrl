from __future__ import annotations

import numpy as np

from tsrl_lite.algorithms.common import EpisodeBatch, compute_gae
from tsrl_lite.encoders.base import BaseStateEncoder
from tsrl_lite.networks import LinearActorCriticNetwork
from tsrl_lite.registry import register_agent
from tsrl_lite.state import TimeSeriesState
from tsrl_lite.utils import clip_gradient


@register_agent("linear-actor-critic")
class ActorCriticAgent:
    def __init__(
        self,
        encoder: BaseStateEncoder,
        action_dim: int,
        gamma: float,
        gae_lambda: float,
        policy_lr: float,
        value_lr: float,
        gradient_clip: float,
        seed: int = 7,
    ) -> None:
        self.encoder = encoder
        self.network = LinearActorCriticNetwork(
            observation_dim=encoder.observation_dim,
            action_dim=action_dim,
            seed=seed,
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.gradient_clip = gradient_clip

    def act(
        self,
        state: TimeSeriesState,
        rng: np.random.Generator,
        greedy: bool = False,
    ) -> tuple[int, np.ndarray, float, np.ndarray]:
        observation = self.encoder.encode(state)
        action, probs, value = self.network.act(observation, rng, greedy=greedy)
        return action, probs, value, observation

    def update(self, batch: EpisodeBatch) -> dict[str, float]:
        advantages, returns = compute_gae(
            rewards=batch.rewards,
            values=batch.values,
            dones=batch.dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_grad_w = np.zeros_like(self.network.policy_weights)
        policy_grad_b = np.zeros_like(self.network.policy_bias)
        value_grad_w = np.zeros_like(self.network.value_weights)
        value_grad_b = 0.0
        entropy = 0.0
        log_prob_sum = 0.0

        for obs, action, advantage, ret in zip(
            batch.observations,
            batch.actions,
            advantages,
            returns,
            strict=True,
        ):
            probs = self.network.policy(obs)
            value = self.network.value(obs)
            one_hot = np.zeros_like(probs)
            one_hot[action] = 1.0
            policy_delta = (one_hot - probs) * advantage
            policy_grad_w += np.outer(obs, policy_delta)
            policy_grad_b += policy_delta

            value_error = ret - value
            value_grad_w += obs * value_error
            value_grad_b += value_error

            entropy += float(-np.sum(probs * np.log(probs + 1e-8)))
            log_prob_sum += float(np.log(probs[action] + 1e-8))

        policy_grad_w, policy_grad_b, value_grad_w, value_grad_b_arr = clip_gradient(
            policy_grad_w,
            policy_grad_b,
            value_grad_w,
            np.asarray([value_grad_b], dtype=float),
            max_norm=self.gradient_clip,
        )
        value_grad_b = float(value_grad_b_arr[0])

        scale = 1.0 / max(len(batch.rewards), 1)
        self.network.policy_weights += self.policy_lr * policy_grad_w * scale
        self.network.policy_bias += self.policy_lr * policy_grad_b * scale
        self.network.value_weights += self.value_lr * value_grad_w * scale
        self.network.value_bias += self.value_lr * value_grad_b * scale

        value_predictions = np.asarray([self.network.value(obs) for obs in batch.observations], dtype=float)
        value_loss = float(np.mean((returns - value_predictions) ** 2))
        policy_loss = float(-(log_prob_sum * scale))
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy * scale,
            "advantage_mean": float(np.mean(advantages)),
        }

    def save(self, path: str) -> None:
        self.network.save(path)

    def load(self, path: str) -> None:
        self.network = LinearActorCriticNetwork.load(path)
