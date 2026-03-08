from __future__ import annotations

from pathlib import Path

import numpy as np

from tsrl_lite.utils import softmax


class LinearActorCriticNetwork:
    def __init__(self, observation_dim: int, action_dim: int, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        scale = 0.05
        self.policy_weights = rng.normal(0.0, scale, size=(observation_dim, action_dim))
        self.policy_bias = np.zeros(action_dim, dtype=float)
        self.value_weights = rng.normal(0.0, scale, size=observation_dim)
        self.value_bias = 0.0

    def policy(self, observation: np.ndarray) -> np.ndarray:
        logits = observation @ self.policy_weights + self.policy_bias
        return softmax(logits)

    def value(self, observation: np.ndarray) -> float:
        return float(observation @ self.value_weights + self.value_bias)

    def act(
        self,
        observation: np.ndarray,
        rng: np.random.Generator,
        greedy: bool = False,
    ) -> tuple[int, np.ndarray, float]:
        probs = self.policy(observation)
        if greedy:
            action = int(np.argmax(probs))
        else:
            action = int(rng.choice(self.action_dim, p=probs))
        return action, probs, self.value(observation)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            policy_weights=self.policy_weights,
            policy_bias=self.policy_bias,
            value_weights=self.value_weights,
            value_bias=np.asarray([self.value_bias], dtype=float),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearActorCriticNetwork":
        payload = np.load(Path(path))
        policy_weights = payload["policy_weights"]
        policy_bias = payload["policy_bias"]
        value_weights = payload["value_weights"]
        value_bias = float(payload["value_bias"][0])

        network = cls(
            observation_dim=policy_weights.shape[0],
            action_dim=policy_weights.shape[1],
            seed=0,
        )
        network.policy_weights = policy_weights
        network.policy_bias = policy_bias
        network.value_weights = value_weights
        network.value_bias = value_bias
        return network
