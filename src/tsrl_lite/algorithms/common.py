from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EpisodeBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    action_probs: np.ndarray


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=float)
    gae = 0.0
    next_value = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        mask = 0.0 if dones[idx] else 1.0
        delta = rewards[idx] + gamma * next_value * mask - values[idx]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[idx] = gae
        next_value = values[idx]
    returns = advantages + values
    return advantages, returns
