from __future__ import annotations

import numpy as np

from tsrl_lite.algorithms import ActorCriticAgent
from tsrl_lite.envs.base import BaseTimeSeriesEnv
from tsrl_lite.utils import collect_numeric_info, compute_trading_episode_metrics


def evaluate_agent(
    agent: ActorCriticAgent,
    env: BaseTimeSeriesEnv,
    episodes: int,
    seed: int,
    random_start: bool = False,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    lengths: list[int] = []
    terminal_scores: list[float] = []
    episode_metrics: dict[str, list[float]] = {}
    info_episode_means: dict[str, list[float]] = {}

    for _ in range(episodes):
        state = env.reset(random_start=random_start)
        done = False
        total_reward = 0.0
        steps = 0
        score = 0.0
        reward_trace: list[float] = []
        info_metrics: dict[str, list[float]] = {}
        while not done:
            action, _, _, _ = agent.act(state, rng, greedy=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            reward_trace.append(float(reward))
            collect_numeric_info(info, info_metrics)
            if "equity" in info:
                score = float(info["equity"])
            elif "accuracy" in info:
                score = float(info["accuracy"])
            else:
                score = float(total_reward)

        rewards.append(total_reward)
        lengths.append(steps)
        terminal_scores.append(score)
        if "equity" in info_metrics:
            trading_metrics = compute_trading_episode_metrics(np.asarray(reward_trace, dtype=float), info_metrics)
            for key, value in trading_metrics.items():
                episode_metrics.setdefault(key, []).append(float(value))
        for key, values in info_metrics.items():
            info_episode_means.setdefault(key, []).append(float(np.mean(values)))

    metrics = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "mean_terminal_score": float(np.mean(terminal_scores)),
    }
    for key, values in episode_metrics.items():
        metrics[f"mean_{key}"] = float(np.mean(values))
    for key, values in info_episode_means.items():
        metrics[f"mean_info_{key}"] = float(np.mean(values))
    return metrics
