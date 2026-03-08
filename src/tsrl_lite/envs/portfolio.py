from __future__ import annotations

import numpy as np

from tsrl_lite.envs.base import BaseTimeSeriesEnv
from tsrl_lite.registry import register_env
from tsrl_lite.state import TimeSeriesState


def _default_allocation_candidates(asset_count: int) -> list[list[float]]:
    candidates: list[list[float]] = [[0.0] * asset_count]
    equal_weight = [1.0 / float(asset_count)] * asset_count
    candidates.append(equal_weight)
    for asset_index in range(asset_count):
        weights = [0.0] * asset_count
        weights[asset_index] = 1.0
        candidates.append(weights)
    return candidates


@register_env("portfolio-v0")
class PortfolioEnv(BaseTimeSeriesEnv):
    def __init__(
        self,
        prices: np.ndarray,
        window_size: int,
        trading_cost: float,
        allocation_candidates: list[list[float]] | None = None,
        reward_scale: float = 1.0,
        episode_horizon: int | None = None,
        random_reset: bool = False,
        seed: int = 7,
        index_offset: int = 0,
    ) -> None:
        price_array = np.asarray(prices, dtype=float)
        if price_array.ndim != 2:
            raise ValueError("portfolio-v0 expects prices with shape [time, assets]")
        if price_array.shape[0] <= window_size + 2:
            raise ValueError("prices must be longer than window_size + 2")

        self.prices = price_array
        self.window_size = int(window_size)
        self.asset_count = int(price_array.shape[1])
        self.trading_cost = float(trading_cost)
        self.reward_scale = float(reward_scale)
        self.episode_horizon = episode_horizon
        self.random_reset = random_reset
        self.rng = np.random.default_rng(seed)
        self.index_offset = int(index_offset)

        candidate_payload = allocation_candidates or _default_allocation_candidates(self.asset_count)
        self.allocation_candidates = np.asarray(candidate_payload, dtype=float)
        if self.allocation_candidates.ndim != 2 or self.allocation_candidates.shape[1] != self.asset_count:
            raise ValueError("allocation_candidates must be a 2D list with one weight per asset")

        self.action_dim = int(self.allocation_candidates.shape[0])
        self.agent_feature_dim = self.asset_count
        self.window_feature_dim = self.asset_count

        self.start_step = self.window_size - 1
        self.current_step = self.start_step
        self.last_step = len(self.prices) - 2
        self.weights = np.zeros(self.asset_count, dtype=float)
        self.prev_reward = 0.0
        self.equity = 1.0

    def reset(self, random_start: bool = False) -> TimeSeriesState:
        min_start = self.window_size - 1
        max_terminal = len(self.prices) - 2
        should_randomize = random_start and self.random_reset and self.episode_horizon is not None

        if should_randomize:
            max_start = max(min_start, max_terminal - self.episode_horizon)
            self.start_step = int(self.rng.integers(min_start, max_start + 1))
            self.last_step = min(max_terminal, self.start_step + self.episode_horizon)
        else:
            self.start_step = min_start
            self.last_step = (
                min(max_terminal, self.start_step + self.episode_horizon)
                if self.episode_horizon is not None and not random_start
                else max_terminal
            )

        self.current_step = self.start_step
        self.weights = np.zeros(self.asset_count, dtype=float)
        self.prev_reward = 0.0
        self.equity = 1.0
        return self._state()

    def step(self, action: int) -> tuple[TimeSeriesState, float, bool, dict]:
        if action < 0 or action >= self.action_dim:
            raise IndexError(f"action {action} is out of range")

        current_prices = self.prices[self.current_step]
        next_prices = self.prices[self.current_step + 1]
        asset_returns = (next_prices - current_prices) / np.maximum(current_prices, 1e-8)
        target_weights = self.allocation_candidates[action]
        turnover = float(np.sum(np.abs(target_weights - self.weights)))
        portfolio_return = float(np.dot(target_weights, asset_returns))
        reward = (portfolio_return - (self.trading_cost * turnover)) * self.reward_scale

        self.weights = target_weights.copy()
        self.prev_reward = reward
        self.equity *= 1.0 + reward
        self.current_step += 1

        done = self.current_step >= self.last_step
        state = self._state()
        info = {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "equity": self.equity,
            "gross_exposure": float(np.sum(np.abs(self.weights))),
            "active_assets": float(np.sum(np.abs(self.weights) > 1e-8)),
            "step": self.current_step,
            "task": "portfolio",
        }
        return state, float(reward), done, info

    def _state(self) -> TimeSeriesState:
        window = self.prices[self.current_step - self.window_size + 1 : self.current_step + 1]
        return TimeSeriesState(
            window=window,
            agent_features=self.weights.copy(),
            step=self.current_step,
            context={
                "task": "portfolio",
                "equity": self.equity,
                "global_step": self.index_offset + self.current_step,
            },
        )
