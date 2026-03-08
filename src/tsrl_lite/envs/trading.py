from __future__ import annotations

import numpy as np

from tsrl_lite.envs.base import BaseTimeSeriesEnv
from tsrl_lite.registry import register_env
from tsrl_lite.state import TimeSeriesState


@register_env("trading-v0")
class TradingEnv(BaseTimeSeriesEnv):
    def __init__(
        self,
        prices: np.ndarray,
        window_size: int,
        positions: list[float],
        trading_cost: float,
        reward_scale: float = 1.0,
        episode_horizon: int | None = None,
        random_reset: bool = False,
        seed: int = 7,
        index_offset: int = 0,
    ) -> None:
        if len(prices) <= window_size + 2:
            raise ValueError("prices must be longer than window_size + 2")

        self.prices = np.asarray(prices, dtype=float)
        self.window_size = int(window_size)
        self.positions = np.asarray(positions, dtype=float)
        self.trading_cost = float(trading_cost)
        self.reward_scale = float(reward_scale)
        self.episode_horizon = episode_horizon
        self.random_reset = random_reset
        self.rng = np.random.default_rng(seed)
        self.index_offset = int(index_offset)

        self.action_dim = len(self.positions)
        self.agent_feature_dim = 1
        self.window_feature_dim = 1

        self.start_step = self.window_size - 1
        self.current_step = self.start_step
        self.last_step = len(self.prices) - 2
        self.position = 0.0
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
        self.position = 0.0
        self.prev_reward = 0.0
        self.equity = 1.0
        return self._state()

    def step(self, action: int) -> tuple[TimeSeriesState, float, bool, dict]:
        if action < 0 or action >= self.action_dim:
            raise IndexError(f"action {action} is out of range")

        current_price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1]
        market_return = (next_price - current_price) / current_price
        target_position = float(self.positions[action])
        turnover = abs(target_position - self.position)
        reward = (target_position * market_return) - (self.trading_cost * turnover)
        reward *= self.reward_scale

        self.position = target_position
        self.prev_reward = reward
        self.equity *= 1.0 + reward
        self.current_step += 1

        done = self.current_step >= self.last_step
        state = self._state()
        info = {
            "market_return": market_return,
            "position": self.position,
            "turnover": turnover,
            "equity": self.equity,
            "step": self.current_step,
            "task": "trading",
        }
        return state, float(reward), done, info

    def _state(self) -> TimeSeriesState:
        window = self.prices[self.current_step - self.window_size + 1 : self.current_step + 1]
        return TimeSeriesState(
            window=window,
            agent_features=np.asarray([self.position], dtype=float),
            step=self.current_step,
            context={
                "task": "trading",
                "equity": self.equity,
                "global_step": self.index_offset + self.current_step,
            },
        )
