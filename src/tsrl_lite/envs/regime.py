from __future__ import annotations

import numpy as np

from tsrl_lite.envs.base import BaseTimeSeriesEnv
from tsrl_lite.registry import register_env
from tsrl_lite.state import TimeSeriesState


@register_env("regime-v0")
class RegimeClassificationEnv(BaseTimeSeriesEnv):
    def __init__(
        self,
        prices: np.ndarray,
        window_size: int,
        forecast_horizon: int = 8,
        regime_threshold: float = 0.002,
        reward_scale: float = 1.0,
        episode_horizon: int | None = None,
        random_reset: bool = False,
        seed: int = 7,
        index_offset: int = 0,
    ) -> None:
        if len(prices) <= window_size + forecast_horizon + 2:
            raise ValueError("prices must be longer than window_size + forecast_horizon + 2")

        self.prices = np.asarray(prices, dtype=float)
        self.window_size = int(window_size)
        self.forecast_horizon = int(forecast_horizon)
        self.regime_threshold = float(regime_threshold)
        self.reward_scale = float(reward_scale)
        self.episode_horizon = episode_horizon
        self.random_reset = random_reset
        self.rng = np.random.default_rng(seed)
        self.index_offset = int(index_offset)

        self.action_values = np.asarray([-1.0, 0.0, 1.0], dtype=float)
        self.action_dim = len(self.action_values)
        self.agent_feature_dim = 1
        self.window_feature_dim = 1

        self.start_step = self.window_size - 1
        self.current_step = self.start_step
        self.last_step = len(self.prices) - self.forecast_horizon - 1
        self.last_prediction = 0.0

    @classmethod
    def required_future_steps(cls, params: dict[str, object]) -> int:
        return int(params.get("forecast_horizon", 8))

    def reset(self, random_start: bool = False) -> TimeSeriesState:
        min_start = self.window_size - 1
        max_terminal = len(self.prices) - self.forecast_horizon - 1
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
        self.last_prediction = 0.0
        return self._state()

    def step(self, action: int) -> tuple[TimeSeriesState, float, bool, dict]:
        if action < 0 or action >= self.action_dim:
            raise IndexError(f"action {action} is out of range")

        current_price = self.prices[self.current_step]
        future_price = self.prices[self.current_step + self.forecast_horizon]
        future_return = (future_price - current_price) / current_price
        target_regime = self._label(future_return)
        predicted_regime = float(self.action_values[action])
        reward = (1.0 - abs(predicted_regime - target_regime)) * self.reward_scale
        self.last_prediction = predicted_regime
        self.current_step += 1

        done = self.current_step >= self.last_step
        state = self._state()
        info = {
            "future_return": future_return,
            "target_regime": target_regime,
            "predicted_regime": predicted_regime,
            "accuracy": float(predicted_regime == target_regime),
            "step": self.current_step,
            "task": "regime_classification",
        }
        return state, float(reward), done, info

    def _label(self, future_return: float) -> float:
        if future_return > self.regime_threshold:
            return 1.0
        if future_return < -self.regime_threshold:
            return -1.0
        return 0.0

    def _state(self) -> TimeSeriesState:
        window = self.prices[self.current_step - self.window_size + 1 : self.current_step + 1]
        return TimeSeriesState(
            window=window,
            agent_features=np.asarray([self.last_prediction], dtype=float),
            step=self.current_step,
            context={
                "task": "regime_classification",
                "global_step": self.index_offset + self.current_step,
            },
        )
