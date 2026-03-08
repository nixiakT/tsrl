from __future__ import annotations

from abc import ABC, abstractmethod

from tsrl_lite.state import TimeSeriesState


class BaseTimeSeriesEnv(ABC):
    window_size: int
    action_dim: int
    agent_feature_dim: int
    window_feature_dim: int

    @abstractmethod
    def reset(self, random_start: bool = False) -> TimeSeriesState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> tuple[TimeSeriesState, float, bool, dict]:
        raise NotImplementedError

    @classmethod
    def required_future_steps(cls, _: dict[str, object]) -> int:
        return 1
