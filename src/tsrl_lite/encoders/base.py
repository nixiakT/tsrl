from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tsrl_lite.state import TimeSeriesState


class BaseStateEncoder(ABC):
    def __init__(
        self,
        window_size: int,
        agent_feature_dim: int,
        window_feature_dim: int = 1,
        **_: object,
    ) -> None:
        self.window_size = int(window_size)
        self.agent_feature_dim = int(agent_feature_dim)
        self.window_feature_dim = max(1, int(window_feature_dim))

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self.observation_dim,)

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode(self, state: TimeSeriesState) -> np.ndarray:
        raise NotImplementedError
