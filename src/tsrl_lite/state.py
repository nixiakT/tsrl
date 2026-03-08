from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class TimeSeriesState:
    window: np.ndarray
    agent_features: np.ndarray
    step: int
    context: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.window = np.asarray(self.window, dtype=float)
        self.agent_features = np.asarray(self.agent_features, dtype=float)
