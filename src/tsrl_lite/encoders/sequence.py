from __future__ import annotations

import numpy as np

from tsrl_lite.encoders.base import BaseStateEncoder
from tsrl_lite.registry import register_encoder
from tsrl_lite.state import TimeSeriesState


def _ensure_window_2d(state: TimeSeriesState, window_feature_dim: int) -> np.ndarray:
    window = np.asarray(state.window, dtype=float)
    if window.ndim == 1:
        window = window.reshape(-1, 1)
    if window.ndim != 2:
        raise ValueError("sequence encoder expects a 1D or 2D window")
    if window.shape[1] != window_feature_dim:
        raise ValueError(
            f"expected window feature dim {window_feature_dim}, got {window.shape[1]}"
        )
    return window


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=0, keepdims=True)
    stds = np.std(values, axis=0, keepdims=True) + 1e-6
    return (values - means) / stds


@register_encoder("sequence-window-v0")
class SequenceWindowEncoder(BaseStateEncoder):
    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self.window_size, (2 * self.window_feature_dim) + self.agent_feature_dim)

    @property
    def observation_dim(self) -> int:
        shape = self.observation_shape
        return int(shape[0] * shape[1])

    def encode(self, state: TimeSeriesState) -> np.ndarray:
        window = _ensure_window_2d(state, window_feature_dim=self.window_feature_dim)
        log_prices = np.log(window)
        normalized_prices = _normalize_columns(log_prices)
        log_returns = np.vstack([np.zeros((1, self.window_feature_dim), dtype=float), np.diff(log_prices, axis=0)])
        normalized_returns = _normalize_columns(log_returns)
        repeated_agent = np.tile(state.agent_features.reshape(1, -1), (self.window_size, 1))
        return np.concatenate(
            [
                normalized_prices,
                normalized_returns,
                repeated_agent,
            ],
            axis=1,
        ).astype(float)
