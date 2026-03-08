from __future__ import annotations

import numpy as np

from tsrl_lite.encoders.base import BaseStateEncoder
from tsrl_lite.registry import register_encoder
from tsrl_lite.state import TimeSeriesState


def _normalize(values: np.ndarray) -> np.ndarray:
    return (values - np.mean(values)) / (np.std(values) + 1e-6)


def _normalize_columns(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=0, keepdims=True)
    stds = np.std(values, axis=0, keepdims=True) + 1e-6
    return (values - means) / stds


def _ensure_univariate_window(state: TimeSeriesState) -> np.ndarray:
    if state.window.ndim != 1:
        raise ValueError("encoder expects a single-series window; use a multi-asset encoder instead")
    return state.window


def _ensure_multivariate_window(state: TimeSeriesState, window_feature_dim: int) -> np.ndarray:
    window = np.asarray(state.window, dtype=float)
    if window.ndim == 1:
        window = window.reshape(-1, 1)
    if window.ndim != 2:
        raise ValueError("multi-asset encoder expects a 2D [time, features] window")
    if window.shape[1] != window_feature_dim:
        raise ValueError(
            f"expected window feature dim {window_feature_dim}, got {window.shape[1]}"
        )
    return window


@register_encoder("returns-context-v0")
class ReturnsContextEncoder(BaseStateEncoder):
    @property
    def observation_dim(self) -> int:
        return (self.window_size - 1) + self.agent_feature_dim

    def encode(self, state: TimeSeriesState) -> np.ndarray:
        log_returns = np.diff(np.log(_ensure_univariate_window(state)))
        return np.concatenate([_normalize(log_returns), state.agent_features])


@register_encoder("price-context-v0")
class PriceContextEncoder(BaseStateEncoder):
    @property
    def observation_dim(self) -> int:
        return self.window_size + self.agent_feature_dim

    def encode(self, state: TimeSeriesState) -> np.ndarray:
        log_prices = np.log(_ensure_univariate_window(state))
        return np.concatenate([_normalize(log_prices), state.agent_features])


@register_encoder("multi-scale-context-v0")
class MultiScaleContextEncoder(BaseStateEncoder):
    def __init__(
        self,
        window_size: int,
        agent_feature_dim: int,
        short_window: int = 4,
        long_window: int = 12,
        **kwargs: object,
    ) -> None:
        super().__init__(window_size=window_size, agent_feature_dim=agent_feature_dim, **kwargs)
        self.short_window = max(2, min(int(short_window), self.window_size - 1))
        self.long_window = max(self.short_window + 1, min(int(long_window), self.window_size - 1))

    @property
    def observation_dim(self) -> int:
        return self.window_size + (self.window_size - 1) + 2 + self.agent_feature_dim

    def encode(self, state: TimeSeriesState) -> np.ndarray:
        log_prices = np.log(_ensure_univariate_window(state))
        log_returns = np.diff(log_prices)
        normalized_prices = _normalize(log_prices)
        normalized_returns = _normalize(log_returns)
        short_mean = float(np.mean(log_returns[-self.short_window :]))
        long_mean = float(np.mean(log_returns[-self.long_window :]))
        realized_vol = float(np.std(log_returns[-self.long_window :]))
        summary = np.asarray([short_mean - long_mean, realized_vol], dtype=float)
        return np.concatenate(
            [
                normalized_prices,
                normalized_returns,
                summary,
                state.agent_features,
            ]
        )


@register_encoder("multi-asset-context-v0")
class MultiAssetContextEncoder(BaseStateEncoder):
    @property
    def observation_dim(self) -> int:
        return (
            (self.window_size * self.window_feature_dim)
            + ((self.window_size - 1) * self.window_feature_dim)
            + 4
            + self.agent_feature_dim
        )

    def encode(self, state: TimeSeriesState) -> np.ndarray:
        window = _ensure_multivariate_window(state, window_feature_dim=self.window_feature_dim)
        log_prices = np.log(window)
        log_returns = np.diff(log_prices, axis=0)
        normalized_prices = _normalize_columns(log_prices).reshape(-1)
        normalized_returns = _normalize_columns(log_returns).reshape(-1)
        latest_returns = log_returns[-1]
        summary = np.asarray(
            [
                float(np.mean(latest_returns)),
                float(np.std(latest_returns)),
                float(np.mean(log_returns)),
                float(np.std(log_returns)),
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                normalized_prices,
                normalized_returns,
                summary,
                state.agent_features,
            ]
        )
