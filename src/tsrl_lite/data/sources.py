from __future__ import annotations

import csv
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from tsrl_lite.config import DataConfig


@dataclass(slots=True)
class LoadedPriceSeries:
    prices: np.ndarray
    timestamps: tuple[datetime, ...] | None = None
    feature_names: tuple[str, ...] | None = None


@dataclass(slots=True)
class PriceSplits:
    full_prices: np.ndarray
    train_prices: np.ndarray
    val_prices: np.ndarray
    eval_prices: np.ndarray
    train_offset: int
    val_offset: int
    eval_offset: int
    split_mode: str
    data_start_time: str | None = None
    data_end_time: str | None = None
    train_end_time: str | None = None
    val_end_time: str | None = None


def generate_synthetic_prices(config: DataConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = config.synthetic.steps
    asset_count = max(1, int(config.synthetic.assets))
    t = np.arange(steps - 1, dtype=float)
    regime = np.sign(np.sin(t / 48.0)) + 0.5 * np.sign(np.sin(t / 13.0))
    regime[regime == 0.0] = 1.0
    drift = config.synthetic.drift * regime
    seasonality = config.synthetic.seasonality * np.sin(t / 6.0)

    if asset_count == 1:
        log_returns = np.empty(steps - 1, dtype=float)
        prev_return = 0.0
        for idx in range(steps - 1):
            innovation = rng.normal(0.0, config.synthetic.volatility)
            prev_return = (0.55 * prev_return) + drift[idx] + seasonality[idx] + innovation
            log_returns[idx] = prev_return
        prices = np.empty(steps, dtype=float)
        prices[0] = 100.0
        prices[1:] = prices[0] * np.exp(np.cumsum(log_returns))
        return prices

    asset_scales = np.linspace(0.85, 1.15, num=asset_count, dtype=float)
    volatility_scales = np.linspace(0.9, 1.1, num=asset_count, dtype=float)
    common_scale = np.sqrt(float(np.clip(config.synthetic.correlation, 0.0, 0.999)))
    idio_scale = np.sqrt(max(1e-8, 1.0 - (common_scale * common_scale)))
    log_returns = np.empty((steps - 1, asset_count), dtype=float)
    prev_returns = np.zeros(asset_count, dtype=float)
    for idx in range(steps - 1):
        common_noise = rng.normal(0.0, config.synthetic.volatility * common_scale)
        idio_noise = rng.normal(
            0.0,
            config.synthetic.volatility * idio_scale,
            size=asset_count,
        )
        innovations = (common_noise + idio_noise) * volatility_scales
        mean_component = (drift[idx] + seasonality[idx]) * asset_scales
        prev_returns = (0.55 * prev_returns) + mean_component + innovations
        log_returns[idx] = prev_returns

    prices = np.empty((steps, asset_count), dtype=float)
    prices[0] = 100.0 + np.arange(asset_count, dtype=float)
    prices[1:] = prices[0] * np.exp(np.cumsum(log_returns, axis=0))
    return prices


def _parse_timestamp(raw_value: str, timestamp_format: str | None) -> datetime:
    normalized = raw_value.strip()
    if timestamp_format:
        return datetime.strptime(normalized, timestamp_format)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "failed to parse timestamp; provide data.timestamp_format or use ISO-8601 strings"
        ) from exc


def _serialize_timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _resolve_price_columns(
    price_column: str,
    price_columns: list[str] | None,
) -> list[str]:
    columns = [column for column in price_columns or [] if column]
    if columns:
        return columns
    return [price_column]


def load_prices_from_csv(
    csv_path: str | Path,
    price_column: str,
    price_columns: list[str] | None = None,
    timestamp_column: str | None = None,
    timestamp_format: str | None = None,
    sort_ascending: bool = True,
    start_time: str | None = None,
    end_time: str | None = None,
) -> LoadedPriceSeries:
    path = Path(csv_path)
    resolved_columns = _resolve_price_columns(price_column, price_columns)
    rows: list[tuple[datetime | None, list[float]]] = []
    start_dt = _parse_timestamp(start_time, timestamp_format) if start_time is not None else None
    end_dt = _parse_timestamp(end_time, timestamp_format) if end_time is not None else None
    if (start_dt is not None or end_dt is not None) and timestamp_column is None:
        raise ValueError("start_time/end_time require data.timestamp_column for csv data")
    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        raise ValueError("start_time must be earlier than or equal to end_time")

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for column in resolved_columns:
                if column not in row:
                    raise KeyError(f"missing column '{column}' in {path}")
            if timestamp_column is not None and timestamp_column not in row:
                raise KeyError(f"missing column '{timestamp_column}' in {path}")
            price_values: list[float] = []
            skip_row = False
            for column in resolved_columns:
                raw_value = row[column]
                if raw_value is None or raw_value == "":
                    skip_row = True
                    break
                price_values.append(float(raw_value))
            if skip_row:
                continue
            timestamp_value: datetime | None = None
            if timestamp_column is not None:
                raw_timestamp = row[timestamp_column]
                if raw_timestamp is None or raw_timestamp == "":
                    continue
                timestamp_value = _parse_timestamp(raw_timestamp, timestamp_format)
                if start_dt is not None and timestamp_value < start_dt:
                    continue
                if end_dt is not None and timestamp_value > end_dt:
                    continue
            rows.append((timestamp_value, price_values))

    if timestamp_column is not None:
        rows.sort(key=lambda item: item[0], reverse=not sort_ascending)

    values = np.asarray([value for _, value in rows], dtype=float)
    if len(values) < 64:
        raise ValueError("expected at least 64 price points for training")
    timestamps = None
    if timestamp_column is not None:
        timestamps = tuple(timestamp for timestamp, _ in rows if timestamp is not None)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    return LoadedPriceSeries(
        prices=values,
        timestamps=timestamps,
        feature_names=tuple(resolved_columns),
    )


def load_price_dataset(config: DataConfig, seed: int) -> LoadedPriceSeries:
    if config.source == "synthetic":
        return LoadedPriceSeries(prices=generate_synthetic_prices(config, seed))
    if config.source == "csv":
        if not config.csv_path:
            raise ValueError("csv_path is required when data.source='csv'")
        return load_prices_from_csv(
            csv_path=config.csv_path,
            price_column=config.price_column,
            price_columns=config.price_columns,
            timestamp_column=config.timestamp_column,
            timestamp_format=config.timestamp_format,
            sort_ascending=config.sort_ascending,
            start_time=config.start_time,
            end_time=config.end_time,
        )
    raise ValueError(f"unsupported data source: {config.source}")


def load_price_series(config: DataConfig, seed: int) -> np.ndarray:
    return load_price_dataset(config, seed).prices


def split_train_eval_prices(
    prices: np.ndarray,
    train_ratio: float,
    window_size: int,
    min_future_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    split_index = int(len(prices) * train_ratio)
    margin = window_size + max(1, min_future_steps) + 1
    split_index = max(split_index, margin)
    split_index = min(split_index, len(prices) - margin)
    if split_index <= margin - 1:
        raise ValueError("not enough data after split; increase the series length")
    train_prices = prices[:split_index]
    eval_prices = prices[split_index - window_size :]
    return train_prices, eval_prices


def split_train_val_eval_prices(
    prices: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    window_size: int,
    min_future_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= train_ratio < 1.0:
        raise ValueError("train_ratio must be in [0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    split_index = int(len(prices) * train_ratio)
    margin = window_size + max(1, min_future_steps) + 1
    split_index = max(split_index, margin)
    if val_ratio == 0.0:
        split_index = min(split_index, len(prices) - margin)
        if split_index <= margin - 1:
            raise ValueError("not enough data after split; increase the series length")
        train_prices = prices[:split_index]
        eval_prices = prices[split_index - window_size :]
        return train_prices, eval_prices, eval_prices

    eval_start = int(len(prices) * (train_ratio + val_ratio))
    eval_start = max(eval_start, split_index + margin)
    eval_start = min(eval_start, len(prices) - margin)
    split_index = min(split_index, eval_start - margin)

    if split_index <= margin - 1 or eval_start <= split_index + margin - 1:
        raise ValueError("not enough data after split; increase the series length")

    train_prices = prices[:split_index]
    val_prices = prices[split_index - window_size : eval_start]
    eval_prices = prices[eval_start - window_size :]
    return train_prices, val_prices, eval_prices


def split_train_val_eval_prices_by_time(
    prices: np.ndarray,
    timestamps: tuple[datetime, ...],
    train_end_time: str,
    val_end_time: str | None,
    timestamp_format: str | None,
    window_size: int,
    min_future_steps: int = 1,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if len(prices) != len(timestamps):
        raise ValueError("prices and timestamps must have the same length")
    margin = window_size + max(1, min_future_steps) + 1
    train_end_dt = _parse_timestamp(train_end_time, timestamp_format)
    train_end_index = bisect_right(timestamps, train_end_dt)

    if train_end_index < margin:
        raise ValueError("train_end_time leaves too few points for training")

    if val_ratio == 0.0:
        if len(prices) - train_end_index < margin:
            raise ValueError("train_end_time leaves too few points for evaluation")
        eval_offset = train_end_index - window_size
        train_prices = prices[:train_end_index]
        eval_prices = prices[eval_offset:]
        return train_prices, eval_prices, eval_prices, eval_offset, eval_offset

    if val_end_time is None:
        raise ValueError("val_end_time is required when using time-based train/val/eval splits")

    val_end_dt = _parse_timestamp(val_end_time, timestamp_format)
    val_end_index = bisect_right(timestamps, val_end_dt)
    if val_end_index <= train_end_index:
        raise ValueError("val_end_time must be later than train_end_time")
    if val_end_index - train_end_index < margin:
        raise ValueError("time-based validation slice is too short")
    if len(prices) - val_end_index < margin:
        raise ValueError("time-based evaluation slice is too short")

    val_offset = train_end_index - window_size
    eval_offset = val_end_index - window_size
    train_prices = prices[:train_end_index]
    val_prices = prices[val_offset:val_end_index]
    eval_prices = prices[eval_offset:]
    return train_prices, val_prices, eval_prices, val_offset, eval_offset


def resolve_price_splits(
    config: DataConfig,
    seed: int,
    min_future_steps: int = 1,
) -> PriceSplits:
    dataset = load_price_dataset(config, seed)
    data_start_time = _serialize_timestamp(dataset.timestamps[0]) if dataset.timestamps else None
    data_end_time = _serialize_timestamp(dataset.timestamps[-1]) if dataset.timestamps else None

    if config.train_end_time is not None or config.val_end_time is not None:
        if dataset.timestamps is None:
            raise ValueError("time-based splits require csv data with data.timestamp_column")
        if not config.sort_ascending:
            raise ValueError("time-based splits require data.sort_ascending=true")
        if config.train_end_time is None:
            raise ValueError("train_end_time is required for time-based splits")
        train_prices, val_prices, eval_prices, val_offset, eval_offset = split_train_val_eval_prices_by_time(
            prices=dataset.prices,
            timestamps=dataset.timestamps,
            train_end_time=config.train_end_time,
            val_end_time=config.val_end_time,
            timestamp_format=config.timestamp_format,
            window_size=config.window_size,
            min_future_steps=min_future_steps,
            val_ratio=config.val_ratio,
        )
        return PriceSplits(
            full_prices=dataset.prices,
            train_prices=train_prices,
            val_prices=val_prices,
            eval_prices=eval_prices,
            train_offset=0,
            val_offset=val_offset,
            eval_offset=eval_offset,
            split_mode="time",
            data_start_time=data_start_time,
            data_end_time=data_end_time,
            train_end_time=config.train_end_time,
            val_end_time=config.val_end_time,
        )

    train_prices, val_prices, eval_prices = split_train_val_eval_prices(
        prices=dataset.prices,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        window_size=config.window_size,
        min_future_steps=min_future_steps,
    )
    val_offset = len(train_prices) - config.window_size
    eval_offset = len(dataset.prices) - len(eval_prices)
    return PriceSplits(
        full_prices=dataset.prices,
        train_prices=train_prices,
        val_prices=val_prices,
        eval_prices=eval_prices,
        train_offset=0,
        val_offset=val_offset,
        eval_offset=eval_offset,
        split_mode="ratio",
        data_start_time=data_start_time,
        data_end_time=data_end_time,
    )


def generate_walk_forward_splits(
    prices: np.ndarray,
    n_folds: int,
    train_ratio_start: float,
    window_size: int,
    min_future_steps: int = 1,
) -> list[dict[str, object]]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not 0.0 < train_ratio_start < 1.0:
        raise ValueError("train_ratio_start must be in (0, 1)")

    margin = window_size + max(1, min_future_steps) + 1
    total = len(prices)
    initial_train_end = max(int(total * train_ratio_start), margin)
    remaining = total - initial_train_end
    segment_length = remaining // (2 * n_folds)
    if segment_length < margin:
        raise ValueError("not enough points to build walk-forward folds")

    folds: list[dict[str, object]] = []
    for fold_index in range(n_folds):
        train_end = initial_train_end + (2 * fold_index * segment_length)
        val_start = train_end
        val_end = val_start + segment_length
        eval_start = val_end
        eval_end = total if fold_index == n_folds - 1 else eval_start + segment_length

        if eval_end - eval_start < margin:
            break

        folds.append(
            {
                "fold": fold_index,
                "train_prices": prices[:train_end],
                "val_prices": prices[val_start - window_size : val_end],
                "eval_prices": prices[eval_start - window_size : eval_end],
                "train_offset": 0,
                "val_offset": val_start - window_size,
                "eval_offset": eval_start - window_size,
            }
        )

    if not folds:
        raise ValueError("unable to produce any valid walk-forward folds")
    return folds
