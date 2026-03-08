from __future__ import annotations

import json
import csv
from pathlib import Path

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def dump_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def dump_records_csv(path: str | Path, records: list[dict[str, object]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    if not records:
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([])
        return

    fieldnames = sorted({key for record in records for key in record})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def clip_gradient(*grads: np.ndarray, max_norm: float) -> tuple[np.ndarray, ...]:
    total_norm = np.sqrt(sum(float(np.sum(grad * grad)) for grad in grads))
    if total_norm <= max_norm or total_norm == 0.0:
        return grads
    scale = max_norm / (total_norm + 1e-8)
    return tuple(grad * scale for grad in grads)


def collect_numeric_info(info: dict[str, object], bucket: dict[str, list[float]]) -> None:
    for key, value in info.items():
        if key == "step" or isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            bucket.setdefault(key, []).append(float(value))


def compute_trading_episode_metrics(
    rewards: np.ndarray,
    info_metrics: dict[str, list[float]],
) -> dict[str, float]:
    if rewards.size == 0:
        return {}

    mean_reward = float(np.mean(rewards))
    reward_std = float(np.std(rewards))
    downside = rewards[rewards < 0.0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if downside.size > 0 else 0.0
    metrics = {
        "win_rate": float(np.mean(rewards > 0.0)),
        "reward_volatility": reward_std,
        "sharpe_ratio": (
            float((mean_reward / reward_std) * np.sqrt(float(rewards.size))) if reward_std > 1e-8 else 0.0
        ),
        "sortino_ratio": (
            float((mean_reward / downside_deviation) * np.sqrt(float(rewards.size)))
            if downside_deviation > 1e-8
            else 0.0
        ),
    }

    equity_values = info_metrics.get("equity")
    if equity_values:
        equity_curve = np.asarray(equity_values, dtype=float)
        equity_path = np.concatenate([np.asarray([1.0], dtype=float), equity_curve])
        running_peaks = np.maximum.accumulate(equity_path)
        drawdowns = 1.0 - (equity_path / np.maximum(running_peaks, 1e-8))
        final_equity = float(equity_path[-1])
        terminal_return = final_equity - 1.0
        max_drawdown = float(np.max(drawdowns))
        metrics.update(
            {
                "final_equity": final_equity,
                "terminal_return": terminal_return,
                "max_drawdown": max_drawdown,
                "calmar_ratio": (float(terminal_return / max_drawdown) if max_drawdown > 1e-8 else 0.0),
            }
        )

    return metrics
