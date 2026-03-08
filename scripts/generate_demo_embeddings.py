from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsrl_lite.config import load_experiment_config
from tsrl_lite.data import load_price_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo time-series embeddings for tsrl-lite.")
    parser.add_argument("--config", required=True, help="experiment config used to load the price series")
    parser.add_argument("--output", required=True, help="target .npy file")
    parser.add_argument("--dim", type=int, default=16, help="embedding dimension")
    parser.add_argument("--seed", type=int, default=7, help="random seed for the projection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    prices = load_price_series(config.data, seed=config.seed)
    log_prices = np.log(prices)
    returns = np.concatenate([[0.0], np.diff(log_prices)])
    rolling_vol = np.sqrt(np.convolve(returns**2, np.ones(8, dtype=float) / 8.0, mode="same"))
    features = np.stack(
        [
            log_prices,
            returns,
            rolling_vol,
        ],
        axis=1,
    )
    rng = np.random.default_rng(args.seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(features.shape[1]), size=(features.shape[1], args.dim))
    embeddings = features @ projection
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings.astype(np.float32))
    print(f"saved {embeddings.shape[0]} embeddings with dim={embeddings.shape[1]} to {output_path}")


if __name__ == "__main__":
    main()
