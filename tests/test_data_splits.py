from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.config import DataConfig
from tsrl_lite.data.sources import (
    generate_walk_forward_splits,
    load_price_dataset,
    resolve_price_splits,
    split_train_val_eval_prices,
)


class DataSplitTest(unittest.TestCase):
    def test_split_train_val_eval_prices_returns_three_nonempty_splits(self) -> None:
        prices = np.linspace(100.0, 200.0, num=500)
        train_prices, val_prices, eval_prices = split_train_val_eval_prices(
            prices=prices,
            train_ratio=0.7,
            val_ratio=0.15,
            window_size=16,
            min_future_steps=4,
        )

        self.assertGreater(len(train_prices), 0)
        self.assertGreater(len(val_prices), 16)
        self.assertGreater(len(eval_prices), 16)
        self.assertLess(len(train_prices), len(prices))
        self.assertLess(len(val_prices), len(prices))
        self.assertLess(len(eval_prices), len(prices))

    def test_split_with_zero_val_ratio_keeps_eval_available(self) -> None:
        prices = np.linspace(100.0, 200.0, num=300)
        train_prices, val_prices, eval_prices = split_train_val_eval_prices(
            prices=prices,
            train_ratio=0.8,
            val_ratio=0.0,
            window_size=16,
            min_future_steps=1,
        )

        self.assertGreater(len(train_prices), 0)
        self.assertEqual(len(val_prices), len(eval_prices))

    def test_generate_walk_forward_splits_produces_ordered_folds(self) -> None:
        prices = np.linspace(100.0, 200.0, num=1200)
        folds = generate_walk_forward_splits(
            prices=prices,
            n_folds=3,
            train_ratio_start=0.45,
            window_size=16,
            min_future_steps=2,
        )

        self.assertEqual(len(folds), 3)
        self.assertLess(len(folds[0]["train_prices"]), len(prices))
        self.assertLess(int(folds[0]["val_offset"]), int(folds[0]["eval_offset"]))
        self.assertLess(int(folds[0]["eval_offset"]), int(folds[1]["eval_offset"]))
        self.assertGreater(len(folds[0]["val_prices"]), 16)
        self.assertGreater(len(folds[0]["eval_prices"]), 16)

    def test_load_price_dataset_sorts_and_filters_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "prices.csv"
            start_date = datetime(2024, 1, 1)
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "close"])
                writer.writeheader()
                for day in range(119, -1, -1):
                    current = start_date + timedelta(days=day)
                    writer.writerow(
                        {
                            "timestamp": current.isoformat(),
                            "close": f"{100.0 + day:.2f}",
                        }
                    )

            dataset = load_price_dataset(
                DataConfig(
                    source="csv",
                    csv_path=str(csv_path),
                    price_column="close",
                    timestamp_column="timestamp",
                    start_time="2024-01-20",
                    end_time="2024-04-10",
                ),
                seed=3,
            )

            self.assertIsNotNone(dataset.timestamps)
            self.assertEqual(dataset.timestamps[0].isoformat(), "2024-01-20T00:00:00")
            self.assertEqual(dataset.timestamps[-1].isoformat(), "2024-04-10T00:00:00")
            self.assertAlmostEqual(float(dataset.prices[0]), 119.0)
            self.assertAlmostEqual(float(dataset.prices[-1]), 200.0)

    def test_load_price_dataset_supports_multi_asset_csv_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "prices.csv"
            start_date = datetime(2024, 1, 1)
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "asset_a", "asset_b"])
                writer.writeheader()
                for day in range(96):
                    current = start_date + timedelta(days=day)
                    writer.writerow(
                        {
                            "timestamp": current.isoformat(),
                            "asset_a": f"{100.0 + (0.5 * day):.4f}",
                            "asset_b": f"{80.0 + (0.25 * day):.4f}",
                        }
                    )

            dataset = load_price_dataset(
                DataConfig(
                    source="csv",
                    csv_path=str(csv_path),
                    price_columns=["asset_a", "asset_b"],
                    timestamp_column="timestamp",
                ),
                seed=9,
            )

            self.assertEqual(dataset.prices.shape, (96, 2))
            self.assertEqual(dataset.feature_names, ("asset_a", "asset_b"))
            self.assertAlmostEqual(float(dataset.prices[0, 0]), 100.0)
            self.assertAlmostEqual(float(dataset.prices[-1, 1]), 103.75)

    def test_resolve_price_splits_supports_time_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "prices.csv"
            start_date = datetime(2024, 1, 1)
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "close"])
                writer.writeheader()
                for day in range(160):
                    current = start_date + timedelta(days=day)
                    writer.writerow(
                        {
                            "timestamp": current.isoformat(),
                            "close": f"{100.0 + day:.4f}",
                        }
                    )

            splits = resolve_price_splits(
                config=DataConfig(
                    source="csv",
                    csv_path=str(csv_path),
                    price_column="close",
                    timestamp_column="timestamp",
                    train_end_time="2024-03-15T00:00:00",
                    val_end_time="2024-04-20T00:00:00",
                    window_size=16,
                    val_ratio=0.1,
                ),
                seed=5,
                min_future_steps=1,
            )

            self.assertEqual(splits.split_mode, "time")
            self.assertGreater(len(splits.train_prices), 16)
            self.assertGreater(len(splits.val_prices), 16)
            self.assertGreater(len(splits.eval_prices), 16)
            self.assertEqual(splits.train_offset, 0)
            self.assertLess(splits.val_offset, splits.eval_offset)
            self.assertEqual(splits.train_end_time, "2024-03-15T00:00:00")
            self.assertEqual(splits.val_end_time, "2024-04-20T00:00:00")

    def test_load_price_dataset_rejects_inverted_time_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "prices.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "close"])
                writer.writeheader()
                for day in range(80):
                    current = datetime(2024, 1, 1) + timedelta(days=day)
                    writer.writerow({"timestamp": current.isoformat(), "close": f"{100.0 + day:.2f}"})

            with self.assertRaises(ValueError):
                load_price_dataset(
                    DataConfig(
                        source="csv",
                        csv_path=str(csv_path),
                        price_column="close",
                        timestamp_column="timestamp",
                        start_time="2024-03-01T00:00:00",
                        end_time="2024-02-01T00:00:00",
                    ),
                    seed=11,
                )


if __name__ == "__main__":
    unittest.main()
