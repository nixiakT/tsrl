from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.encoders.handcrafted import (
    MultiAssetContextEncoder,
    MultiScaleContextEncoder,
    PriceContextEncoder,
    ReturnsContextEncoder,
)
from tsrl_lite.encoders.sequence import SequenceWindowEncoder
from tsrl_lite.state import TimeSeriesState


class EncoderTest(unittest.TestCase):
    def test_encoder_dimensions_match_output(self) -> None:
        state = TimeSeriesState(
            window=np.linspace(100.0, 110.0, num=16),
            agent_features=np.asarray([1.0]),
            step=15,
        )

        encoders = [
            ReturnsContextEncoder(window_size=16, agent_feature_dim=1),
            PriceContextEncoder(window_size=16, agent_feature_dim=1),
            MultiScaleContextEncoder(window_size=16, agent_feature_dim=1, short_window=3, long_window=8),
        ]

        for encoder in encoders:
            encoded = encoder.encode(state)
            self.assertEqual(encoded.shape, (encoder.observation_dim,))

    def test_sequence_encoder_emits_2d_sequence(self) -> None:
        state = TimeSeriesState(
            window=np.linspace(100.0, 110.0, num=16),
            agent_features=np.asarray([1.0]),
            step=15,
        )
        encoder = SequenceWindowEncoder(window_size=16, agent_feature_dim=1)
        encoded = encoder.encode(state)
        self.assertEqual(encoded.shape, encoder.observation_shape)

    def test_multi_asset_encoders_support_2d_windows(self) -> None:
        state = TimeSeriesState(
            window=np.stack(
                [
                    np.linspace(100.0, 110.0, num=16),
                    np.linspace(95.0, 107.0, num=16),
                    np.linspace(102.0, 108.0, num=16),
                ],
                axis=1,
            ),
            agent_features=np.asarray([0.4, 0.4, 0.2]),
            step=15,
        )

        flat_encoder = MultiAssetContextEncoder(window_size=16, agent_feature_dim=3, window_feature_dim=3)
        flat_encoded = flat_encoder.encode(state)
        self.assertEqual(flat_encoded.shape, (flat_encoder.observation_dim,))

        sequence_encoder = SequenceWindowEncoder(window_size=16, agent_feature_dim=3, window_feature_dim=3)
        sequence_encoded = sequence_encoder.encode(state)
        self.assertEqual(sequence_encoded.shape, sequence_encoder.observation_shape)


if __name__ == "__main__":
    unittest.main()
