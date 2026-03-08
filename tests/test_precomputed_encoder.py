from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsrl_lite.encoders.precomputed import PrecomputedEmbeddingEncoder
from tsrl_lite.state import TimeSeriesState


class PrecomputedEncoderTest(unittest.TestCase):
    def test_precomputed_encoder_reads_embedding_row(self) -> None:
        embeddings = np.arange(60, dtype=float).reshape(10, 6)
        state = TimeSeriesState(
            window=np.linspace(100.0, 104.0, num=8),
            agent_features=np.asarray([1.0]),
            step=4,
            context={"global_step": 4},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "embeddings.npy"
            np.save(path, embeddings)
            encoder = PrecomputedEmbeddingEncoder(
                window_size=8,
                agent_feature_dim=1,
                embedding_path=str(path),
                normalize_embeddings=False,
            )
            encoded = encoder.encode(state)

        self.assertEqual(encoded.shape, (7,))
        self.assertTrue(np.allclose(encoded[:-1], embeddings[4]))
        self.assertEqual(encoded[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
