from __future__ import annotations

from pathlib import Path

import numpy as np

from tsrl_lite.encoders.base import BaseStateEncoder
from tsrl_lite.registry import register_encoder
from tsrl_lite.state import TimeSeriesState


@register_encoder("precomputed-context-v0")
class PrecomputedEmbeddingEncoder(BaseStateEncoder):
    def __init__(
        self,
        window_size: int,
        agent_feature_dim: int,
        embedding_path: str,
        normalize_embeddings: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__(window_size=window_size, agent_feature_dim=agent_feature_dim, **kwargs)
        embeddings = np.load(Path(embedding_path))
        if embeddings.ndim != 2:
            raise ValueError("embedding_path must point to a 2D .npy embedding matrix")
        self.embeddings = embeddings.astype(float)
        self.normalize_embeddings = normalize_embeddings

    @property
    def observation_dim(self) -> int:
        return int(self.embeddings.shape[1] + self.agent_feature_dim)

    def encode(self, state: TimeSeriesState) -> np.ndarray:
        step_index = int(state.context.get("global_step", state.step))
        if step_index >= len(self.embeddings):
            raise IndexError("state.step exceeds the loaded embedding matrix")
        embedding = self.embeddings[step_index]
        if self.normalize_embeddings:
            embedding = (embedding - np.mean(embedding)) / (np.std(embedding) + 1e-6)
        return np.concatenate([embedding, state.agent_features])
