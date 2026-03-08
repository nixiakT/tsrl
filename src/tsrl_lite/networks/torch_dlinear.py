from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    functional = None


if torch is not None:

    class MovingAverage(nn.Module):
        def __init__(self, kernel_size: int) -> None:
            super().__init__()
            if kernel_size < 1:
                raise ValueError("moving average kernel_size must be >= 1")
            self.kernel_size = int(kernel_size)

        def forward(self, sequence_batch):
            left_padding = (self.kernel_size - 1) // 2
            right_padding = self.kernel_size // 2
            if left_padding > 0:
                left_context = sequence_batch[:, :1, :].repeat(1, left_padding, 1)
            else:
                left_context = sequence_batch[:, :0, :]
            if right_padding > 0:
                right_context = sequence_batch[:, -1:, :].repeat(1, right_padding, 1)
            else:
                right_context = sequence_batch[:, :0, :]
            padded = torch.cat([left_context, sequence_batch, right_context], dim=1)
            trend = functional.avg_pool1d(
                padded.transpose(1, 2),
                kernel_size=self.kernel_size,
                stride=1,
            ).transpose(1, 2)
            return trend


    class SeriesDecomposition(nn.Module):
        def __init__(self, kernel_size: int) -> None:
            super().__init__()
            self.moving_average = MovingAverage(kernel_size)

        def forward(self, sequence_batch):
            trend = self.moving_average(sequence_batch)
            seasonal = sequence_batch - trend
            return seasonal, trend


    class TorchDLinearActorCriticNetwork(nn.Module):
        def __init__(
            self,
            input_dim: int,
            sequence_length: int,
            hidden_size: int,
            action_dim: int,
            moving_avg: int = 25,
            individual: bool = False,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.sequence_length = sequence_length
            self.hidden_size = hidden_size
            self.action_dim = action_dim
            self.moving_avg = moving_avg
            self.individual = bool(individual)
            self.dropout = dropout

            self.decomposition = SeriesDecomposition(moving_avg)
            if self.individual:
                self.seasonal_linears = nn.ModuleList(
                    nn.Linear(sequence_length, hidden_size) for _ in range(input_dim)
                )
                self.trend_linears = nn.ModuleList(
                    nn.Linear(sequence_length, hidden_size) for _ in range(input_dim)
                )
            else:
                self.seasonal_linear = nn.Linear(sequence_length, hidden_size)
                self.trend_linear = nn.Linear(sequence_length, hidden_size)
            self.feature_projection = nn.Linear(input_dim * hidden_size, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)
            self.dropout_layer = nn.Dropout(dropout)
            self.policy_head = nn.Linear(hidden_size, action_dim)
            self.value_head = nn.Linear(hidden_size, 1)

        def _project_components(self, seasonal, trend):
            seasonal = seasonal.transpose(1, 2)
            trend = trend.transpose(1, 2)
            if self.individual:
                seasonal_features = torch.stack(
                    [layer(seasonal[:, feature_index, :]) for feature_index, layer in enumerate(self.seasonal_linears)],
                    dim=1,
                )
                trend_features = torch.stack(
                    [layer(trend[:, feature_index, :]) for feature_index, layer in enumerate(self.trend_linears)],
                    dim=1,
                )
            else:
                seasonal_features = self.seasonal_linear(seasonal)
                trend_features = self.trend_linear(trend)
            return seasonal_features + trend_features

        def forward(self, sequence_batch):
            seasonal, trend = self.decomposition(sequence_batch)
            temporal_features = self._project_components(seasonal, trend)
            fused = temporal_features.reshape(sequence_batch.size(0), -1)
            fused = self.feature_projection(fused)
            fused = self.dropout_layer(functional.gelu(self.norm(fused)))
            logits = self.policy_head(fused)
            value = self.value_head(fused).squeeze(-1)
            return logits, value

        def save(self, path: str | Path, optimizer=None) -> None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "sequence_length": self.sequence_length,
                    "hidden_size": self.hidden_size,
                    "action_dim": self.action_dim,
                    "moving_avg": self.moving_avg,
                    "individual": self.individual,
                    "dropout": self.dropout,
                },
            }
            if optimizer is not None:
                payload["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(payload, target)

        @classmethod
        def load(cls, path: str | Path, map_location=None):
            payload = torch.load(Path(path), map_location=map_location)
            network = cls(**payload["config"])
            network.load_state_dict(payload["state_dict"])
            return network, payload.get("optimizer_state_dict")
