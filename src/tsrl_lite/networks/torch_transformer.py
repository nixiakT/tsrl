from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None


if torch is not None:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512) -> None:
            super().__init__()
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
            )
            pe = torch.zeros(max_len, d_model, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :]


    class TorchTransformerActorCriticNetwork(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            action_dim: int,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
            max_len: int = 512,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_size = hidden_size
            self.action_dim = action_dim
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.dropout = dropout
            self.max_len = max_len

            self.input_proj = nn.Linear(input_dim, hidden_size)
            self.positional_encoding = PositionalEncoding(hidden_size, max_len=max_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(hidden_size)
            self.policy_head = nn.Linear(hidden_size, action_dim)
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, sequence_batch):
            x = self.input_proj(sequence_batch)
            x = self.positional_encoding(x)
            x = self.encoder(x)
            features = self.norm(x[:, -1, :])
            logits = self.policy_head(features)
            value = self.value_head(features).squeeze(-1)
            return logits, value

        def save(self, path: str | Path, optimizer=None) -> None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "state_dict": self.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_size": self.hidden_size,
                    "action_dim": self.action_dim,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
                    "max_len": self.max_len,
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
