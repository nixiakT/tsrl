from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None


if torch is not None:

    class TorchGRUActorCriticNetwork(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            action_dim: int,
            num_layers: int = 1,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_size = hidden_size
            self.action_dim = action_dim
            self.num_layers = num_layers
            self.dropout = dropout

            gru_dropout = dropout if num_layers > 1 else 0.0
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=gru_dropout,
            )
            self.policy_head = nn.Linear(hidden_size, action_dim)
            self.value_head = nn.Linear(hidden_size, 1)

        def forward(self, sequence_batch):
            outputs, _ = self.gru(sequence_batch)
            features = outputs[:, -1, :]
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
