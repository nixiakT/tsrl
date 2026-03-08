from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None


if torch is not None:

    class TorchPatchTSTActorCriticNetwork(nn.Module):
        def __init__(
            self,
            input_dim: int,
            sequence_length: int,
            hidden_size: int,
            action_dim: int,
            patch_len: int = 8,
            stride: int = 4,
            num_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.1,
            use_cls_token: bool = True,
        ) -> None:
            super().__init__()
            if patch_len < 1:
                raise ValueError("patch_len must be >= 1")
            if stride < 1:
                raise ValueError("stride must be >= 1")
            if patch_len > sequence_length:
                raise ValueError("patch_len must be <= sequence_length")

            num_patches = 1 + ((sequence_length - patch_len) // stride)
            if num_patches < 1:
                raise ValueError("patch configuration produced zero patches")

            self.input_dim = input_dim
            self.sequence_length = sequence_length
            self.hidden_size = hidden_size
            self.action_dim = action_dim
            self.patch_len = patch_len
            self.stride = stride
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.dropout = dropout
            self.use_cls_token = bool(use_cls_token)
            self.num_patches = num_patches

            self.patch_projection = nn.Linear(patch_len * input_dim, hidden_size)
            self.input_dropout = nn.Dropout(dropout)
            token_count = num_patches + (1 if self.use_cls_token else 0)
            self.position_embedding = nn.Parameter(torch.zeros(1, token_count, hidden_size))
            if self.use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            else:
                self.register_parameter("cls_token", None)
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

        def _patchify(self, sequence_batch):
            patches = sequence_batch.unfold(dimension=1, size=self.patch_len, step=self.stride)
            patches = patches.permute(0, 1, 3, 2).contiguous()
            return patches.view(sequence_batch.size(0), self.num_patches, self.patch_len * self.input_dim)

        def forward(self, sequence_batch):
            tokens = self.patch_projection(self._patchify(sequence_batch))
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(sequence_batch.size(0), -1, -1)
                tokens = torch.cat([cls_tokens, tokens], dim=1)
            tokens = self.input_dropout(tokens + self.position_embedding[:, : tokens.size(1), :])
            encoded = self.encoder(tokens)
            if self.use_cls_token:
                features = encoded[:, 0, :]
            else:
                features = encoded.mean(dim=1)
            features = self.norm(features)
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
                    "sequence_length": self.sequence_length,
                    "hidden_size": self.hidden_size,
                    "action_dim": self.action_dim,
                    "patch_len": self.patch_len,
                    "stride": self.stride,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
                    "use_cls_token": self.use_cls_token,
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
