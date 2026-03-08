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

    class TorchPatchTSTActorCriticNetwork(nn.Module):
        BACKBONE_PREFIXES = (
            "patch_projection.",
            "reconstruction_head.",
            "position_embedding",
            "cls_token",
            "mask_token",
            "encoder.",
            "norm.",
        )

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
            channel_independent: bool = False,
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
            self.channel_independent = bool(channel_independent)
            self.num_patches = num_patches
            self.patch_token_count = num_patches * input_dim if self.channel_independent else num_patches

            projection_input_dim = patch_len if self.channel_independent else patch_len * input_dim
            self.patch_projection = nn.Linear(projection_input_dim, hidden_size)
            self.reconstruction_head = nn.Linear(hidden_size, projection_input_dim)
            self.input_dropout = nn.Dropout(dropout)
            token_count = self.patch_token_count + (1 if self.use_cls_token else 0)
            self.position_embedding = nn.Parameter(torch.zeros(1, token_count, hidden_size))
            if self.use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            else:
                self.register_parameter("cls_token", None)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
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

        def patchify(self, sequence_batch):
            patches = sequence_batch.unfold(dimension=1, size=self.patch_len, step=self.stride)
            if self.channel_independent:
                return patches.contiguous().view(sequence_batch.size(0), self.patch_token_count, self.patch_len)
            patches = patches.permute(0, 1, 3, 2).contiguous()
            return patches.view(sequence_batch.size(0), self.num_patches, self.patch_len * self.input_dim)

        def encode_patch_tokens(self, sequence_batch, patch_mask=None):
            patch_embeddings = self.patch_projection(self.patchify(sequence_batch))
            if patch_mask is not None:
                if tuple(patch_mask.shape) != (sequence_batch.size(0), self.patch_token_count):
                    raise ValueError("patch_mask must match batch_size x patch_token_count")
                masked_embeddings = self.mask_token.expand(sequence_batch.size(0), self.patch_token_count, -1)
                patch_embeddings = torch.where(patch_mask.unsqueeze(-1), masked_embeddings, patch_embeddings)
            tokens = patch_embeddings
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(sequence_batch.size(0), -1, -1)
                tokens = torch.cat([cls_tokens, tokens], dim=1)
            tokens = self.input_dropout(tokens + self.position_embedding[:, : tokens.size(1), :])
            return self.encoder(tokens)

        def encode_features(self, sequence_batch):
            encoded = self.encode_patch_tokens(sequence_batch)
            if self.use_cls_token:
                features = encoded[:, 0, :]
            else:
                features = encoded.mean(dim=1)
            features = self.norm(features)
            return features

        def masked_patch_reconstruction_loss(self, sequence_batch, mask_ratio: float):
            if mask_ratio <= 0.0:
                zero = torch.zeros((), dtype=sequence_batch.dtype, device=sequence_batch.device)
                return zero, 0.0
            patches = self.patchify(sequence_batch)
            batch_size = sequence_batch.size(0)
            patch_count = self.patch_token_count
            mask_count = max(1, min(patch_count, int(round(patch_count * float(mask_ratio)))))
            random_scores = torch.rand(batch_size, patch_count, device=sequence_batch.device)
            mask = random_scores.argsort(dim=1) < mask_count
            encoded = self.encode_patch_tokens(sequence_batch, patch_mask=mask)
            encoded_patches = encoded[:, 1:, :] if self.use_cls_token else encoded
            reconstruction = self.reconstruction_head(encoded_patches)
            loss = functional.mse_loss(reconstruction[mask], patches[mask])
            return loss, float(mask.float().mean().item())

        def forward(self, sequence_batch):
            features = self.encode_features(sequence_batch)
            logits = self.policy_head(features)
            value = self.value_head(features).squeeze(-1)
            return logits, value

        def backbone_state_dict(self) -> dict[str, torch.Tensor]:
            state = self.state_dict()
            return {
                key: value
                for key, value in state.items()
                if key.startswith(self.BACKBONE_PREFIXES)
            }

        def load_backbone_state_dict(self, backbone_state_dict: dict[str, torch.Tensor]) -> None:
            current_state = self.state_dict()
            matched_state = {
                key: value
                for key, value in backbone_state_dict.items()
                if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
            }
            if not matched_state:
                raise ValueError("no compatible PatchTST backbone parameters found in checkpoint")
            current_state.update(matched_state)
            self.load_state_dict(current_state)

        def backbone_parameters(self):
            for module in (
                self.patch_projection,
                self.encoder,
                self.norm,
                self.reconstruction_head,
            ):
                yield from module.parameters()
            yield self.position_embedding
            yield self.mask_token
            if self.cls_token is not None:
                yield self.cls_token

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
                    "channel_independent": self.channel_independent,
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
