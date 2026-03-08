from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tsrl_lite.algorithms  # noqa: F401
import tsrl_lite.encoders  # noqa: F401
import tsrl_lite.envs  # noqa: F401
from tsrl_lite.builders import build_encoder, build_env, required_future_steps
from tsrl_lite.config import load_experiment_config
from tsrl_lite.data import resolve_price_splits
from tsrl_lite.state import TimeSeriesState
from tsrl_lite.utils import dump_json, dump_records_csv, ensure_dir

try:
    import torch
    from torch import nn, optim
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    optim = None

from tsrl_lite.networks import TorchPatchTSTActorCriticNetwork


@dataclass(slots=True)
class PretrainingArtifacts:
    root_dir: Path
    checkpoint_path: Path
    summary_path: Path
    history_csv_path: Path
    config_copy_path: Path


def _label_future_regime(
    prices: np.ndarray,
    current_index: int,
    forecast_horizon: int,
    regime_threshold: float,
) -> int:
    current_price = prices[current_index]
    future_price = prices[current_index + forecast_horizon]
    if np.ndim(current_price) > 0:
        current_value = float(np.mean(current_price))
        future_value = float(np.mean(future_price))
    else:
        current_value = float(current_price)
        future_value = float(future_price)
    future_return = (future_value - current_value) / max(current_value, 1e-8)
    if future_return > regime_threshold:
        return 2
    if future_return < -regime_threshold:
        return 0
    return 1


def _build_regime_pretraining_dataset(
    prices: np.ndarray,
    *,
    encoder,
    forecast_horizon: int,
    regime_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    labels: list[int] = []
    prices_array = np.asarray(prices, dtype=float)
    for current_index in range(encoder.window_size - 1, len(prices_array) - forecast_horizon):
        window = prices_array[current_index - encoder.window_size + 1 : current_index + 1]
        state = TimeSeriesState(
            window=window,
            agent_features=np.zeros(encoder.agent_feature_dim, dtype=float),
            step=current_index,
            context={"task": "patchtst_pretraining"},
        )
        observations.append(encoder.encode(state).astype(np.float32))
        labels.append(
            _label_future_regime(
                prices_array,
                current_index=current_index,
                forecast_horizon=forecast_horizon,
                regime_threshold=regime_threshold,
            )
        )
    if not observations:
        raise ValueError("pretraining dataset is empty; increase series length or reduce forecast_horizon")
    return np.asarray(observations, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _classification_accuracy(logits, labels) -> float:
    predictions = torch.argmax(logits, dim=-1)
    return float((predictions == labels).float().mean().item())


def pretrain_patchtst_backbone(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float | None = None,
) -> tuple[PretrainingArtifacts, dict]:
    if torch is None or TorchPatchTSTActorCriticNetwork is None:
        raise ModuleNotFoundError("torch is required for PatchTST pretraining")

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    config = load_experiment_config(config_path)
    if config.agent.name != "torch-patchtst-ppo":
        raise ValueError("pretrain-patchtst currently requires agent.name='torch-patchtst-ppo'")
    if config.env.name != "regime-v0":
        raise ValueError("pretrain-patchtst currently requires env.name='regime-v0'")

    splits = resolve_price_splits(
        config=config.data,
        seed=config.seed,
        min_future_steps=required_future_steps(config),
    )
    train_env = build_env(config, prices=splits.train_prices, seed=config.seed, index_offset=splits.train_offset)
    encoder = build_encoder(config, train_env)
    if len(encoder.observation_shape) != 2:
        raise ValueError("PatchTST pretraining requires a sequence-shaped encoder output")

    forecast_horizon = int(config.env.params.get("forecast_horizon", 8))
    regime_threshold = float(config.env.params.get("regime_threshold", 0.002))
    train_observations, train_labels = _build_regime_pretraining_dataset(
        splits.train_prices,
        encoder=encoder,
        forecast_horizon=forecast_horizon,
        regime_threshold=regime_threshold,
    )
    min_validation_points = encoder.window_size + forecast_horizon + 1
    validation_prices = splits.val_prices if len(splits.val_prices) >= min_validation_points else splits.eval_prices
    val_observations, val_labels = _build_regime_pretraining_dataset(
        validation_prices,
        encoder=encoder,
        forecast_horizon=forecast_horizon,
        regime_threshold=regime_threshold,
    )

    sequence_length, input_dim = encoder.observation_shape
    params = dict(config.agent.params)
    hidden_size = int(params.get("hidden_size", 64))
    patch_len = int(params.get("patch_len", 8))
    stride = int(params.get("stride", 4))
    num_layers = int(params.get("num_layers", 2))
    num_heads = int(params.get("num_heads", 4))
    dropout = float(params.get("dropout", 0.1))
    use_cls_token = bool(params.get("use_cls_token", True))
    aux_loss_coef = max(0.0, float(params.get("aux_loss_coef", 0.0)))
    aux_mask_ratio = float(params.get("aux_mask_ratio", 0.4))
    device_name = str(params.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    torch.manual_seed(config.seed)
    network = TorchPatchTSTActorCriticNetwork(
        input_dim=input_dim,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        action_dim=1,
        patch_len=patch_len,
        stride=stride,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        use_cls_token=use_cls_token,
    ).to(device)
    classifier = nn.Linear(hidden_size, 3).to(device)
    optimizer = optim.Adam(
        [
            {"params": list(network.backbone_parameters()), "lr": float(learning_rate or config.agent.policy_lr)},
            {"params": classifier.parameters(), "lr": float(learning_rate or config.agent.policy_lr)},
        ]
    )

    train_observations_tensor = torch.as_tensor(train_observations, dtype=torch.float32, device=device)
    train_labels_tensor = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    val_observations_tensor = torch.as_tensor(val_observations, dtype=torch.float32, device=device)
    val_labels_tensor = torch.as_tensor(val_labels, dtype=torch.long, device=device)

    pretrain_root = ensure_dir(output_dir or Path("runs") / f"{config.experiment_name}_patchtst_pretrain")
    checkpoint_path = pretrain_root / "backbone_checkpoint.pt"
    summary_path = pretrain_root / "summary.json"
    history_csv_path = pretrain_root / "history.csv"
    config_copy_path = pretrain_root / "config.json"
    dump_json(config_copy_path, config.to_dict())

    history: list[dict[str, float]] = []
    best_val_accuracy = float("-inf")
    best_summary: dict[str, float] | None = None
    resolved_batch_size = min(batch_size, len(train_observations))

    for epoch in range(1, epochs + 1):
        network.train()
        classifier.train()
        order = torch.randperm(len(train_observations_tensor), device=device)
        train_loss_total = 0.0
        train_accuracy_total = 0.0
        train_aux_loss_total = 0.0
        train_aux_ratio_total = 0.0
        step_count = 0

        for start_index in range(0, len(train_observations_tensor), resolved_batch_size):
            batch_index = order[start_index : start_index + resolved_batch_size]
            batch_observations = train_observations_tensor[batch_index]
            batch_labels = train_labels_tensor[batch_index]
            features = network.encode_features(batch_observations)
            logits = classifier(features)
            classification_loss = nn.functional.cross_entropy(logits, batch_labels)
            aux_loss_value = torch.zeros((), dtype=batch_observations.dtype, device=device)
            realized_mask_ratio = 0.0
            if aux_loss_coef > 0.0:
                aux_loss_value, realized_mask_ratio = network.masked_patch_reconstruction_loss(
                    batch_observations,
                    mask_ratio=aux_mask_ratio,
                )
            total_loss = classification_loss + (aux_loss_coef * aux_loss_value)
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(list(network.backbone_parameters()) + list(classifier.parameters()), 1.0)
            optimizer.step()

            train_loss_total += float(classification_loss.item())
            train_accuracy_total += _classification_accuracy(logits, batch_labels)
            train_aux_loss_total += float(aux_loss_value.item())
            train_aux_ratio_total += float(realized_mask_ratio)
            step_count += 1

        network.eval()
        classifier.eval()
        with torch.no_grad():
            val_features = network.encode_features(val_observations_tensor)
            val_logits = classifier(val_features)
            val_loss = float(nn.functional.cross_entropy(val_logits, val_labels_tensor).item())
            val_accuracy = _classification_accuracy(val_logits, val_labels_tensor)

        record = {
            "epoch": float(epoch),
            "train_loss": train_loss_total / max(step_count, 1),
            "train_accuracy": train_accuracy_total / max(step_count, 1),
            "train_aux_loss": train_aux_loss_total / max(step_count, 1),
            "train_aux_mask_ratio": train_aux_ratio_total / max(step_count, 1),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(record)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_summary = dict(record)
            torch.save(
                {
                    "backbone_state_dict": network.backbone_state_dict(),
                    "backbone_config": {
                        "input_dim": input_dim,
                        "sequence_length": sequence_length,
                        "hidden_size": hidden_size,
                        "patch_len": patch_len,
                        "stride": stride,
                        "num_layers": num_layers,
                        "num_heads": num_heads,
                        "dropout": dropout,
                        "use_cls_token": use_cls_token,
                    },
                    "task": "regime_classification_pretraining",
                    "label_count": 3,
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

    dump_records_csv(history_csv_path, history)
    summary = {
        "experiment_name": config.experiment_name,
        "agent": config.agent.name,
        "encoder": config.encoder.name,
        "task": "regime_classification_pretraining",
        "epochs": int(epochs),
        "batch_size": int(resolved_batch_size),
        "learning_rate": float(learning_rate or config.agent.policy_lr),
        "train_examples": int(len(train_observations)),
        "val_examples": int(len(val_observations)),
        "best_val_accuracy": float(best_val_accuracy),
        "history": history,
        "best_epoch": int(best_summary["epoch"]) if best_summary is not None else None,
        "best_metrics": best_summary,
        "checkpoint_path": str(checkpoint_path),
    }
    dump_json(summary_path, summary)
    artifacts = PretrainingArtifacts(
        root_dir=pretrain_root,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        history_csv_path=history_csv_path,
        config_copy_path=config_copy_path,
    )
    return artifacts, summary
