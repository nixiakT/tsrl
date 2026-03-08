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


@dataclass(slots=True)
class PatchTSTPretrainingDataset:
    observations: np.ndarray
    classification_labels: np.ndarray | None = None
    regression_targets: np.ndarray | None = None


CLASSIFICATION_TASKS = {"regime_classification", "joint_regime_return"}
REGRESSION_TASKS = {"future_return_regression", "joint_regime_return"}
SUPPORTED_PRETRAINING_TASKS = CLASSIFICATION_TASKS | REGRESSION_TASKS


def _task_uses_classification(task_type: str) -> bool:
    return task_type in CLASSIFICATION_TASKS


def _task_uses_regression(task_type: str) -> bool:
    return task_type in REGRESSION_TASKS


def _task_heads(task_type: str) -> list[str]:
    heads: list[str] = []
    if _task_uses_classification(task_type):
        heads.append("regime_classification")
    if _task_uses_regression(task_type):
        heads.append("future_return_regression")
    return heads


def _future_return_value(
    prices: np.ndarray,
    current_index: int,
    forecast_horizon: int,
) -> float:
    current_price = prices[current_index]
    future_price = prices[current_index + forecast_horizon]
    if np.ndim(current_price) > 0:
        current_value = float(np.mean(current_price))
        future_value = float(np.mean(future_price))
    else:
        current_value = float(current_price)
        future_value = float(future_price)
    return float((future_value - current_value) / max(current_value, 1e-8))


def _label_future_regime(
    future_return: float,
    regime_threshold: float,
) -> int:
    if future_return > regime_threshold:
        return 2
    if future_return < -regime_threshold:
        return 0
    return 1


def _build_patchtst_pretraining_dataset(
    prices: np.ndarray,
    *,
    encoder,
    forecast_horizon: int,
    regime_threshold: float,
    task_type: str,
) -> PatchTSTPretrainingDataset:
    observations: list[np.ndarray] = []
    classification_labels: list[int] | None = [] if _task_uses_classification(task_type) else None
    regression_targets: list[float] | None = [] if _task_uses_regression(task_type) else None
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
        future_return = _future_return_value(
            prices_array,
            current_index=current_index,
            forecast_horizon=forecast_horizon,
        )
        if classification_labels is not None:
            classification_labels.append(_label_future_regime(future_return, regime_threshold=regime_threshold))
        if regression_targets is not None:
            regression_targets.append(future_return)
    if not observations:
        raise ValueError("pretraining dataset is empty; increase series length or reduce forecast_horizon")
    return PatchTSTPretrainingDataset(
        observations=np.asarray(observations, dtype=np.float32),
        classification_labels=(
            None
            if classification_labels is None
            else np.asarray(classification_labels, dtype=np.int64)
        ),
        regression_targets=(
            None if regression_targets is None else np.asarray(regression_targets, dtype=np.float32)
        ),
    )


def _classification_accuracy(logits, labels) -> float:
    predictions = torch.argmax(logits, dim=-1)
    return float((predictions == labels).float().mean().item())


def _regression_mae(predictions, labels) -> float:
    return float(torch.mean(torch.abs(predictions - labels)).item())


def _regression_rmse(predictions, labels) -> float:
    return float(torch.sqrt(torch.mean(torch.square(predictions - labels))).item())


def _regression_correlation(predictions, labels) -> float:
    pred_centered = predictions - torch.mean(predictions)
    label_centered = labels - torch.mean(labels)
    denominator = torch.sqrt(torch.sum(torch.square(pred_centered)) * torch.sum(torch.square(label_centered)))
    if float(denominator.item()) <= 1e-8:
        return 0.0
    return float((torch.sum(pred_centered * label_centered) / denominator).item())


def pretrain_patchtst_backbone(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float | None = None,
    task_type: str = "regime_classification",
) -> tuple[PretrainingArtifacts, dict]:
    if torch is None or TorchPatchTSTActorCriticNetwork is None:
        raise ModuleNotFoundError("torch is required for PatchTST pretraining")

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if task_type not in SUPPORTED_PRETRAINING_TASKS:
        raise ValueError(
            "task_type must be 'regime_classification', 'future_return_regression', "
            "or 'joint_regime_return'"
        )

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
    train_dataset = _build_patchtst_pretraining_dataset(
        splits.train_prices,
        encoder=encoder,
        forecast_horizon=forecast_horizon,
        regime_threshold=regime_threshold,
        task_type=task_type,
    )
    min_validation_points = encoder.window_size + forecast_horizon + 1
    validation_prices = splits.val_prices if len(splits.val_prices) >= min_validation_points else splits.eval_prices
    val_dataset = _build_patchtst_pretraining_dataset(
        validation_prices,
        encoder=encoder,
        forecast_horizon=forecast_horizon,
        regime_threshold=regime_threshold,
        task_type=task_type,
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
    channel_independent = bool(params.get("channel_independent", False))
    aux_loss_coef = max(0.0, float(params.get("aux_loss_coef", 0.0)))
    aux_mask_ratio = float(params.get("aux_mask_ratio", 0.4))
    classification_loss_coef = max(0.0, float(params.get("pretrain_classification_loss_coef", 1.0)))
    regression_loss_coef = max(0.0, float(params.get("pretrain_regression_loss_coef", 1.0)))
    if _task_uses_classification(task_type) and classification_loss_coef <= 0.0:
        raise ValueError("classification pretraining requires pretrain_classification_loss_coef > 0")
    if _task_uses_regression(task_type) and regression_loss_coef <= 0.0:
        raise ValueError("regression pretraining requires pretrain_regression_loss_coef > 0")
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
        channel_independent=channel_independent,
    ).to(device)
    classification_head = nn.Linear(hidden_size, 3).to(device) if _task_uses_classification(task_type) else None
    regression_head = nn.Linear(hidden_size, 1).to(device) if _task_uses_regression(task_type) else None
    optimizer_parameters = [
        {"params": list(network.backbone_parameters()), "lr": float(learning_rate or config.agent.policy_lr)},
    ]
    if classification_head is not None:
        optimizer_parameters.append(
            {"params": classification_head.parameters(), "lr": float(learning_rate or config.agent.policy_lr)}
        )
    if regression_head is not None:
        optimizer_parameters.append(
            {"params": regression_head.parameters(), "lr": float(learning_rate or config.agent.policy_lr)}
        )
    optimizer = optim.Adam(optimizer_parameters)

    train_observations_tensor = torch.as_tensor(train_dataset.observations, dtype=torch.float32, device=device)
    train_classification_tensor = (
        None
        if train_dataset.classification_labels is None
        else torch.as_tensor(train_dataset.classification_labels, dtype=torch.long, device=device)
    )
    train_regression_tensor = (
        None
        if train_dataset.regression_targets is None
        else torch.as_tensor(train_dataset.regression_targets, dtype=torch.float32, device=device)
    )
    val_observations_tensor = torch.as_tensor(val_dataset.observations, dtype=torch.float32, device=device)
    val_classification_tensor = (
        None
        if val_dataset.classification_labels is None
        else torch.as_tensor(val_dataset.classification_labels, dtype=torch.long, device=device)
    )
    val_regression_tensor = (
        None
        if val_dataset.regression_targets is None
        else torch.as_tensor(val_dataset.regression_targets, dtype=torch.float32, device=device)
    )

    pretrain_root = ensure_dir(output_dir or Path("runs") / f"{config.experiment_name}_patchtst_pretrain")
    checkpoint_path = pretrain_root / "backbone_checkpoint.pt"
    summary_path = pretrain_root / "summary.json"
    history_csv_path = pretrain_root / "history.csv"
    config_copy_path = pretrain_root / "config.json"
    dump_json(config_copy_path, config.to_dict())

    history: list[dict[str, float]] = []
    if task_type == "regime_classification":
        best_selection_value = float("-inf")
        best_selection_mode = "max"
        selection_metric_name = "val_accuracy"
    elif task_type == "future_return_regression":
        best_selection_value = float("inf")
        best_selection_mode = "min"
        selection_metric_name = "val_mae"
    else:
        best_selection_value = float("inf")
        best_selection_mode = "min"
        selection_metric_name = "val_joint_loss"
    best_summary: dict[str, float] | None = None
    resolved_batch_size = min(batch_size, len(train_dataset.observations))

    for epoch in range(1, epochs + 1):
        network.train()
        if classification_head is not None:
            classification_head.train()
        if regression_head is not None:
            regression_head.train()
        order = torch.randperm(len(train_observations_tensor), device=device)
        train_loss_total = 0.0
        train_classification_loss_total = 0.0
        train_regression_loss_total = 0.0
        train_accuracy_total = 0.0
        train_mae_total = 0.0
        train_rmse_total = 0.0
        train_aux_loss_total = 0.0
        train_aux_ratio_total = 0.0
        step_count = 0

        for start_index in range(0, len(train_observations_tensor), resolved_batch_size):
            batch_index = order[start_index : start_index + resolved_batch_size]
            batch_observations = train_observations_tensor[batch_index]
            features = network.encode_features(batch_observations)
            supervised_loss = torch.zeros((), dtype=batch_observations.dtype, device=device)
            classification_loss_value = 0.0
            regression_loss_value = 0.0
            accuracy_value = 0.0
            mae_value = 0.0
            rmse_value = 0.0
            if classification_head is not None and train_classification_tensor is not None:
                classification_labels = train_classification_tensor[batch_index]
                classification_logits = classification_head(features)
                classification_loss = nn.functional.cross_entropy(classification_logits, classification_labels)
                supervised_loss = supervised_loss + (classification_loss_coef * classification_loss)
                classification_loss_value = float(classification_loss.item())
                accuracy_value = _classification_accuracy(classification_logits, classification_labels)
            if regression_head is not None and train_regression_tensor is not None:
                regression_labels = train_regression_tensor[batch_index]
                regression_predictions = regression_head(features).squeeze(-1)
                regression_loss = nn.functional.mse_loss(regression_predictions, regression_labels)
                supervised_loss = supervised_loss + (regression_loss_coef * regression_loss)
                regression_loss_value = float(regression_loss.item())
                mae_value = _regression_mae(regression_predictions, regression_labels)
                rmse_value = _regression_rmse(regression_predictions, regression_labels)
            aux_loss_value = torch.zeros((), dtype=batch_observations.dtype, device=device)
            realized_mask_ratio = 0.0
            if aux_loss_coef > 0.0:
                aux_loss_value, realized_mask_ratio = network.masked_patch_reconstruction_loss(
                    batch_observations,
                    mask_ratio=aux_mask_ratio,
                )
            total_loss = supervised_loss + (aux_loss_coef * aux_loss_value)
            optimizer.zero_grad()
            total_loss.backward()
            parameters_for_clip = list(network.backbone_parameters())
            if classification_head is not None:
                parameters_for_clip.extend(list(classification_head.parameters()))
            if regression_head is not None:
                parameters_for_clip.extend(list(regression_head.parameters()))
            nn.utils.clip_grad_norm_(parameters_for_clip, 1.0)
            optimizer.step()

            train_loss_total += float(supervised_loss.item())
            train_classification_loss_total += classification_loss_value
            train_regression_loss_total += regression_loss_value
            train_accuracy_total += accuracy_value
            train_mae_total += mae_value
            train_rmse_total += rmse_value
            train_aux_loss_total += float(aux_loss_value.item())
            train_aux_ratio_total += float(realized_mask_ratio)
            step_count += 1

        network.eval()
        if classification_head is not None:
            classification_head.eval()
        if regression_head is not None:
            regression_head.eval()
        with torch.no_grad():
            val_features = network.encode_features(val_observations_tensor)
            record = {
                "epoch": float(epoch),
                "train_loss": train_loss_total / max(step_count, 1),
                "train_aux_loss": train_aux_loss_total / max(step_count, 1),
                "train_aux_mask_ratio": train_aux_ratio_total / max(step_count, 1),
            }
            if classification_head is not None and val_classification_tensor is not None:
                val_classification_logits = classification_head(val_features)
                val_classification_loss = float(
                    nn.functional.cross_entropy(val_classification_logits, val_classification_tensor).item()
                )
                val_accuracy = _classification_accuracy(val_classification_logits, val_classification_tensor)
                record.update(
                    {
                        "train_classification_loss": train_classification_loss_total / max(step_count, 1),
                        "train_accuracy": train_accuracy_total / max(step_count, 1),
                        "val_classification_loss": val_classification_loss,
                        "val_accuracy": val_accuracy,
                    }
                )
            if regression_head is not None and val_regression_tensor is not None:
                val_regression_predictions = regression_head(val_features).squeeze(-1)
                val_regression_loss = float(
                    nn.functional.mse_loss(val_regression_predictions, val_regression_tensor).item()
                )
                val_mae = _regression_mae(val_regression_predictions, val_regression_tensor)
                val_rmse = _regression_rmse(val_regression_predictions, val_regression_tensor)
                val_corr = _regression_correlation(val_regression_predictions, val_regression_tensor)
                record.update(
                    {
                        "train_regression_loss": train_regression_loss_total / max(step_count, 1),
                        "train_mae": train_mae_total / max(step_count, 1),
                        "train_rmse": train_rmse_total / max(step_count, 1),
                        "val_regression_loss": val_regression_loss,
                        "val_mae": val_mae,
                        "val_rmse": val_rmse,
                        "val_correlation": val_corr,
                    }
                )
            if task_type == "regime_classification":
                record["val_loss"] = float(record["val_classification_loss"])
                val_selection_value = float(record["val_accuracy"])
            elif task_type == "future_return_regression":
                record["val_loss"] = float(record["val_regression_loss"])
                val_selection_value = float(record["val_mae"])
            else:
                joint_loss = (
                    classification_loss_coef * float(record["val_classification_loss"])
                    + regression_loss_coef * float(record["val_regression_loss"])
                )
                record["train_joint_loss"] = (
                    classification_loss_coef * float(record["train_classification_loss"])
                    + regression_loss_coef * float(record["train_regression_loss"])
                )
                record["val_joint_loss"] = joint_loss
                record["val_loss"] = joint_loss
                val_selection_value = joint_loss
        history.append(record)

        is_better = (
            val_selection_value >= best_selection_value
            if best_selection_mode == "max"
            else val_selection_value <= best_selection_value
        )
        if is_better:
            best_selection_value = val_selection_value
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
                        "channel_independent": channel_independent,
                    },
                    "task": task_type,
                    "task_heads": _task_heads(task_type),
                    "label_count": 3 if _task_uses_classification(task_type) else 1,
                    "best_selection_metric": selection_metric_name,
                    "best_selection_value": best_selection_value,
                },
                checkpoint_path,
            )

    dump_records_csv(history_csv_path, history)
    summary = {
        "experiment_name": config.experiment_name,
        "agent": config.agent.name,
        "encoder": config.encoder.name,
        "task": task_type,
        "task_heads": _task_heads(task_type),
        "epochs": int(epochs),
        "batch_size": int(resolved_batch_size),
        "learning_rate": float(learning_rate or config.agent.policy_lr),
        "train_examples": int(len(train_dataset.observations)),
        "val_examples": int(len(val_dataset.observations)),
        "selection_metric": selection_metric_name,
        "selection_mode": best_selection_mode,
        "best_selection_value": float(best_selection_value),
        "classification_loss_coef": float(classification_loss_coef),
        "regression_loss_coef": float(regression_loss_coef),
        "history": history,
        "best_epoch": int(best_summary["epoch"]) if best_summary is not None else None,
        "best_metrics": best_summary,
        "checkpoint_path": str(checkpoint_path),
    }
    if task_type == "regime_classification":
        summary["best_val_accuracy"] = float(best_selection_value)
    elif task_type == "future_return_regression":
        summary["best_val_mae"] = float(best_selection_value)
    else:
        summary["best_val_joint_loss"] = float(best_selection_value)
    dump_json(summary_path, summary)
    artifacts = PretrainingArtifacts(
        root_dir=pretrain_root,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        history_csv_path=history_csv_path,
        config_copy_path=config_copy_path,
    )
    return artifacts, summary
