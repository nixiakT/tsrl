from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tsrl_lite.algorithms  # noqa: F401
import tsrl_lite.encoders  # noqa: F401
import tsrl_lite.envs  # noqa: F401
from tsrl_lite.builders import build_encoder, build_env, required_future_steps
from tsrl_lite.config import load_experiment_config
from tsrl_lite.data import resolve_price_splits
from tsrl_lite.pretrain import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    _build_patchtst_pretraining_dataset,
    _classification_accuracy,
    _regression_correlation,
    _regression_mae,
    _regression_rmse,
    _task_heads,
    _task_uses_classification,
    _task_uses_regression,
    _task_uses_vector_regression,
)
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
class ProbeArtifacts:
    root_dir: Path
    checkpoint_path: Path
    summary_path: Path
    history_csv_path: Path
    config_copy_path: Path


SUPPORTED_PROBE_TASKS = CLASSIFICATION_TASKS | REGRESSION_TASKS


def _extract_backbone_payload(payload: dict) -> tuple[dict, dict, dict]:
    backbone_state_dict = payload.get("backbone_state_dict", payload.get("state_dict"))
    if not isinstance(backbone_state_dict, dict):
        raise ValueError("checkpoint does not contain a PatchTST backbone state dict")
    raw_config = payload.get("backbone_config", payload.get("config"))
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint does not contain PatchTST backbone configuration metadata")
    backbone_config = {
        key: raw_config[key]
        for key in (
            "input_dim",
            "sequence_length",
            "hidden_size",
            "patch_len",
            "stride",
            "num_layers",
            "num_heads",
            "dropout",
            "use_cls_token",
            "channel_independent",
        )
        if key in raw_config
    }
    missing_keys = {
        "input_dim",
        "sequence_length",
        "hidden_size",
        "patch_len",
        "stride",
        "num_layers",
        "num_heads",
        "dropout",
        "use_cls_token",
        "channel_independent",
    } - set(backbone_config)
    if missing_keys:
        missing_list = ", ".join(sorted(missing_keys))
        raise ValueError(f"checkpoint backbone configuration is missing required keys: {missing_list}")
    source_metadata = {
        "task": payload.get("task"),
        "task_heads": payload.get("task_heads"),
        "best_selection_metric": payload.get("best_selection_metric"),
        "best_selection_value": payload.get("best_selection_value"),
        "regression_target_dim": payload.get("regression_target_dim"),
    }
    return backbone_state_dict, backbone_config, source_metadata


def _validate_backbone_encoder_compatibility(backbone_config: dict, observation_shape: tuple[int, ...]) -> None:
    if len(observation_shape) != 2:
        raise ValueError("PatchTST linear probing requires a sequence-shaped encoder output")
    sequence_length, input_dim = observation_shape
    mismatches: list[str] = []
    if int(backbone_config["sequence_length"]) != int(sequence_length):
        mismatches.append(
            f"sequence_length={backbone_config['sequence_length']!r} "
            f"(expected {int(sequence_length)!r})"
        )
    if int(backbone_config["input_dim"]) != int(input_dim):
        mismatches.append(f"input_dim={backbone_config['input_dim']!r} (expected {int(input_dim)!r})")
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(f"incompatible PatchTST probe config: {mismatch_text}")


def _encode_backbone_features(network, observations, batch_size: int):
    encoded_batches = []
    network.eval()
    with torch.no_grad():
        for start_index in range(0, observations.shape[0], batch_size):
            batch_observations = observations[start_index : start_index + batch_size]
            encoded_batches.append(network.encode_features(batch_observations))
    return torch.cat(encoded_batches, dim=0)


def probe_patchtst_backbone(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float | None = None,
    task_type: str = "regime_classification",
) -> tuple[ProbeArtifacts, dict]:
    if torch is None or TorchPatchTSTActorCriticNetwork is None:
        raise ModuleNotFoundError("torch is required for PatchTST linear probing")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if task_type not in SUPPORTED_PROBE_TASKS:
        raise ValueError(
            "task_type must be 'regime_classification', 'future_return_regression', "
            "'future_return_vector_regression', or 'joint_regime_return'"
        )

    config = load_experiment_config(config_path)
    if config.agent.name != "torch-patchtst-ppo":
        raise ValueError("probe-patchtst currently requires agent.name='torch-patchtst-ppo'")
    probe_future_steps = int(config.env.params.get("forecast_horizon", 8))

    splits = resolve_price_splits(
        config=config.data,
        seed=config.seed,
        min_future_steps=max(required_future_steps(config), probe_future_steps),
    )
    train_env = build_env(config, prices=splits.train_prices, seed=config.seed, index_offset=splits.train_offset)
    encoder = build_encoder(config, train_env)
    if len(encoder.observation_shape) != 2:
        raise ValueError("PatchTST linear probing requires a sequence-shaped encoder output")

    regime_threshold = float(config.env.params.get("regime_threshold", 0.002))
    forecast_horizon = probe_future_steps
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

    params = dict(config.agent.params)
    device_name = str(params.get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    learning_rate_value = float(learning_rate or config.agent.policy_lr)

    payload = torch.load(Path(checkpoint_path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a serialized PatchTST payload")
    backbone_state_dict, backbone_config, source_metadata = _extract_backbone_payload(payload)
    _validate_backbone_encoder_compatibility(backbone_config, encoder.observation_shape)

    torch.manual_seed(config.seed)
    network = TorchPatchTSTActorCriticNetwork(
        input_dim=int(backbone_config["input_dim"]),
        sequence_length=int(backbone_config["sequence_length"]),
        hidden_size=int(backbone_config["hidden_size"]),
        action_dim=1,
        patch_len=int(backbone_config["patch_len"]),
        stride=int(backbone_config["stride"]),
        num_layers=int(backbone_config["num_layers"]),
        num_heads=int(backbone_config["num_heads"]),
        dropout=float(backbone_config["dropout"]),
        use_cls_token=bool(backbone_config["use_cls_token"]),
        channel_independent=bool(backbone_config["channel_independent"]),
    ).to(device)
    network.load_backbone_state_dict(backbone_state_dict)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    network.eval()

    train_observations_tensor = torch.as_tensor(train_dataset.observations, dtype=torch.float32, device=device)
    val_observations_tensor = torch.as_tensor(val_dataset.observations, dtype=torch.float32, device=device)
    resolved_feature_batch_size = min(batch_size, len(train_dataset.observations))
    train_features = _encode_backbone_features(network, train_observations_tensor, resolved_feature_batch_size)
    val_features = _encode_backbone_features(network, val_observations_tensor, resolved_feature_batch_size)

    train_classification_tensor = (
        None
        if train_dataset.classification_labels is None
        else torch.as_tensor(train_dataset.classification_labels, dtype=torch.long, device=device)
    )
    val_classification_tensor = (
        None
        if val_dataset.classification_labels is None
        else torch.as_tensor(val_dataset.classification_labels, dtype=torch.long, device=device)
    )
    train_regression_tensor = (
        None
        if train_dataset.regression_targets is None
        else torch.as_tensor(train_dataset.regression_targets, dtype=torch.float32, device=device)
    )
    val_regression_tensor = (
        None
        if val_dataset.regression_targets is None
        else torch.as_tensor(val_dataset.regression_targets, dtype=torch.float32, device=device)
    )

    classification_head = (
        nn.Linear(int(backbone_config["hidden_size"]), 3).to(device) if _task_uses_classification(task_type) else None
    )
    regression_target_dim = (
        0
        if train_dataset.regression_targets is None
        else 1 if train_dataset.regression_targets.ndim == 1 else int(train_dataset.regression_targets.shape[1])
    )
    regression_head = (
        nn.Linear(int(backbone_config["hidden_size"]), max(regression_target_dim, 1)).to(device)
        if _task_uses_regression(task_type)
        else None
    )
    optimizer_parameters = []
    if classification_head is not None:
        optimizer_parameters.append({"params": classification_head.parameters(), "lr": learning_rate_value})
    if regression_head is not None:
        optimizer_parameters.append({"params": regression_head.parameters(), "lr": learning_rate_value})
    optimizer = optim.Adam(optimizer_parameters)

    probe_root = ensure_dir(output_dir or Path("runs") / f"{config.experiment_name}_patchtst_probe")
    probe_checkpoint_path = probe_root / "probe_checkpoint.pt"
    summary_path = probe_root / "summary.json"
    history_csv_path = probe_root / "history.csv"
    config_copy_path = probe_root / "config.json"
    dump_json(config_copy_path, config.to_dict())

    history: list[dict[str, float]] = []
    if task_type == "regime_classification":
        best_selection_value = float("-inf")
        best_selection_mode = "max"
        selection_metric_name = "val_accuracy"
    elif task_type in {"future_return_regression", "future_return_vector_regression"}:
        best_selection_value = float("inf")
        best_selection_mode = "min"
        selection_metric_name = "val_mae"
    else:
        best_selection_value = float("inf")
        best_selection_mode = "min"
        selection_metric_name = "val_joint_loss"
    best_summary: dict[str, float] | None = None
    resolved_train_batch_size = min(batch_size, len(train_dataset.observations))

    for epoch in range(1, epochs + 1):
        if classification_head is not None:
            classification_head.train()
        if regression_head is not None:
            regression_head.train()
        order = torch.randperm(train_features.shape[0], device=device)
        train_loss_total = 0.0
        train_classification_loss_total = 0.0
        train_regression_loss_total = 0.0
        train_accuracy_total = 0.0
        train_mae_total = 0.0
        train_rmse_total = 0.0
        step_count = 0

        for start_index in range(0, train_features.shape[0], resolved_train_batch_size):
            batch_index = order[start_index : start_index + resolved_train_batch_size]
            batch_features = train_features[batch_index]
            total_loss = torch.zeros((), dtype=batch_features.dtype, device=device)
            classification_loss_value = 0.0
            regression_loss_value = 0.0
            accuracy_value = 0.0
            mae_value = 0.0
            rmse_value = 0.0

            if classification_head is not None and train_classification_tensor is not None:
                classification_labels = train_classification_tensor[batch_index]
                classification_logits = classification_head(batch_features)
                classification_loss = nn.functional.cross_entropy(classification_logits, classification_labels)
                total_loss = total_loss + classification_loss
                classification_loss_value = float(classification_loss.item())
                accuracy_value = _classification_accuracy(classification_logits, classification_labels)

            if regression_head is not None and train_regression_tensor is not None:
                regression_labels = train_regression_tensor[batch_index]
                regression_predictions = regression_head(batch_features)
                if regression_target_dim == 1:
                    regression_predictions = regression_predictions.squeeze(-1)
                regression_loss = nn.functional.mse_loss(regression_predictions, regression_labels)
                total_loss = total_loss + regression_loss
                regression_loss_value = float(regression_loss.item())
                mae_value = _regression_mae(regression_predictions, regression_labels)
                rmse_value = _regression_rmse(regression_predictions, regression_labels)

            optimizer.zero_grad()
            total_loss.backward()
            parameters_for_clip = []
            if classification_head is not None:
                parameters_for_clip.extend(list(classification_head.parameters()))
            if regression_head is not None:
                parameters_for_clip.extend(list(regression_head.parameters()))
            nn.utils.clip_grad_norm_(parameters_for_clip, 1.0)
            optimizer.step()

            train_loss_total += float(total_loss.item())
            train_classification_loss_total += classification_loss_value
            train_regression_loss_total += regression_loss_value
            train_accuracy_total += accuracy_value
            train_mae_total += mae_value
            train_rmse_total += rmse_value
            step_count += 1

        if classification_head is not None:
            classification_head.eval()
        if regression_head is not None:
            regression_head.eval()
        with torch.no_grad():
            record = {
                "epoch": float(epoch),
                "train_loss": train_loss_total / max(step_count, 1),
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
                val_regression_predictions = regression_head(val_features)
                if regression_target_dim == 1:
                    val_regression_predictions = val_regression_predictions.squeeze(-1)
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
            elif task_type in {"future_return_regression", "future_return_vector_regression"}:
                record["val_loss"] = float(record["val_regression_loss"])
                val_selection_value = float(record["val_mae"])
            else:
                joint_loss = float(record["val_classification_loss"]) + float(record["val_regression_loss"])
                record["train_joint_loss"] = (
                    float(record["train_classification_loss"]) + float(record["train_regression_loss"])
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
            best_selection_value = float(val_selection_value)
            best_summary = dict(record)
            torch.save(
                {
                    "probe_task": task_type,
                    "probe_task_heads": _task_heads(task_type),
                    "probe_head_state_dict": {
                        "classification": None
                        if classification_head is None
                        else classification_head.state_dict(),
                        "regression": None if regression_head is None else regression_head.state_dict(),
                    },
                    "backbone_config": backbone_config,
                    "source_checkpoint_path": str(Path(checkpoint_path)),
                    "source_pretrained_task": source_metadata["task"],
                    "source_pretrained_task_heads": source_metadata["task_heads"],
                    "source_selection_metric": source_metadata["best_selection_metric"],
                    "source_selection_value": source_metadata["best_selection_value"],
                    "source_regression_target_dim": source_metadata["regression_target_dim"],
                    "selection_metric": selection_metric_name,
                    "selection_mode": best_selection_mode,
                    "best_selection_value": best_selection_value,
                    "regression_target_dim": int(regression_target_dim),
                    "label_count": (
                        3
                        if _task_uses_classification(task_type)
                        else max(regression_target_dim, 1) if _task_uses_regression(task_type) else 0
                    ),
                },
                probe_checkpoint_path,
            )

    dump_records_csv(history_csv_path, history)
    summary = {
        "experiment_name": config.experiment_name,
        "agent": config.agent.name,
        "encoder": config.encoder.name,
        "task": task_type,
        "task_heads": _task_heads(task_type),
        "epochs": int(epochs),
        "batch_size": int(resolved_train_batch_size),
        "feature_batch_size": int(resolved_feature_batch_size),
        "learning_rate": learning_rate_value,
        "train_examples": int(len(train_dataset.observations)),
        "val_examples": int(len(val_dataset.observations)),
        "selection_metric": selection_metric_name,
        "selection_mode": best_selection_mode,
        "best_selection_value": float(best_selection_value),
        "regression_target_dim": int(regression_target_dim),
        "history": history,
        "best_epoch": int(best_summary["epoch"]) if best_summary is not None else None,
        "best_metrics": best_summary,
        "checkpoint_path": str(probe_checkpoint_path),
        "source_checkpoint_path": str(Path(checkpoint_path)),
        "source_pretrained_task": source_metadata["task"],
        "source_pretrained_task_heads": source_metadata["task_heads"],
        "source_selection_metric": source_metadata["best_selection_metric"],
        "source_selection_value": source_metadata["best_selection_value"],
        "source_regression_target_dim": source_metadata["regression_target_dim"],
        "source_backbone_config": backbone_config,
    }
    if task_type == "regime_classification":
        summary["best_val_accuracy"] = float(best_selection_value)
    elif task_type in {"future_return_regression", "future_return_vector_regression"}:
        summary["best_val_mae"] = float(best_selection_value)
    else:
        summary["best_val_joint_loss"] = float(best_selection_value)
    dump_json(summary_path, summary)
    artifacts = ProbeArtifacts(
        root_dir=probe_root,
        checkpoint_path=probe_checkpoint_path,
        summary_path=summary_path,
        history_csv_path=history_csv_path,
        config_copy_path=config_copy_path,
    )
    return artifacts, summary
