from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tsrl_lite.builders import required_future_steps
from tsrl_lite.config import load_experiment_config
from tsrl_lite.data import generate_walk_forward_splits, load_price_series
from tsrl_lite.trainer import train_from_config, train_on_price_splits
from tsrl_lite.utils import dump_json, ensure_dir


@dataclass(slots=True)
class BenchmarkArtifacts:
    root_dir: Path
    summary_path: Path


def aggregate_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {}

    all_keys = sorted({key for metrics in metric_dicts for key in metrics})
    aggregated: dict[str, dict[str, float]] = {}
    for key in all_keys:
        values = np.asarray([metrics[key] for metrics in metric_dicts if key in metrics], dtype=float)
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return aggregated


def aggregate_train_update_metrics(
    summaries: list[dict],
    scope: str,
) -> dict[str, dict[str, float]]:
    metric_dicts: list[dict[str, float]] = []
    for summary in summaries:
        train_update_metrics = summary.get("train_update_metrics")
        if not isinstance(train_update_metrics, dict):
            continue
        scoped_metrics = train_update_metrics.get(scope)
        if not isinstance(scoped_metrics, dict):
            continue
        metric_values: dict[str, float] = {}
        for metric_name, stats in scoped_metrics.items():
            if not isinstance(stats, dict):
                continue
            mean_value = stats.get("mean")
            if not isinstance(mean_value, int | float) or isinstance(mean_value, bool):
                continue
            metric_values[str(metric_name)] = float(mean_value)
        if metric_values:
            metric_dicts.append(metric_values)
    return aggregate_metric_dict(metric_dicts)


def run_benchmark(
    config_path: str | Path,
    seeds: list[int],
    output_dir: str | Path | None = None,
) -> tuple[BenchmarkArtifacts, dict]:
    base_config = load_experiment_config(config_path)
    benchmark_root = ensure_dir(output_dir or Path(base_config.trainer.checkpoint_dir) / "benchmark")

    run_summaries: list[dict] = []
    training_summaries: list[dict] = []
    for seed in seeds:
        run_config = copy.deepcopy(base_config)
        run_config.seed = int(seed)
        run_config.experiment_name = f"{base_config.experiment_name}_seed{seed}"
        run_output_dir = benchmark_root / f"seed_{seed}"
        artifacts, summary = train_from_config(run_config, output_dir=run_output_dir)
        training_summaries.append(summary)
        run_summaries.append(
            {
                "seed": int(seed),
                "output_dir": str(run_output_dir),
                "summary_path": str(artifacts.summary_path),
                "checkpoint_path": str(artifacts.checkpoint_path),
                "best_checkpoint_path": (
                    str(artifacts.best_checkpoint_path) if artifacts.best_checkpoint_path is not None else None
                ),
                "best_validation": summary["best_validation"],
                "evaluation": summary["evaluation"],
                "train_update_metrics": summary.get("train_update_metrics"),
            }
        )

    aggregated = aggregate_metric_dict([summary["evaluation"] for summary in run_summaries])
    benchmark_summary = {
        "experiment": base_config.experiment_name,
        "env": base_config.env.name,
        "encoder": base_config.encoder.name,
        "agent": base_config.agent.name,
        "seeds": [int(seed) for seed in seeds],
        "runs": run_summaries,
        "aggregate_evaluation": aggregated,
        "aggregate_best_validation": aggregate_metric_dict(
            [summary["best_validation"] for summary in run_summaries if summary["best_validation"]]
        ),
        "aggregate_train_update": aggregate_train_update_metrics(training_summaries, scope="overall"),
        "aggregate_train_update_tail": aggregate_train_update_metrics(training_summaries, scope="tail"),
    }

    summary_path = benchmark_root / "benchmark_summary.json"
    dump_json(summary_path, benchmark_summary)
    artifacts = BenchmarkArtifacts(root_dir=benchmark_root, summary_path=summary_path)
    return artifacts, benchmark_summary


def run_walk_forward_benchmark(
    config_path: str | Path,
    n_folds: int,
    train_ratio_start: float = 0.5,
    output_dir: str | Path | None = None,
) -> tuple[BenchmarkArtifacts, dict]:
    base_config = load_experiment_config(config_path)
    benchmark_root = ensure_dir(output_dir or Path(base_config.trainer.checkpoint_dir) / "walk_forward")
    prices = load_price_series(base_config.data, seed=base_config.seed)
    folds = generate_walk_forward_splits(
        prices=prices,
        n_folds=n_folds,
        train_ratio_start=train_ratio_start,
        window_size=base_config.data.window_size,
        min_future_steps=required_future_steps(base_config),
    )

    run_summaries: list[dict] = []
    training_summaries: list[dict] = []
    for fold_payload in folds:
        fold_index = int(fold_payload["fold"])
        run_config = copy.deepcopy(base_config)
        run_config.experiment_name = f"{base_config.experiment_name}_fold{fold_index}"
        run_output_dir = benchmark_root / f"fold_{fold_index}"
        artifacts, summary = train_on_price_splits(
            config=run_config,
            train_prices=fold_payload["train_prices"],
            val_prices=fold_payload["val_prices"],
            eval_prices=fold_payload["eval_prices"],
            output_dir=run_output_dir,
            train_offset=int(fold_payload["train_offset"]),
            val_offset=int(fold_payload["val_offset"]),
            eval_offset=int(fold_payload["eval_offset"]),
        )
        training_summaries.append(summary)
        run_summaries.append(
            {
                "fold": fold_index,
                "output_dir": str(run_output_dir),
                "summary_path": str(artifacts.summary_path),
                "checkpoint_path": str(artifacts.checkpoint_path),
                "best_checkpoint_path": (
                    str(artifacts.best_checkpoint_path) if artifacts.best_checkpoint_path is not None else None
                ),
                "train_price_points": summary["train_price_points"],
                "val_price_points": summary["val_price_points"],
                "eval_price_points": summary["eval_price_points"],
                "train_offset": summary["train_offset"],
                "val_offset": summary["val_offset"],
                "eval_offset": summary["eval_offset"],
                "best_validation": summary["best_validation"],
                "evaluation": summary["evaluation"],
                "train_update_metrics": summary.get("train_update_metrics"),
            }
        )

    benchmark_summary = {
        "experiment": base_config.experiment_name,
        "env": base_config.env.name,
        "encoder": base_config.encoder.name,
        "agent": base_config.agent.name,
        "walk_forward_folds": int(len(run_summaries)),
        "train_ratio_start": float(train_ratio_start),
        "runs": run_summaries,
        "aggregate_evaluation": aggregate_metric_dict([summary["evaluation"] for summary in run_summaries]),
        "aggregate_best_validation": aggregate_metric_dict(
            [summary["best_validation"] for summary in run_summaries if summary["best_validation"]]
        ),
        "aggregate_train_update": aggregate_train_update_metrics(training_summaries, scope="overall"),
        "aggregate_train_update_tail": aggregate_train_update_metrics(training_summaries, scope="tail"),
    }

    summary_path = benchmark_root / "walk_forward_summary.json"
    dump_json(summary_path, benchmark_summary)
    artifacts = BenchmarkArtifacts(root_dir=benchmark_root, summary_path=summary_path)
    return artifacts, benchmark_summary
