from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import tsrl_lite.algorithms  # noqa: F401
from tsrl_lite.algorithms import ActorCriticAgent, EpisodeBatch
from tsrl_lite.builders import build_agent, build_encoder, build_env, required_future_steps
from tsrl_lite.config import ExperimentConfig, load_experiment_config
from tsrl_lite.data import resolve_price_splits
from tsrl_lite.evaluate import evaluate_agent
from tsrl_lite.utils import (
    collect_numeric_info,
    compute_trading_episode_metrics,
    dump_json,
    dump_records_csv,
    ensure_dir,
)


@dataclass(slots=True)
class TrainingArtifacts:
    config_path: Path
    checkpoint_path: Path
    best_checkpoint_path: Path | None
    summary_path: Path
    history_path: Path
    history_csv_path: Path


def terminal_score(info: dict[str, float], total_reward: float) -> float:
    if "equity" in info:
        return float(info["equity"])
    if "accuracy" in info:
        return float(info["accuracy"])
    return float(total_reward)


def summarize_numeric_metrics(
    records: list[dict],
    metric_keys: set[str],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(metric_keys):
        values = [float(record[key]) for record in records if key in record]
        if not values:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "last": float(values[-1]),
        }
    return summary


def format_episode_log(record: dict[str, float]) -> str:
    parts = [
        f"reward={record['episode_reward']:.4f}",
        f"score={record['terminal_score']:.4f}",
        f"value_loss={record['value_loss']:.4f}",
    ]
    optional_metrics = (
        ("approx_kl", "kl"),
        ("clip_fraction", "clip"),
        ("explained_variance", "ev"),
        ("update_steps", "updates"),
    )
    for metric_key, label in optional_metrics:
        if metric_key in record:
            parts.append(f"{label}={record[metric_key]:.4f}")
    if "early_stop_triggered" in record:
        parts.append(f"kl_stop={int(record['early_stop_triggered'] >= 0.5)}")
    return " ".join(parts)


def collect_episode(
    env,
    agent,
    seed: int,
    random_start: bool,
    greedy: bool = False,
) -> tuple[EpisodeBatch, dict[str, float]]:
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    values: list[float] = []
    action_probs: list[np.ndarray] = []

    state = env.reset(random_start=random_start)
    done = False
    total_reward = 0.0
    last_info: dict[str, float] = {}
    info_metrics: dict[str, list[float]] = {}

    while not done:
        action, probs, value, observation = agent.act(state, rng, greedy=greedy)
        next_state, reward, done, info = env.step(action)
        observations.append(observation.copy())
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        action_probs.append(probs.copy())
        state = next_state
        total_reward += reward
        last_info = info
        collect_numeric_info(info, info_metrics)

    batch = EpisodeBatch(
        observations=np.asarray(observations, dtype=float),
        actions=np.asarray(actions, dtype=int),
        rewards=np.asarray(rewards, dtype=float),
        dones=np.asarray(dones, dtype=bool),
        values=np.asarray(values, dtype=float),
        action_probs=np.asarray(action_probs, dtype=float),
    )
    stats = {
        "episode_reward": float(total_reward),
        "episode_length": float(len(rewards)),
        "terminal_score": terminal_score(last_info, total_reward),
    }
    reward_series = np.asarray(rewards, dtype=float)
    if "equity" in info_metrics:
        stats.update(compute_trading_episode_metrics(reward_series, info_metrics))
    if "equity" in last_info:
        stats["final_equity"] = float(last_info["equity"])
    for key, values in info_metrics.items():
        stats[f"mean_info_{key}"] = float(np.mean(values))
    return batch, stats


def train_experiment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[TrainingArtifacts, dict]:
    config = load_experiment_config(config_path)
    return train_from_config(config, output_dir=output_dir)


def train_from_config(
    config: ExperimentConfig,
    output_dir: str | Path | None = None,
) -> tuple[TrainingArtifacts, dict]:
    splits = resolve_price_splits(
        config=config.data,
        seed=config.seed,
        min_future_steps=required_future_steps(config),
    )
    return train_on_price_splits(
        config=config,
        train_prices=splits.train_prices,
        val_prices=splits.val_prices,
        eval_prices=splits.eval_prices,
        output_dir=output_dir,
        train_offset=splits.train_offset,
        val_offset=splits.val_offset,
        eval_offset=splits.eval_offset,
        split_mode=splits.split_mode,
        data_start_time=splits.data_start_time,
        data_end_time=splits.data_end_time,
        train_end_time=splits.train_end_time,
        val_end_time=splits.val_end_time,
    )


def train_on_price_splits(
    config: ExperimentConfig,
    train_prices: np.ndarray,
    val_prices: np.ndarray,
    eval_prices: np.ndarray,
    output_dir: str | Path | None = None,
    train_offset: int = 0,
    val_offset: int = 0,
    eval_offset: int = 0,
    split_mode: str = "ratio",
    data_start_time: str | None = None,
    data_end_time: str | None = None,
    train_end_time: str | None = None,
    val_end_time: str | None = None,
) -> tuple[TrainingArtifacts, dict]:
    checkpoint_root = Path(output_dir or config.trainer.checkpoint_dir)
    ensure_dir(checkpoint_root)

    train_env = build_env(
        config,
        prices=train_prices,
        seed=config.seed,
        index_offset=train_offset,
    )
    val_env = build_env(
        config,
        prices=val_prices,
        seed=config.seed + 1,
        episode_horizon=None,
        random_reset=False,
        index_offset=val_offset,
    )
    eval_env = build_env(
        config,
        prices=eval_prices,
        seed=config.seed + 2,
        episode_horizon=None,
        random_reset=False,
        index_offset=eval_offset,
    )

    encoder = build_encoder(config, train_env)
    agent = build_agent(config, train_env, encoder=encoder, seed=config.seed)

    history: list[dict] = []
    validation_history: list[dict] = []
    update_metric_keys: set[str] = set()
    checkpoint_suffix = getattr(agent, "checkpoint_suffix", ".npz")
    checkpoint_path = checkpoint_root / f"agent_checkpoint{checkpoint_suffix}"
    best_checkpoint_path = checkpoint_root / f"best_checkpoint{checkpoint_suffix}"
    selection_mode = config.trainer.selection_mode.lower()
    if selection_mode not in {"max", "min"}:
        raise ValueError("trainer.selection_mode must be 'max' or 'min'")

    def is_better(candidate: float, current: float | None) -> bool:
        if current is None:
            return True
        if selection_mode == "max":
            return candidate > current
        return candidate < current

    best_metric: float | None = None
    best_validation: dict[str, float] | None = None
    epochs_since_improvement = 0

    for episode in range(1, config.trainer.episodes + 1):
        batch, rollout_stats = collect_episode(
            train_env,
            agent,
            seed=config.seed + episode,
            random_start=config.env.random_reset,
        )
        update_stats = agent.update(batch)
        update_metric_keys.update(update_stats.keys())
        record = {
            "episode": episode,
            **rollout_stats,
            **update_stats,
        }
        history.append(record)

        if episode % config.trainer.eval_interval == 0 or episode == config.trainer.episodes:
            validation = evaluate_agent(
                agent=agent,
                env=val_env,
                episodes=config.trainer.val_episodes,
                seed=config.seed + 2000 + episode,
                random_start=False,
            )
            validation_record = {"episode": episode, **validation}
            validation_history.append(validation_record)

            metric_name = config.trainer.selection_metric
            if metric_name not in validation:
                raise KeyError(f"selection metric '{metric_name}' missing from validation metrics")
            candidate_metric = float(validation[metric_name])
            record["validation_metric"] = candidate_metric

            if is_better(candidate_metric, best_metric):
                best_metric = candidate_metric
                best_validation = validation
                epochs_since_improvement = 0
                if config.trainer.save_best:
                    agent.save(best_checkpoint_path)
            else:
                epochs_since_improvement += 1

            print(
                f"[validation {episode:03d}] "
                f"{metric_name}={candidate_metric:.4f} "
                f"best={best_metric:.4f}"
            )

        if episode % config.trainer.log_interval == 0 or episode == config.trainer.episodes:
            print(f"[episode {episode:03d}] {format_episode_log(record)}")

        if (
            config.trainer.early_stop_patience is not None
            and epochs_since_improvement >= config.trainer.early_stop_patience
        ):
            print(f"[early-stop] patience reached at episode {episode:03d}")
            break

    config_copy_path = checkpoint_root / "resolved_config.json"
    history_path = checkpoint_root / "history.json"
    history_csv_path = checkpoint_root / "history.csv"
    summary_path = checkpoint_root / "summary.json"

    agent.save(checkpoint_path)
    dump_json(config_copy_path, config.to_dict())
    dump_json(history_path, {"history": history, "validation_history": validation_history})
    dump_records_csv(history_csv_path, history)

    evaluation_last = evaluate_agent(
        agent=agent,
        env=eval_env,
        episodes=config.trainer.eval_episodes,
        seed=config.seed + 1000,
        random_start=False,
    )
    evaluation_best = None
    if config.trainer.save_best and best_checkpoint_path.exists():
        best_agent = load_trained_agent(config=config, checkpoint_path=best_checkpoint_path, env=eval_env)
        evaluation_best = evaluate_agent(
            agent=best_agent,
            env=eval_env,
            episodes=config.trainer.eval_episodes,
            seed=config.seed + 3000,
            random_start=False,
        )

    preferred_evaluation = evaluation_best or evaluation_last
    train_update_metrics = {
        "episodes_tracked": int(len(history)),
        "tail_window": int(min(5, len(history))),
        "overall": summarize_numeric_metrics(history, update_metric_keys),
        "tail": summarize_numeric_metrics(history[-5:], update_metric_keys),
    }
    summary = {
        "experiment": config.experiment_name,
        "env": config.env.name,
        "encoder": config.encoder.name,
        "agent": config.agent.name,
        "train_history_tail": history[-5:],
        "validation_history_tail": validation_history[-5:],
        "best_validation": best_validation,
        "best_selection_metric": best_metric,
        "train_update_metrics": train_update_metrics,
        "evaluation": preferred_evaluation,
        "evaluation_last": evaluation_last,
        "evaluation_best": evaluation_best,
        "train_price_points": int(len(train_prices)),
        "val_price_points": int(len(val_prices)),
        "eval_price_points": int(len(eval_prices)),
        "train_offset": int(train_offset),
        "val_offset": int(val_offset),
        "eval_offset": int(eval_offset),
        "data_source": config.data.source,
        "data_split_mode": split_mode,
        "data_start_time": data_start_time,
        "data_end_time": data_end_time,
        "train_end_time": train_end_time,
        "val_end_time": val_end_time,
    }
    dump_json(summary_path, summary)

    artifacts = TrainingArtifacts(
        config_path=config_copy_path,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path if config.trainer.save_best else None,
        summary_path=summary_path,
        history_path=history_path,
        history_csv_path=history_csv_path,
    )
    return artifacts, summary


def load_trained_agent(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    env,
):
    encoder = build_encoder(config, env)
    agent = build_agent(config, env, encoder=encoder, seed=config.seed)
    agent.load(str(checkpoint_path))
    return agent
