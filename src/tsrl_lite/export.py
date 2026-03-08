from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tsrl_lite.builders import build_env, required_future_steps
from tsrl_lite.config import load_experiment_config
from tsrl_lite.data import resolve_price_splits
from tsrl_lite.trainer import load_trained_agent
from tsrl_lite.utils import collect_numeric_info, dump_json, ensure_dir


@dataclass(slots=True)
class ExportArtifacts:
    rollout_path: Path
    metadata_path: Path


def export_rollouts(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    split: str = "eval",
    episodes: int = 1,
    greedy: bool = True,
) -> tuple[ExportArtifacts, dict]:
    config = load_experiment_config(config_path)
    splits = resolve_price_splits(
        config=config.data,
        seed=config.seed,
        min_future_steps=required_future_steps(config),
    )

    if split == "train":
        split_prices = splits.train_prices
        index_offset = splits.train_offset
        episode_horizon = config.env.episode_horizon
        random_reset = config.env.random_reset
    elif split == "val":
        split_prices = splits.val_prices
        index_offset = splits.val_offset
        episode_horizon = None
        random_reset = False
    elif split == "eval":
        split_prices = splits.eval_prices
        index_offset = splits.eval_offset
        episode_horizon = None
        random_reset = False
    else:
        raise ValueError("split must be 'train', 'val', or 'eval'")

    env = build_env(
        config,
        prices=split_prices,
        seed=config.seed + 99,
        episode_horizon=episode_horizon,
        random_reset=random_reset,
        index_offset=index_offset,
    )
    agent = load_trained_agent(config=config, checkpoint_path=checkpoint_path, env=env)

    rng = np.random.default_rng(config.seed + 1234)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    values: list[float] = []
    episode_ids: list[int] = []
    global_steps: list[int] = []
    info_traces: dict[str, list[float]] = {}

    for episode_id in range(episodes):
        state = env.reset(random_start=random_reset)
        done = False
        while not done:
            action, _, value, observation = agent.act(state, rng, greedy=greedy)
            next_state, reward, done, info = env.step(action)
            observations.append(observation.copy())
            actions.append(int(action))
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(float(value))
            episode_ids.append(int(episode_id))
            global_steps.append(int(state.context.get("global_step", state.step)))
            collect_numeric_info(info, info_traces)
            state = next_state

    rollout_path = Path(output_path)
    ensure_dir(rollout_path.parent)
    rollout_payload: dict[str, np.ndarray] = {
        "observations": np.asarray(observations, dtype=float),
        "actions": np.asarray(actions, dtype=int),
        "rewards": np.asarray(rewards, dtype=float),
        "dones": np.asarray(dones, dtype=bool),
        "values": np.asarray(values, dtype=float),
        "episode_ids": np.asarray(episode_ids, dtype=int),
        "global_steps": np.asarray(global_steps, dtype=int),
    }
    for key, trace in info_traces.items():
        rollout_payload[f"info_{key}"] = np.asarray(trace, dtype=float)
    np.savez(
        rollout_path,
        **rollout_payload,
    )

    metadata = {
        "experiment": config.experiment_name,
        "split": split,
        "episodes": int(episodes),
        "greedy": bool(greedy),
        "transitions": int(len(actions)),
        "env": config.env.name,
        "encoder": config.encoder.name,
        "agent": config.agent.name,
        "info_keys": sorted(info_traces),
    }
    metadata_path = rollout_path.with_suffix(".json")
    dump_json(metadata_path, metadata)
    artifacts = ExportArtifacts(rollout_path=rollout_path, metadata_path=metadata_path)
    return artifacts, metadata
