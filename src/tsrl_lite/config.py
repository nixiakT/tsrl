from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SyntheticDataConfig:
    steps: int = 2000
    assets: int = 1
    drift: float = 0.0012
    volatility: float = 0.01
    seasonality: float = 0.002
    correlation: float = 0.35


@dataclass(slots=True)
class DataConfig:
    source: str = "synthetic"
    csv_path: str | None = None
    price_column: str = "price"
    price_columns: list[str] = field(default_factory=list)
    timestamp_column: str | None = None
    timestamp_format: str | None = None
    sort_ascending: bool = True
    start_time: str | None = None
    end_time: str | None = None
    train_end_time: str | None = None
    val_end_time: str | None = None
    window_size: int = 32
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)


@dataclass(slots=True)
class EnvConfig:
    name: str = "trading-v0"
    reward_scale: float = 1.0
    episode_horizon: int | None = 256
    random_reset: bool = True
    params: dict[str, object] = field(
        default_factory=lambda: {
            "positions": [-1.0, 0.0, 1.0],
            "trading_cost": 0.0005,
        }
    )


@dataclass(slots=True)
class EncoderConfig:
    name: str = "returns-context-v0"
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class AgentConfig:
    name: str = "linear-actor-critic"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_lr: float = 0.04
    value_lr: float = 0.08
    gradient_clip: float = 5.0
    params: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TrainerConfig:
    episodes: int = 60
    val_episodes: int = 1
    eval_episodes: int = 1
    log_interval: int = 10
    eval_interval: int = 10
    selection_metric: str = "mean_terminal_score"
    selection_mode: str = "max"
    save_best: bool = True
    early_stop_patience: int | None = None
    checkpoint_dir: str = "runs/default"


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str = "default_experiment"
    seed: int = 7
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    data_cfg = DataConfig(
        **{k: v for k, v in payload.get("data", {}).items() if k != "synthetic"},
        synthetic=SyntheticDataConfig(**payload.get("data", {}).get("synthetic", {})),
    )
    env_cfg = EnvConfig(**payload.get("env", {}))
    encoder_cfg = EncoderConfig(**payload.get("encoder", {}))
    agent_cfg = AgentConfig(**payload.get("agent", {}))
    trainer_cfg = TrainerConfig(**payload.get("trainer", {}))

    return ExperimentConfig(
        experiment_name=payload.get("experiment_name", "default_experiment"),
        seed=payload.get("seed", 7),
        data=data_cfg,
        env=env_cfg,
        encoder=encoder_cfg,
        agent=agent_cfg,
        trainer=trainer_cfg,
    )
