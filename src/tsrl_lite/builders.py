from __future__ import annotations

from typing import Any

import tsrl_lite.encoders  # noqa: F401
import tsrl_lite.envs  # noqa: F401
from tsrl_lite.config import ExperimentConfig
from tsrl_lite.registry import get_agent, get_encoder, get_env


def required_future_steps(config: ExperimentConfig) -> int:
    env_cls = get_env(config.env.name)
    return int(env_cls.required_future_steps(config.env.params))


def build_env(
    config: ExperimentConfig,
    prices,
    seed: int,
    episode_horizon: int | None = None,
    random_reset: bool | None = None,
    index_offset: int = 0,
):
    env_cls = get_env(config.env.name)
    return env_cls(
        prices=prices,
        window_size=config.data.window_size,
        reward_scale=config.env.reward_scale,
        episode_horizon=config.env.episode_horizon if episode_horizon is None else episode_horizon,
        random_reset=config.env.random_reset if random_reset is None else random_reset,
        seed=seed,
        index_offset=index_offset,
        **config.env.params,
    )


def build_encoder(config: ExperimentConfig, env) -> Any:
    encoder_cls = get_encoder(config.encoder.name)
    return encoder_cls(
        window_size=config.data.window_size,
        agent_feature_dim=env.agent_feature_dim,
        window_feature_dim=getattr(env, "window_feature_dim", 1),
        **config.encoder.params,
    )


def build_agent(config: ExperimentConfig, env, encoder, seed: int | None = None):
    agent_cls = get_agent(config.agent.name)
    return agent_cls(
        encoder=encoder,
        action_dim=env.action_dim,
        gamma=config.agent.gamma,
        gae_lambda=config.agent.gae_lambda,
        policy_lr=config.agent.policy_lr,
        value_lr=config.agent.value_lr,
        gradient_clip=config.agent.gradient_clip,
        seed=config.seed if seed is None else seed,
        **config.agent.params,
    )
