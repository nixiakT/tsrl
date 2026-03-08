from __future__ import annotations

from collections.abc import Callable


ENV_REGISTRY: dict[str, type] = {}
ENCODER_REGISTRY: dict[str, type] = {}
AGENT_REGISTRY: dict[str, type] = {}


def register_env(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        ENV_REGISTRY[name] = cls
        return cls

    return decorator


def register_agent(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        AGENT_REGISTRY[name] = cls
        return cls

    return decorator


def register_encoder(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def get_env(name: str) -> type:
    if name not in ENV_REGISTRY:
        raise KeyError(f"unknown env: {name}")
    return ENV_REGISTRY[name]


def get_encoder(name: str) -> type:
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"unknown encoder: {name}")
    return ENCODER_REGISTRY[name]


def get_agent(name: str) -> type:
    if name not in AGENT_REGISTRY:
        raise KeyError(f"unknown agent: {name}")
    return AGENT_REGISTRY[name]


def list_envs() -> list[str]:
    return sorted(ENV_REGISTRY)


def list_encoders() -> list[str]:
    return sorted(ENCODER_REGISTRY)


def list_agents() -> list[str]:
    return sorted(AGENT_REGISTRY)
