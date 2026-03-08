from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    torch = None
    nn = None
    Categorical = None

from tsrl_lite.algorithms.common import EpisodeBatch, compute_gae


def compute_explained_variance(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    target_var = float(np.var(targets))
    if target_var <= 1e-8:
        return 0.0
    residual_var = float(np.var(targets - predictions))
    return float(1.0 - (residual_var / target_var))


def prepare_torch_ppo_batch(
    batch: EpisodeBatch,
    *,
    gamma: float,
    gae_lambda: float,
    device,
    normalize_advantages: bool,
) -> dict[str, object]:
    advantages, returns = compute_gae(
        rewards=batch.rewards,
        values=batch.values,
        dones=batch.dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_action_probs = batch.action_probs[np.arange(len(batch.actions)), batch.actions]
    return {
        "observations": torch.as_tensor(batch.observations, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(batch.actions, dtype=torch.long, device=device),
        "advantages": advantages,
        "advantages_tensor": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "returns": returns,
        "returns_tensor": torch.as_tensor(returns, dtype=torch.float32, device=device),
        "old_values_tensor": torch.as_tensor(batch.values, dtype=torch.float32, device=device),
        "old_log_probs": torch.log(torch.as_tensor(old_action_probs, dtype=torch.float32, device=device) + 1e-8),
    }


def run_torch_ppo_update(
    *,
    network,
    optimizer,
    prepared_batch: dict[str, object],
    update_epochs: int,
    gradient_clip: float,
    clip_epsilon: float,
    entropy_coef: float,
    value_coef: float,
    mini_batch_size: int | None,
    shuffle_minibatches: bool,
    value_clip_epsilon: float | None,
    target_kl: float | None,
) -> dict[str, float]:
    observations = prepared_batch["observations"]
    actions = prepared_batch["actions"]
    advantages = prepared_batch["advantages"]
    advantages_tensor = prepared_batch["advantages_tensor"]
    returns = prepared_batch["returns"]
    returns_tensor = prepared_batch["returns_tensor"]
    old_values_tensor = prepared_batch["old_values_tensor"]
    old_log_probs = prepared_batch["old_log_probs"]

    batch_size = int(actions.shape[0])
    resolved_mini_batch_size = batch_size if mini_batch_size is None else max(1, min(int(mini_batch_size), batch_size))
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "entropy": 0.0,
    }
    update_steps = 0
    early_stop_triggered = False

    network.train()
    for epoch_index in range(max(1, int(update_epochs))):
        if shuffle_minibatches:
            order = torch.randperm(batch_size, device=actions.device)
        else:
            order = torch.arange(batch_size, device=actions.device)

        for start_index in range(0, batch_size, resolved_mini_batch_size):
            batch_index = order[start_index : start_index + resolved_mini_batch_size]
            logits, values = network(observations[batch_index])
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions[batch_index])
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs[batch_index])
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_loss = -torch.min(
                ratio * advantages_tensor[batch_index],
                clipped_ratio * advantages_tensor[batch_index],
            ).mean()

            if value_clip_epsilon is not None:
                clipped_values = old_values_tensor[batch_index] + torch.clamp(
                    values - old_values_tensor[batch_index],
                    -float(value_clip_epsilon),
                    float(value_clip_epsilon),
                )
                unclipped_value_loss = torch.square(values - returns_tensor[batch_index])
                clipped_value_loss = torch.square(clipped_values - returns_tensor[batch_index])
                value_loss = 0.5 * torch.maximum(unclipped_value_loss, clipped_value_loss).mean()
            else:
                value_loss = nn.functional.mse_loss(values, returns_tensor[batch_index])

            loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)
            optimizer.step()

            clip_fraction = ((ratio - clipped_ratio).abs() > 1e-8).float().mean()
            approx_kl = (old_log_probs[batch_index] - log_probs).mean()

            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["approx_kl"] += float(approx_kl.item())
            metrics["clip_fraction"] += float(clip_fraction.item())
            metrics["entropy"] += float(entropy.item())
            update_steps += 1

            if target_kl is not None and float(approx_kl.item()) > float(target_kl):
                early_stop_triggered = True
                break

        if early_stop_triggered:
            break

    network.eval()
    with torch.no_grad():
        _, final_values = network(observations)
    explained_variance = compute_explained_variance(
        predictions=final_values.detach().cpu().numpy(),
        targets=np.asarray(returns, dtype=float),
    )

    scale = 1.0 / max(update_steps, 1)
    return {
        "policy_loss": metrics["policy_loss"] * scale,
        "value_loss": metrics["value_loss"] * scale,
        "approx_kl": metrics["approx_kl"] * scale,
        "clip_fraction": metrics["clip_fraction"] * scale,
        "entropy": metrics["entropy"] * scale,
        "advantage_mean": float(np.mean(advantages)),
        "explained_variance": float(explained_variance),
        "update_steps": float(update_steps),
        "early_stop_triggered": float(1.0 if early_stop_triggered else 0.0),
    }
