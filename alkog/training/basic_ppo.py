"""Minimal PPO trainer for Gymnasium control environments.

This module is intentionally small and self-contained so ALKoG can run an
end-to-end PPO baseline before the full project-specific trainer is built.
"""

from __future__ import annotations

from collections import deque
import importlib
import logging
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from alkog.config.base import ALKoGConfig
from alkog.utils.logging import log_metrics


def _largest_divisor_leq(n: int, upper_bound: int) -> int:
    """Return the largest divisor of n that is <= upper_bound."""
    candidate = min(n, upper_bound)
    while candidate > 1:
        if n % candidate == 0:
            return candidate
        candidate -= 1
    return 1


def _make_env(env_id: str, seed: int, idx: int) -> Any:
    def thunk() -> gym.Env[Any, Any]:
        if ":" in env_id:
            module_name, class_name = env_id.split(":", maxsplit=1)
            module = importlib.import_module(module_name)
            env_class = getattr(module, class_name)
            env = env_class()
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env

    return thunk


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_space: Discrete | Box, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.is_discrete = isinstance(action_space, Discrete)
        if self.is_discrete:
            self.actor = nn.Linear(hidden_dim, int(action_space.n))
        else:
            action_dim = int(np.prod(action_space.shape))
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.critic = nn.Linear(hidden_dim, 1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.trunk(obs)).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs)
        if self.is_discrete:
            logits = self.actor(hidden)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), self.critic(hidden).squeeze(-1)

        mean = self.actor_mean(hidden)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, self.critic(hidden).squeeze(-1)


def train_basic_ppo(
    cfg: ALKoGConfig,
    env_id: str,
    total_timesteps: int,
    seed: int,
    log: logging.Logger,
) -> dict[str, float]:
    """Train a minimal PPO agent on a Gymnasium environment."""
    ppo_cfg = cfg.training.ppo
    num_envs = cfg.training.num_envs
    rollout_length = ppo_cfg.rollout_length
    batch_size = num_envs * rollout_length
    minibatch_size = _largest_divisor_leq(batch_size, ppo_cfg.minibatch_size)

    if minibatch_size != ppo_cfg.minibatch_size:
        log.warning(
            "Adjusted minibatch_size from %s to %s so it divides batch_size=%s.",
            ppo_cfg.minibatch_size,
            minibatch_size,
            batch_size,
        )

    device = torch.device(cfg.resolved_device)
    np.random.seed(seed)
    torch.manual_seed(seed)

    envs = gym.vector.SyncVectorEnv([_make_env(env_id, seed, i) for i in range(num_envs)])

    if not isinstance(envs.single_action_space, (Discrete, Box)):
        raise ValueError(
            "Basic PPO implementation supports only Discrete or Box action spaces. "
            f"Got: {envs.single_action_space}"
        )
    if not isinstance(envs.single_observation_space, Box):
        raise ValueError(
            "Basic PPO implementation expects a Box observation space. "
            f"Got: {envs.single_observation_space}"
        )

    obs_shape = envs.single_observation_space.shape
    if obs_shape is None:
        raise ValueError("Observation space shape must be defined.")

    action_space = envs.single_action_space
    is_discrete_action = isinstance(action_space, Discrete)

    obs_dim = int(np.prod(obs_shape))
    if is_discrete_action:
        action_dim = int(action_space.n)
        action_shape: tuple[int, ...] = ()
        action_low_np = None
        action_high_np = None
    else:
        action_shape = action_space.shape
        if action_shape is None:
            raise ValueError("Continuous action space shape must be defined.")
        action_dim = int(np.prod(action_shape))
        action_low_np = np.asarray(action_space.low, dtype=np.float32)
        action_high_np = np.asarray(action_space.high, dtype=np.float32)

    model = ActorCritic(obs_dim=obs_dim, action_space=action_space).to(device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate, eps=1e-5)

    obs = torch.zeros((rollout_length, num_envs, obs_dim), dtype=torch.float32, device=device)
    if is_discrete_action:
        actions = torch.zeros((rollout_length, num_envs), dtype=torch.long, device=device)
    else:
        actions = torch.zeros((rollout_length, num_envs, action_dim), dtype=torch.float32, device=device)
    logprobs = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((rollout_length, num_envs), dtype=torch.float32, device=device)

    next_obs_np, _ = envs.reset(seed=seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_envs, -1)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    total_updates = max(1, total_timesteps // batch_size)
    global_step = 0
    running_returns = np.zeros(num_envs, dtype=np.float32)
    completed_returns: deque[float] = deque(maxlen=100)

    log.info(
        "Starting basic PPO: env=%s, num_envs=%s, rollout_length=%s, total_updates=%s",
        env_id,
        num_envs,
        rollout_length,
        total_updates,
    )

    for update in range(1, total_updates + 1):
        if ppo_cfg.lr_anneal:
            frac = 1.0 - (update - 1.0) / total_updates
            optimizer.param_groups[0]["lr"] = frac * ppo_cfg.learning_rate

        for step in range(rollout_length):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _entropy, value = model.get_action_and_value(next_obs)

            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            if is_discrete_action:
                env_action = action.cpu().numpy()
            else:
                env_action = action.cpu().numpy().reshape((num_envs, *action_shape))
                env_action = np.clip(env_action, action_low_np, action_high_np)

            next_obs_np, reward_np, terminated, truncated, _infos = envs.step(env_action)
            done_np = np.logical_or(terminated, truncated)

            rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).reshape(
                num_envs,
                -1,
            )
            next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            running_returns += reward_np
            done_indices = np.where(done_np)[0]
            for idx in done_indices:
                completed_returns.append(float(running_returns[idx]))
                running_returns[idx] = 0.0

        with torch.no_grad():
            next_value = model.get_value(next_obs)
            advantages = torch.zeros_like(rewards)
            last_gae_lam = torch.zeros(num_envs, dtype=torch.float32, device=device)
            for t in reversed(range(rollout_length)):
                if t == rollout_length - 1:
                    next_non_terminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + ppo_cfg.gamma * next_values * next_non_terminal - values[t]
                last_gae_lam = (
                    delta
                    + ppo_cfg.gamma * ppo_cfg.gae_lambda * next_non_terminal * last_gae_lam
                )
                advantages[t] = last_gae_lam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        if is_discrete_action:
            b_actions = actions.reshape(-1)
        else:
            b_actions = actions.reshape((-1, action_dim))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)

        pg_loss_value = 0.0
        v_loss_value = 0.0
        entropy_value = 0.0
        approx_kl_value = 0.0

        for _epoch in range(ppo_cfg.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = model.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )

                log_ratio = new_logprob - b_logprobs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl_value = float(((ratio - 1.0) - log_ratio).mean().item())

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std(unbiased=False) + 1e-8
                )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - ppo_cfg.clip_epsilon,
                    1.0 + ppo_cfg.clip_epsilon,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                value_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                value_clipped = b_values[mb_inds] + torch.clamp(
                    new_value - b_values[mb_inds],
                    -ppo_cfg.clip_epsilon,
                    ppo_cfg.clip_epsilon,
                )
                value_loss_clipped = (value_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss + ppo_cfg.value_loss_coef * v_loss - ppo_cfg.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

                pg_loss_value = float(pg_loss.item())
                v_loss_value = float(v_loss.item())
                entropy_value = float(entropy_loss.item())

        mean_return = float(np.mean(completed_returns)) if completed_returns else 0.0
        metrics = {
            "train/global_step": float(global_step),
            "train/update": float(update),
            "train/mean_episode_return_100": mean_return,
            "train/policy_loss": pg_loss_value,
            "train/value_loss": v_loss_value,
            "train/entropy": entropy_value,
            "train/approx_kl": approx_kl_value,
        }
        log_metrics(metrics, step=global_step)
        log.info(
            "update=%s/%s step=%s mean_return_100=%.3f policy_loss=%.4f value_loss=%.4f",
            update,
            total_updates,
            global_step,
            mean_return,
            pg_loss_value,
            v_loss_value,
        )

    envs.close()
    return {
        "global_step": float(global_step),
        "mean_episode_return_100": float(np.mean(completed_returns))
        if completed_returns
        else 0.0,
    }