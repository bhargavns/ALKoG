"""PPO trainer for continuous soft-category slots."""

import numpy as np
import torch

from lib.ObjectCategories import NUM_OBJECT_CATEGORIES
from lib.SoftCategoryGrounding import RELATION_FEATURE_DIM, SLOT_FEATURE_DIM, SoftCategoryMemory


class SoftCategoryPPOTrainer:
    def __init__(
        self,
        model,
        env,
        perception_fn,
        device="cuda",
        horizon=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        epochs=10,
        minibatch_size=256,
        lr=3e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        perceive_every_steps=25,
    ):
        self.model = model.to(device)
        self.env = env
        self.perception_fn = perception_fn
        self.device = device
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.epochs = int(epochs)
        self.minibatch_size = int(minibatch_size)
        self.ent_coef = float(ent_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.perceive_every_steps = int(perceive_every_steps)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.memory = SoftCategoryMemory(max_age=max(2 * perceive_every_steps, 50))
        self.obs = None
        self.episode_return = 0.0
        self.episode_length = 0

    def _refresh_perception(self):
        detections, relations, _debug = self.perception_fn()
        self.memory.update(detections, relations, self.obs[:2])

    def _reset(self):
        self.obs, _ = self.env.reset()
        self.memory.reset()
        self._refresh_perception()
        self.episode_return = 0.0
        self.episode_length = 0

    def _scene_tensors(self, obs):
        slots, relations = self.memory.features(obs)
        return (
            torch.as_tensor(slots, dtype=torch.float32, device=self.device),
            torch.as_tensor(relations, dtype=torch.float32, device=self.device),
        )

    def collect_and_update(self):
        if self.obs is None:
            self._reset()

        obs_dim = int(self.env.observation_space.shape[0])
        obs_buf = torch.zeros(self.horizon, obs_dim, device=self.device)
        slot_buf = torch.zeros(
            self.horizon, NUM_OBJECT_CATEGORIES, SLOT_FEATURE_DIM, device=self.device
        )
        rel_buf = torch.zeros(self.horizon, RELATION_FEATURE_DIM, device=self.device)
        act_buf = torch.zeros(self.horizon, dtype=torch.long, device=self.device)
        logp_buf = torch.zeros(self.horizon, device=self.device)
        rew_buf = torch.zeros(self.horizon, device=self.device)
        done_buf = torch.zeros(self.horizon, device=self.device)
        val_buf = torch.zeros(self.horizon, device=self.device)

        episode_returns = []
        outcomes = {"food": 0, "water": 0, "poison": 0, "death": 0, "timeout": 0}

        self.model.eval()
        for step in range(self.horizon):
            obs_tensor = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
            slots, relations = self._scene_tensors(self.obs)
            action, logp, value = self.model.act(
                obs_tensor.unsqueeze(0), slots.unsqueeze(0), relations.unsqueeze(0)
            )

            next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            done = bool(terminated or truncated)
            episode_reward = float(reward)
            if truncated and not terminated:
                # A time limit is not a terminal MDP state.  Bootstrap its
                # value into the final reward before the memory is reset.
                with torch.no_grad():
                    next_obs_tensor = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=self.device
                    )
                    next_slots, next_relations = self._scene_tensors(next_obs)
                    reward += self.gamma * float(
                        self.model.value(
                            next_obs_tensor.unsqueeze(0),
                            next_slots.unsqueeze(0),
                            next_relations.unsqueeze(0),
                        ).item()
                    )
            self.episode_return += episode_reward
            self.episode_length += 1

            obs_buf[step] = obs_tensor
            slot_buf[step] = slots
            rel_buf[step] = relations
            act_buf[step] = action.squeeze(0)
            logp_buf[step] = logp.squeeze(0)
            rew_buf[step] = reward
            done_buf[step] = float(done)
            val_buf[step] = value.squeeze(0)

            self.memory.advance()
            if done:
                episode_returns.append(float(self.episode_return))
                if info.get("food_reached"):
                    outcomes["food"] += 1
                elif info.get("water_reached"):
                    outcomes["water"] += 1
                elif info.get("poison_touched"):
                    outcomes["poison"] += 1
                elif info.get("lion_caught"):
                    outcomes["death"] += 1
                else:
                    outcomes["timeout"] += 1
                self._reset()
            else:
                self.obs = next_obs
                if self.perceive_every_steps and self.episode_length % self.perceive_every_steps == 0:
                    self._refresh_perception()

        with torch.no_grad():
            obs_tensor = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
            slots, relations = self._scene_tensors(self.obs)
            last_value = self.model.value(
                obs_tensor.unsqueeze(0), slots.unsqueeze(0), relations.unsqueeze(0)
            ).squeeze(0)

        advantages = torch.zeros_like(rew_buf)
        last_gae = torch.zeros((), device=self.device)
        for step in reversed(range(self.horizon)):
            next_value = last_value if step == self.horizon - 1 else val_buf[step + 1]
            nonterminal = 1.0 - done_buf[step]
            delta = rew_buf[step] + self.gamma * next_value * nonterminal - val_buf[step]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + val_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.model.train()
        losses = {"policy": [], "value": [], "entropy": [], "kl": [], "clip": []}
        for _ in range(self.epochs):
            for indices in torch.randperm(self.horizon, device=self.device).split(
                self.minibatch_size
            ):
                logp, entropy, value = self.model.evaluate(
                    obs_buf[indices], slot_buf[indices], rel_buf[indices], act_buf[indices]
                )
                ratio = torch.exp(logp - logp_buf[indices])
                clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(
                    ratio * advantages[indices], clipped * advantages[indices]
                ).mean()
                value_loss = ((value - returns[indices]) ** 2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses["policy"].append(float(policy_loss.detach()))
                losses["value"].append(float(value_loss.detach()))
                losses["entropy"].append(float(entropy_mean.detach()))
                losses["kl"].append(float((logp_buf[indices] - logp).mean().detach()))
                losses["clip"].append(
                    float(((ratio - 1.0).abs() > self.clip_ratio).float().mean().detach())
                )

        episode_count = max(1, len(episode_returns))
        return {
            "episodes": len(episode_returns),
            "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            **{f"{name}_rate": count / episode_count for name, count in outcomes.items()},
            "policy_loss": float(np.mean(losses["policy"])),
            "value_loss": float(np.mean(losses["value"])),
            "entropy": float(np.mean(losses["entropy"])),
            "approx_kl": float(np.mean(losses["kl"])),
            "clip_fraction": float(np.mean(losses["clip"])),
        }
