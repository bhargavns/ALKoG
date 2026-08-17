import numpy as np
import torch

from lib.Grounding import AnchorMemory, deltas_from_anchors


class PPOTrainer:
    """Minimal PPO (clip objective, GAE) over a TripleActorCritic.

    The optimizer covers actor, critic, AND the symbol table (node/relation
    symbols, positional embeddings, distance projection), so the KG's 4-dim
    symbols are shaped by the same gradients that train the policy.
    `triple_fn` is called at every episode reset to re-run perception and
    produce the ([K,3] triple indices, [K,2,2] node offsets) for the new
    episode. Each pass's offsets are anchored as world-frame positions and
    re-derived every step from the agent's current position, rotated into
    its egocentric frame, so the policy's distance input stays current
    between perception passes and aligns with the egocentric action set.
    Anchors persist across passes within an episode (AnchorMemory), so a
    concept one pass fails to detect keeps its last-known anchor instead of
    vanishing from the input. Actions are discrete (Categorical policy).
    """

    def __init__(
        self,
        model,
        env,
        triple_fn,
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
        perceive_every_steps=0,  # 0 = re-perceive only at episode reset
        record_trajectories=False,  # keep per-episode xy paths for diagnostics
        normalize_value_loss=False,  # standardize value loss to the advantage scale
    ):
        self.model = model.to(device)
        self.env = env
        self.triple_fn = triple_fn
        self.device = device
        self.horizon = horizon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.perceive_every_steps = perceive_every_steps
        self.record_trajectories = record_trajectories
        self.normalize_value_loss = normalize_value_loss
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self._obs = None
        self._triples = None
        self._anchors = None
        self._delta_mask = None
        self._anchor_mem = AnchorMemory()
        self._ep_return = 0.0
        self._ep_len = 0
        self._traj = None
        self.trajectories = []  # completed-episode paths; the caller drains this
        self.last_actions = None  # [horizon] sampled action ids of last iteration

    def _reset_env(self):
        obs, _ = self.env.reset()
        self._obs = obs
        self._anchor_mem.reset()  # new layout: remembered anchors are invalid
        self._refresh_perception()
        self._ep_return = 0.0
        self._ep_len = 0
        if self.record_trajectories:
            # layout is fixed for the episode; capture it once at reset
            self._traj = {
                "positions": [self.env.data.qpos[0:2].copy().tolist()],
                "lion": self.env.model.body("lion").pos[:2].copy().tolist(),
                "food": self.env.model.body("food").pos[:2].copy().tolist(),
                "cage": self.env.model.body("cage").pos[:2].copy().tolist(),
                "lion_caged": bool(self.env.lion_caged),
            }

    def _refresh_perception(self):
        # offsets are measured from where the agent stands right now; anchor
        # them in world frame so _current_deltas can track movement until the
        # next pass, and merge with remembered anchors so concepts this pass
        # missed stay in the input
        triples, deltas = self.triple_fn()
        self._triples, self._anchors, self._delta_mask = self._anchor_mem.merge(
            triples, deltas, self._obs[:2]
        )

    def _current_deltas(self, obs):
        return deltas_from_anchors(self._anchors, self._delta_mask, obs)

    def _finish_trajectory(self, info, truncated):
        if self._traj is None:
            return
        if info.get("food_reached", False):
            outcome = "food"
        elif info.get("lion_caught", False):
            outcome = "death"
        else:
            outcome = "timeout" if truncated else "end"
        self._traj["outcome"] = outcome
        self._traj["episode_return"] = float(self._ep_return)
        self.trajectories.append(self._traj)
        self._traj = None

    def collect_and_update(self):
        """One PPO iteration: collect `horizon` steps, then update. Returns stats."""
        if self._obs is None:
            self._reset_env()

        obs_dim = self.env.observation_space.shape[0]
        n_actions = int(self.env.action_space.n)
        k = self._triples.shape[0]

        obs_buf = torch.zeros(self.horizon, obs_dim, device=self.device)
        triple_buf = torch.zeros(self.horizon, k, 3, dtype=torch.long, device=self.device)
        delta_buf = torch.zeros(self.horizon, k, 2, 2, device=self.device)
        act_buf = torch.zeros(self.horizon, dtype=torch.long, device=self.device)
        logp_buf = torch.zeros(self.horizon, device=self.device)
        rew_buf = torch.zeros(self.horizon, device=self.device)
        done_buf = torch.zeros(self.horizon, device=self.device)
        val_buf = torch.zeros(self.horizon, device=self.device)

        ep_returns, ep_lengths, foods, deaths, timeouts = [], [], 0, 0, 0
        caged_eps, caged_foods, loose_eps, loose_foods, loose_deaths = 0, 0, 0, 0, 0
        # conflict layouts (food co-located with the lion): the symbol-forcing
        # cases. near_caged should be grabbed; near_loose should be avoided.
        near_caged_eps, near_caged_foods = 0, 0
        near_loose_eps, near_loose_foods, near_loose_deaths = 0, 0, 0

        self.model.eval()
        for t in range(self.horizon):
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            triples_t = self._triples.to(self.device)
            deltas_t = self._current_deltas(self._obs).to(self.device)
            action, logp, value = self.model.act(
                obs_t.unsqueeze(0), triples_t.unsqueeze(0), deltas_t.unsqueeze(0)
            )

            next_obs, reward, terminated, truncated, info = self.env.step(
                int(action.item())
            )
            self._ep_return += reward
            self._ep_len += 1
            if self._traj is not None:
                self._traj["positions"].append(self.env.data.qpos[0:2].copy().tolist())

            obs_buf[t] = obs_t
            triple_buf[t] = triples_t
            delta_buf[t] = deltas_t
            act_buf[t] = action.squeeze(0)
            logp_buf[t] = logp.squeeze(0)
            val_buf[t] = value.squeeze(0)

            if truncated and not terminated:
                # bootstrap the cut-off return with V(s') so timeouts aren't
                # treated as real terminal states
                with torch.no_grad():
                    next_obs_t = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=self.device
                    )
                    reward += self.gamma * float(
                        self.model.value(
                            next_obs_t.unsqueeze(0),
                            triples_t.unsqueeze(0),
                            self._current_deltas(next_obs).to(self.device).unsqueeze(0),
                        ).squeeze(0)
                    )

            rew_buf[t] = reward
            done_buf[t] = float(terminated or truncated)

            if terminated or truncated:
                ep_returns.append(self._ep_return)
                ep_lengths.append(self._ep_len)
                foods += int(info.get("food_reached", False))
                deaths += int(info.get("lion_caught", False))
                timeouts += int(truncated and not terminated)
                caged = info.get("lion_caged", False)
                if caged:
                    caged_eps += 1
                    caged_foods += int(info.get("food_reached", False))
                else:
                    loose_eps += 1
                    loose_foods += int(info.get("food_reached", False))
                    loose_deaths += int(info.get("lion_caught", False))
                if info.get("food_near_lion", False):
                    if caged:
                        near_caged_eps += 1
                        near_caged_foods += int(info.get("food_reached", False))
                    else:
                        near_loose_eps += 1
                        near_loose_foods += int(info.get("food_reached", False))
                        near_loose_deaths += int(info.get("lion_caught", False))
                self._finish_trajectory(info, truncated)
                self._reset_env()
            else:
                self._obs = next_obs
                if (
                    self.perceive_every_steps
                    and self._ep_len % self.perceive_every_steps == 0
                ):
                    # refresh the triples from the agent's current vantage --
                    # relations invisible from spawn become visible up close
                    self._refresh_perception()

        # bootstrap value for the state the buffer stopped in
        with torch.no_grad():
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=self.device)
            last_val = self.model.value(
                obs_t.unsqueeze(0),
                self._triples.to(self.device).unsqueeze(0),
                self._current_deltas(self._obs).to(self.device).unsqueeze(0),
            ).squeeze(0)

        adv_buf = torch.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(self.horizon)):
            next_val = last_val if t == self.horizon - 1 else val_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + self.gamma * next_val * nonterminal - val_buf[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        # Standardize the value-loss residual by the return std so the critic
        # gradient into the shared symbol trunk is on the same unit scale as
        # the (already unit-std) advantages -- otherwise v_loss ~ Var(returns)
        # and its gradient swamps the actor's by 75-330x (grad_attribution
        # diagnostic). Critic predictions stay in raw return space (GAE and
        # bootstrapping are unaffected); only the loss gradient is rescaled.
        ret_scale = (ret_buf.std() + 1e-8) if self.normalize_value_loss else 1.0

        self.model.train()
        pi_losses, v_losses, entropies, approx_kls, clip_fracs = [], [], [], [], []
        for _ in range(self.epochs):
            for idx in torch.randperm(self.horizon, device=self.device).split(
                self.minibatch_size
            ):
                logp, entropy, value = self.model.evaluate(
                    obs_buf[idx], triple_buf[idx], delta_buf[idx], act_buf[idx]
                )
                ratio = torch.exp(logp - logp_buf[idx])
                clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                pi_loss = -torch.min(ratio * adv_buf[idx], clipped * adv_buf[idx]).mean()
                v_loss = (((value - ret_buf[idx]) / ret_scale) ** 2).mean()
                ent = entropy.mean()

                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pi_losses.append(float(pi_loss.detach()))
                v_losses.append(float(v_loss.detach()))
                entropies.append(float(ent.detach()))
                with torch.no_grad():
                    approx_kls.append(float((logp_buf[idx] - logp).mean()))
                    clip_fracs.append(
                        float(((ratio - 1.0).abs() > self.clip_ratio).float().mean())
                    )

        with torch.no_grad():
            ret_var = float(ret_buf.var())
            explained_var = (
                1.0 - float((ret_buf - val_buf).var()) / ret_var if ret_var > 1e-8 else 0.0
            )
        actions_np = act_buf.cpu().numpy()
        self.last_actions = actions_np
        action_fracs = np.bincount(actions_np, minlength=n_actions) / self.horizon

        n_eps = max(1, len(ep_returns))
        return {
            "episodes": len(ep_returns),
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "food_rate": foods / n_eps,
            "death_rate": deaths / n_eps,
            "timeout_rate": timeouts / n_eps,
            "caged_food_rate": caged_foods / max(1, caged_eps),
            "loose_food_rate": loose_foods / max(1, loose_eps),
            "loose_death_rate": loose_deaths / max(1, loose_eps),
            "near_caged_food_rate": near_caged_foods / max(1, near_caged_eps),
            "near_loose_food_rate": near_loose_foods / max(1, near_loose_eps),
            "near_loose_death_rate": near_loose_deaths / max(1, near_loose_eps),
            "pi_loss": float(np.mean(pi_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_frac": float(np.mean(clip_fracs)),
            "explained_var": explained_var,
            "action_fracs": action_fracs.tolist(),
        }
