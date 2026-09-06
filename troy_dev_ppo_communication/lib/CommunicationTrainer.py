"""PPO over two coupled agents: a transmitter that emits symbols and a blind
receiver that acts on them.

Each step, in this order:

    1. the transmitter reads its perceptual input and emits one symbol
    2. the symbol is appended to the receiver's sliding context
    3. the receiver reads the context and picks a move
    4. the env steps the RECEIVER; the reward is the task outcome

Both agents are credited the SAME reward. Neither has a private objective, so
the only way the transmitter can score is by emitting symbols that steer the
receiver, and the only way the receiver can score is by reading them. Each has
its own actor, critic, optimizer, buffers, and GAE; nothing but the reward and
the symbol stream connects them, and no gradient flows across the channel (the
emitted symbol is a sampled discrete index, not a differentiable path).

Both models share a trunk between actor and critic -- the transmitter its
SymbolTable, the receiver its transformer encoder -- so both need the same
value-loss normalization that RESULTS.md section 3 found necessary: with raw
value targets the critic's gradient through a shared trunk swamps the actor's
by 75-330x.

The transmitter gets a dedicated final slot for the receiver, refreshed from
the env every step (see CommunicationKGWorldEnv.receiver_anchor). The receiver
never appears in a relation -- it is position-only -- because relations come
from 2D box overlap on a SAM pass that is frozen after episode start, which for
a moving object would be not merely imprecise but uncorrelated.

Perception cadence: SAM runs `perceive_passes` times at the start of an episode
-- by default ONCE, at reset, feeding the first emission -- and never again. The
transmitter cannot move and the lion/cage/food do not move, so their anchors
stay exact for the whole episode; re-running SAM later would only re-measure an
unchanged scene from an unchanged viewpoint. Measured on 30 episodes: a second
consecutive pass added a static object 0 times (mean known 2.47/3 either way),
so extra passes buy box jitter rather than information, at ~2 s per episode.
Raise perceive_passes only if the transmitter ever becomes mobile.
"""

import numpy as np
import torch

from lib.Grounding import (
    TRIPLE_PAD,
    AnchorMemory,
    assemble_triples_geo,
    deltas_from_anchors,
)
from lib.ReceiverPolicy import CONTEXT_LEN, build_context


def strip_concept(per_frame, node_id):
    """Drop one concept, and every relation touching it, from a perception pass.

    The receiver is fed to the transmitter as an oracle position instead, so it
    must not also arrive through SAM -- otherwise it would occupy a second slot
    carrying a stale anchor, and could form relations that are stale too.
    """
    out = []
    for boxes, sims, deltas, relations in per_frame:
        out.append(
            (
                {c: b for c, b in boxes.items() if c != node_id},
                {c: s for c, s in sims.items() if c != node_id},
                {c: d for c, d in deltas.items() if c != node_id},
                [r for r in relations if r[0] != node_id and r[2] != node_id],
            )
        )
    return out


def _merge_with_receiver_slot(anchor_mem, triples, deltas, agent_xy, receiver_node):
    """AnchorMemory.merge, plus a reserved final slot holding the receiver.

    The receiver enters as a lone (node, PAD, PAD) entry in slot k, so the
    policy sees k_triples SAM-derived entries plus one position-only receiver
    entry. Its anchor is a placeholder here -- TransmitterInput.deltas
    overwrites anchors[-1, 0] from the env every step.
    """
    tri, anchors, mask = anchor_mem.merge(triples, deltas, agent_xy)
    k = tri.shape[0]
    out_tri = torch.full((k + 1, 3), TRIPLE_PAD, dtype=torch.long)
    out_tri[:k] = tri
    out_tri[k, 0] = receiver_node
    out_anchors = torch.zeros(k + 1, 2, 2, dtype=anchors.dtype)
    out_anchors[:k] = anchors
    out_mask = torch.zeros(k + 1, 2, 1, dtype=mask.dtype)
    out_mask[:k] = mask
    out_mask[k, 0] = 1.0
    return out_tri, out_anchors, out_mask


class TransmitterInput:
    """The one implementation of the transmitter's per-step policy input.

    Two things drive the transmitter -- the trainer and the video recorder --
    and if they build its input differently the video depicts a state the
    policy never saw. So every step that could drift lives here and nowhere
    else: stripping the receiver out of perception, assembling triples at the
    right k, reserving the receiver slot, and refreshing that slot's anchor
    from the env each step.

    Build these through a single zero-argument factory shared by both callers
    (see scripts/train_communication.py) so there is also only one place the
    constructor arguments are chosen.
    """

    def __init__(self, kg, per_frame_fn, receiver_node, k_triples):
        self.kg = kg
        self.per_frame_fn = per_frame_fn
        self.receiver_node = receiver_node
        self.k_triples = k_triples
        self.mem = AnchorMemory()
        self.triples = None
        self.anchors = None
        self.mask = None
        self.boxes = None
        self.relations = None

    def reset(self):
        """New episode: remembered anchors describe the previous layout."""
        self.mem.reset()

    def refresh(self, obs):
        """Run a perception pass and rebuild the slots.

        Returns (boxes, relations) per camera for the video overlay; the
        trainer ignores them. The receiver is stripped HERE, so no caller can
        forget to -- it reaches the policy only through its reserved slot.
        """
        per_frame = strip_concept(self.per_frame_fn(), self.receiver_node)
        self.boxes = [pf[0] for pf in per_frame]
        self.relations = [pf[3] for pf in per_frame]
        triples, deltas = assemble_triples_geo(self.kg, per_frame, k=self.k_triples)
        self.triples, self.anchors, self.mask = _merge_with_receiver_slot(
            self.mem, triples, deltas, obs[:2], self.receiver_node
        )
        return self.boxes, self.relations

    def deltas(self, obs, env):
        """Agent-frame offsets, with the receiver slot refreshed from the env.

        The static objects keep the anchors their last SAM pass measured; only
        the receiver -- the one thing that moves -- is re-read every step.
        """
        self.anchors[-1, 0] = torch.as_tensor(
            env.receiver_anchor(), dtype=self.anchors.dtype
        )
        return deltas_from_anchors(self.anchors, self.mask, obs)


class CommunicationTrainer:
    """Clipped-objective PPO with GAE, run on the transmitter and receiver."""

    def __init__(
        self,
        transmitter,
        receiver,
        env,
        make_tx_input,
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
        perceive_passes=1,
        record_trajectories=False,
        normalize_value_loss=True,
    ):
        self.tx = transmitter.to(device)
        self.rx = receiver.to(device)
        self.env = env
        # the same factory the video recorder uses, so both drive the
        # transmitter through one input implementation
        self.tx_input = make_tx_input()
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
        self.perceive_passes = perceive_passes
        self.record_trajectories = record_trajectories
        self.normalize_value_loss = normalize_value_loss

        self.tx_optimizer = torch.optim.Adam(self.tx.parameters(), lr=lr)
        self.rx_optimizer = torch.optim.Adam(self.rx.parameters(), lr=lr)

        self._obs = None
        self._symbols = []  # emitted this episode, oldest first
        self._ep_return = 0.0
        self._ep_len = 0
        self._traj = None
        self.trajectories = []
        self.last_symbols = None
        self.last_actions = None

    # ------------------------------------------------------------- perception

    def _reset_env(self):
        obs, _ = self.env.reset()
        self._obs = obs
        self.tx_input.reset()
        self._symbols = []
        self.tx_input.refresh(self._obs)
        self._ep_return = 0.0
        self._ep_len = 0
        if self.record_trajectories:
            self._traj = {
                "positions": [self.env.receiver_xy.tolist()],
                "transmitter": self.env.data.qpos[0:2].copy().tolist(),
                "lion": self.env.model.body("lion").pos[:2].copy().tolist(),
                "food": self.env.model.body("food").pos[:2].copy().tolist(),
                "cage": self.env.model.body("cage").pos[:2].copy().tolist(),
                "lion_caged": bool(self.env.lion_caged),
                "symbols": [],
            }

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

    # --------------------------------------------------------------- rollout

    def collect_and_update(self):
        """One iteration: collect `horizon` steps, then update both agents."""
        if self._obs is None:
            self._reset_env()

        H = self.horizon
        dev = self.device
        obs_dim = self.env.observation_space.shape[0]
        k = self.tx_input.triples.shape[0]
        n_actions = int(self.env.action_space.n)
        n_symbols = self.tx.actor[-1].out_features

        tx_obs = torch.zeros(H, obs_dim, device=dev)
        tx_tri = torch.zeros(H, k, 3, dtype=torch.long, device=dev)
        tx_del = torch.zeros(H, k, 2, 2, device=dev)
        tx_act = torch.zeros(H, dtype=torch.long, device=dev)
        tx_logp = torch.zeros(H, device=dev)
        tx_val = torch.zeros(H, device=dev)

        rx_ctx = torch.zeros(H, CONTEXT_LEN, dtype=torch.long, device=dev)
        rx_act = torch.zeros(H, dtype=torch.long, device=dev)
        rx_logp = torch.zeros(H, device=dev)
        rx_val = torch.zeros(H, device=dev)

        rew = torch.zeros(H, device=dev)
        done = torch.zeros(H, device=dev)

        ep_returns, ep_lengths, foods, deaths, timeouts = [], [], 0, 0, 0
        caged_eps, caged_foods, loose_eps, loose_foods, loose_deaths = 0, 0, 0, 0, 0

        self.tx.eval()
        self.rx.eval()
        for t in range(H):
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=dev)
            tri_t = self.tx_input.triples.to(dev)
            del_t = self.tx_input.deltas(self._obs, self.env).to(dev)

            # 1. transmitter emits one symbol
            symbol, s_logp, s_val = self.tx.act(
                obs_t.unsqueeze(0), tri_t.unsqueeze(0), del_t.unsqueeze(0)
            )
            sym = int(symbol.item())
            self._symbols.append(sym)

            # 2-3. receiver reads the updated context and moves
            ctx_t = build_context(self._symbols, device=dev)
            action, a_logp, a_val = self.rx.act(ctx_t.unsqueeze(0))

            next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
            self._ep_return += reward
            self._ep_len += 1
            if self._traj is not None:
                self._traj["positions"].append(self.env.receiver_xy.tolist())
                self._traj["symbols"].append(sym)

            tx_obs[t], tx_tri[t], tx_del[t] = obs_t, tri_t, del_t
            tx_act[t], tx_logp[t], tx_val[t] = symbol.squeeze(0), s_logp.squeeze(0), s_val.squeeze(0)
            rx_ctx[t], rx_act[t] = ctx_t, action.squeeze(0)
            rx_logp[t], rx_val[t] = a_logp.squeeze(0), a_val.squeeze(0)
            rew[t] = reward
            done[t] = float(terminated or truncated)

            if terminated or truncated:
                ep_returns.append(self._ep_return)
                ep_lengths.append(self._ep_len)
                food = bool(info.get("food_reached", False))
                death = bool(info.get("lion_caught", False))
                foods += food
                deaths += death
                timeouts += bool(truncated and not (food or death))
                if info["lion_caged"]:
                    caged_eps += 1
                    caged_foods += food
                else:
                    loose_eps += 1
                    loose_foods += food
                    loose_deaths += death
                self._finish_trajectory(info, truncated)
                self._reset_env()
            else:
                self._obs = next_obs
                if self._ep_len < self.perceive_passes:
                    # SAM runs only on the first `perceive_passes` steps of an
                    # episode; the transmitter never moves and the static
                    # objects never move, so their anchors are exact for the
                    # rest of the episode. Only the receiver keeps updating,
                    # and it comes from the oracle, not from SAM.
                    self.tx_input.refresh(self._obs)

        # bootstrap both critics from the state the buffer stopped in
        with torch.no_grad():
            obs_t = torch.as_tensor(self._obs, dtype=torch.float32, device=dev)
            tx_last = self.tx.value(
                obs_t.unsqueeze(0),
                self.tx_input.triples.to(dev).unsqueeze(0),
                self.tx_input.deltas(self._obs, self.env).to(dev).unsqueeze(0),
            ).squeeze(0)
            rx_last = self.rx.value(
                build_context(self._symbols, device=dev).unsqueeze(0)
            ).squeeze(0)

        tx_adv, tx_ret = self._gae(rew, done, tx_val, tx_last)
        rx_adv, rx_ret = self._gae(rew, done, rx_val, rx_last)

        self.tx.train()
        self.rx.train()
        tx_stats = self._update(
            self.tx, self.tx_optimizer, tx_adv, tx_ret, tx_val, tx_logp, tx_act,
            lambda idx: (tx_obs[idx], tx_tri[idx], tx_del[idx]),
        )
        rx_stats = self._update(
            self.rx, self.rx_optimizer, rx_adv, rx_ret, rx_val, rx_logp, rx_act,
            lambda idx: (rx_ctx[idx],),
        )

        syms = tx_act.cpu().numpy()
        acts = rx_act.cpu().numpy()
        self.last_symbols, self.last_actions = syms, acts
        sym_fracs = np.bincount(syms, minlength=n_symbols) / H
        act_fracs = np.bincount(acts, minlength=n_actions) / H

        n_eps = max(1, len(ep_returns))
        stats = {
            "episodes": len(ep_returns),
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "food_rate": foods / n_eps,
            "death_rate": deaths / n_eps,
            "timeout_rate": timeouts / n_eps,
            "caged_food_rate": caged_foods / max(1, caged_eps),
            "loose_food_rate": loose_foods / max(1, loose_eps),
            "loose_death_rate": loose_deaths / max(1, loose_eps),
            "symbol_fracs": sym_fracs.tolist(),
            "symbol_entropy": _entropy(sym_fracs),
            "action_fracs": act_fracs.tolist(),
        }
        stats.update({f"tx_{k2}": v for k2, v in tx_stats.items()})
        stats.update({f"rx_{k2}": v for k2, v in rx_stats.items()})
        return stats

    # ---------------------------------------------------------------- update

    def _gae(self, rew, done, val, last_val):
        adv = torch.zeros_like(rew)
        running = 0.0
        for t in reversed(range(self.horizon)):
            next_val = last_val if t == self.horizon - 1 else val[t + 1]
            nonterminal = 1.0 - done[t]
            delta = rew[t] + self.gamma * next_val * nonterminal - val[t]
            running = delta + self.gamma * self.gae_lambda * nonterminal * running
            adv[t] = running
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def _update(self, model, optimizer, adv, ret, val, logp_old, act, inputs_fn):
        ret_scale = (ret.std() + 1e-8) if self.normalize_value_loss else 1.0
        pi_losses, v_losses, entropies, kls, clips = [], [], [], [], []
        for _ in range(self.epochs):
            for idx in torch.randperm(self.horizon, device=self.device).split(
                self.minibatch_size
            ):
                logp, entropy, value = model.evaluate(*inputs_fn(idx), act[idx])
                ratio = torch.exp(logp - logp_old[idx])
                clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                pi_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
                v_loss = (((value - ret[idx]) / ret_scale) ** 2).mean()
                ent = entropy.mean()

                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                pi_losses.append(float(pi_loss.detach()))
                v_losses.append(float(v_loss.detach()))
                entropies.append(float(ent.detach()))
                with torch.no_grad():
                    kls.append(float((logp_old[idx] - logp).mean()))
                    clips.append(float(((ratio - 1.0).abs() > self.clip_ratio).float().mean()))

        with torch.no_grad():
            ret_var = float(ret.var())
            ev = 1.0 - float((ret - val).var()) / ret_var if ret_var > 1e-8 else 0.0
        return {
            "pi_loss": float(np.mean(pi_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_frac": float(np.mean(clips)),
            "explained_var": ev,
        }


def _entropy(fracs):
    p = np.asarray(fracs, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
