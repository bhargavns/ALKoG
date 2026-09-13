"""Joint PPO with reconstructed noisy messages, including rollout-window prefixes.

Buffers contain observations and parameter-independent noise, never detached
message activations used as update inputs. The hard channel is evaluation only.
Time limits retain the baseline's finite-task terminal convention.
"""

import numpy as np
import torch

from lib.CommunicationChannel import CommunicationChannel, bits_to_symbols
from lib.CommunicationTrainer import CommunicationTrainer, _entropy
from lib.ReceiverPolicy import MAX_SYMBOLS


def pack_records(records, device):
    return {key: torch.stack([r[key] for r in records]).to(device)
            for key in ('obs', 'triples', 'deltas', 'noise')}


def evaluate_windows(tx, rx, channel, records, windows, *, detach_channel=False,
                     critic_detach=False):
    """Replay every valid message once per minibatch, then restore each window."""
    valid = windows >= 0
    unique, inverse = torch.unique(windows[valid], sorted=True, return_inverse=True)
    if not unique.numel():
        raise ValueError('A receiver decision must contain at least one message')
    logits = tx(records['obs'][unique], records['triples'][unique], records['deltas'][unique])
    messages = channel(logits, records['noise'][unique])
    if detach_channel:
        messages = messages.detach()
    bits = messages.new_zeros((*windows.shape, 2))
    bits[valid] = messages[inverse]
    dist, value = rx.distribution_bits(bits, valid, critic_detach=critic_detach)
    return dist, value, logits, messages


class DIALPPOTrainer(CommunicationTrainer):
    """Uses the baseline's input/trajectory lifecycle, but a single joint loss."""

    def __init__(self, transmitter, receiver, env, make_tx_input, device='cuda',
                 horizon=2048, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
                 epochs=10, minibatch_size=256, lr=3e-4, ent_coef=0.01,
                 vf_coef=0.5, max_grad_norm=0.5, perceive_passes=1,
                 record_trajectories=False, normalize_value_loss=True,
                 sigma=2.0, target_kl=0.03, detach_channel=False, critic_detach=False):
        if horizon < 2 or epochs < 1 or minibatch_size < 1 or perceive_passes < 1:
            raise ValueError('horizon >= 2 and positive epochs/minibatch/perceive_passes required')
        self.tx, self.rx = transmitter.to(device), receiver.to(device)
        self.env, self.tx_input = env, make_tx_input()
        self.device, self.horizon = device, horizon
        self.gamma, self.gae_lambda, self.clip_ratio = gamma, gae_lambda, clip_ratio
        self.epochs, self.minibatch_size = epochs, minibatch_size
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm, self.target_kl = max_grad_norm, target_kl
        self.perceive_passes = perceive_passes
        self.record_trajectories = record_trajectories
        self.normalize_value_loss = normalize_value_loss
        self.channel = CommunicationChannel(sigma)
        self.detach_channel, self.critic_detach = detach_channel, critic_detach
        self.optimizer = torch.optim.Adam([
            {'params': self.tx.parameters(), 'name': 'transmitter'},
            {'params': self.rx.parameters(), 'name': 'receiver'},
        ], lr=lr)
        self._obs, self._pending = None, None
        self._history = []
        self._record_id = 0
        self._episode_id = -1
        self._traj = None
        self.trajectories = []
        self.last_symbols = self.last_actions = None

    def _reset_env(self):
        super()._reset_env()
        self._history, self._pending = [], None
        self._episode_id += 1

    def _new_record(self):
        record = {
            'id': self._record_id, 'episode': self._episode_id, 'step': self._ep_len,
            'obs': torch.as_tensor(self._obs, dtype=torch.float32, device=self.device).clone(),
            'triples': self.tx_input.triples.to(self.device).clone(),
            'deltas': self.tx_input.deltas(self._obs, self.env).to(self.device).clone(),
            'noise': torch.randn(2, device=self.device),
        }
        self._record_id += 1
        return record

    @torch.no_grad()
    def _decision(self, history):
        records = pack_records(history, self.device)
        window = torch.full((1, MAX_SYMBOLS), -1, dtype=torch.long, device=self.device)
        window[0, :len(history)] = torch.arange(len(history), device=self.device)
        dist, value, logits, bits = evaluate_windows(self.tx, self.rx, self.channel, records, window)
        return dist, value, logits[-1], bits[-1]

    def collect_rollout(self):
        if self._obs is None:
            self._reset_env()
        # Dropout is zero. Keep the same module mode for collection and replay.
        self.tx.train()
        self.rx.train()
        registry, local_ids, windows = [], {}, []
        actions, logps, values, rewards, dones, symbols = [], [], [], [], [], []
        logits_seen, bits_seen, episodes = [], [], []
        record_windows = []
        for _ in range(self.horizon):
            record = self._pending if self._pending is not None else self._new_record()
            self._pending = None
            history = self._history + [record]
            assert len({r['episode'] for r in history}) == 1
            window = [-1] * MAX_SYMBOLS
            for i, r in enumerate(history):
                if r['id'] not in local_ids:
                    local_ids[r['id']] = len(registry)
                    registry.append(r)
                window[i] = local_ids[r['id']]
            windows.append(window)
            record_windows.append([r['id'] for r in history])
            dist, value, logits, bits = self._decision(history)
            action = dist.sample()
            logp = dist.log_prob(action)
            sym = int(bits_to_symbols(self.channel(logits, hard=True)))
            next_obs, reward, terminated, truncated, info = self.env.step(int(action))
            self._ep_return += reward
            self._ep_len += 1
            if self._traj is not None:
                self._traj['positions'].append(self.env.receiver_xy.tolist())
                self._traj['symbols'].append(sym)
            actions.append(action.squeeze(0))
            logps.append(logp.squeeze(0))
            values.append(value.squeeze(0))
            rewards.append(reward)
            dones.append(terminated or truncated)
            symbols.append(sym)
            logits_seen.append(logits)
            bits_seen.append(bits)
            if terminated or truncated:
                episodes.append({**info, 'return': self._ep_return, 'length': self._ep_len,
                                 'timeout': bool(truncated and not terminated)})
                self._finish_trajectory(info, truncated)
                self._reset_env()
            else:
                self._obs = next_obs
                self._history = history[-(MAX_SYMBOLS-1):]
                if self._ep_len < self.perceive_passes:
                    self.tx_input.refresh(self._obs)
        # Cache the next message's provenance/noise, not its activation. It will
        # be consumed exactly once, under the updated parameters next rollout.
        self._pending = self._new_record()
        _, last_value, _, _ = self._decision(self._history + [self._pending])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        values = torch.stack(values)
        adv, returns = self._gae(rewards, dones, values, last_value.squeeze(0))
        batch = {
            'records': pack_records(registry, self.device),
            'windows': torch.tensor(windows, dtype=torch.long, device=self.device),
            'actions': torch.stack(actions), 'logp': torch.stack(logps),
            'values': values, 'advantages': adv, 'returns': returns,
        }
        self.last_window_record_ids = record_windows
        self.last_record_episodes = {r['id']: r['episode'] for r in registry}
        self.last_symbols = np.asarray(symbols)
        self.last_actions = batch['actions'].cpu().numpy()
        fracs = np.bincount(self.last_symbols, minlength=4) / self.horizon
        caged = [e for e in episodes if e['lion_caged']]
        loose = [e for e in episodes if not e['lion_caged']]
        def rate(items, key):
            return sum(bool(e.get(key)) for e in items) / max(1, len(items))
        stats = {
            'episodes': len(episodes),
            'mean_return': float(np.mean([e['return'] for e in episodes])) if episodes else 0.,
            'mean_length': float(np.mean([e['length'] for e in episodes])) if episodes else 0.,
            'food_rate': rate(episodes, 'food_reached'),
            'death_rate': rate(episodes, 'lion_caught'),
            'water_rate': rate(episodes, 'water_reached'),
            'poison_rate': rate(episodes, 'poison_touched'),
            'timeout_rate': rate(episodes, 'timeout'),
            'caged_food_rate': rate(caged, 'food_reached'),
            'loose_food_rate': rate(loose, 'food_reached'),
            'loose_death_rate': rate(loose, 'lion_caught'),
            'symbol_fracs': fracs.tolist(), 'symbol_entropy': _entropy(fracs),
            'action_fracs': (np.bincount(self.last_actions, minlength=4) / self.horizon).tolist(),
            'channel_logit_abs': float(torch.stack(logits_seen).abs().mean()),
            'channel_saturation': float(((torch.stack(bits_seen) < .05) |
                                         (torch.stack(bits_seen) > .95)).float().mean()),
            'channel_sigma': self.channel.sigma,
        }
        return batch, stats

    def update(self, batch):
        ret = batch['returns']
        scale = ret.std().clamp_min(1e-8) if self.normalize_value_loss else 1.
        params = list(self.tx.parameters()) + list(self.rx.parameters())
        head_params = tuple(self.tx.message_head.parameters())
        rows, stop, stopped_kl = [], False, 0.
        actor_grad = value_grad = 0.
        for _ in range(self.epochs):
            for idx in torch.randperm(len(ret), device=self.device).split(self.minibatch_size):
                dist, value, _, _ = evaluate_windows(
                    self.tx, self.rx, self.channel, batch['records'], batch['windows'][idx],
                    detach_channel=self.detach_channel, critic_detach=self.critic_detach)
                logp = dist.log_prob(batch['actions'][idx])
                log_ratio = logp - batch['logp'][idx]
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                if self.target_kl > 0 and float(approx_kl.detach()) > self.target_kl:
                    stop = True
                    stopped_kl = float(approx_kl.detach())
                    break
                adv = batch['advantages'][idx]
                pi_loss = -torch.minimum(ratio * adv,
                    ratio.clamp(1-self.clip_ratio, 1+self.clip_ratio) * adv).mean()
                v_loss = ((value-ret[idx]) / scale).square().mean()
                entropy = dist.entropy().mean()
                if not rows:
                    def grad_norm(loss):
                        gs = torch.autograd.grad(loss, head_params, retain_graph=True, allow_unused=True)
                        return sum(float(g.detach().square().sum()) for g in gs if g is not None) ** .5
                    actor_grad = grad_norm(pi_loss - self.ent_coef * entropy)
                    value_grad = grad_norm(self.vf_coef * v_loss)
                loss = pi_loss + self.vf_coef*v_loss - self.ent_coef*entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                rows.append([float(x.detach()) for x in (pi_loss, v_loss, entropy, approx_kl,
                    ((ratio-1).abs() > self.clip_ratio).float().mean())])
            if stop:
                break
        means = np.mean(rows, axis=0) if rows else np.zeros(5)
        stats = dict(zip(('pi_loss', 'v_loss', 'entropy', 'approx_kl', 'clip_frac'), means.tolist()))
        var = float(ret.var())
        stats['explained_var'] = 1-float((ret-batch['values']).var())/var if var > 1e-8 else 0.
        stats.update({'tx_actor_grad_norm': actor_grad, 'tx_value_grad_norm': value_grad,
                      'kl_early_stop': stop, 'stopped_kl': stopped_kl,
                      'optimizer_steps': len(rows)})
        stats.update({f'rx_{k}': stats[k] for k in
                      ('pi_loss', 'v_loss', 'entropy', 'approx_kl', 'clip_frac', 'explained_var')})
        return stats

    def collect_and_update(self):
        batch, stats = self.collect_rollout()
        stats.update(self.update(batch))
        return stats
