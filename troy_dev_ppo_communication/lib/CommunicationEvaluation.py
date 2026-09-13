"""Shared communication inference for evaluation and video recording."""

import numpy as np
import torch

from lib.CommunicationChannel import CommunicationChannel, bits_to_symbols, symbols_to_bits
from lib.ReceiverPolicy import MAX_SYMBOLS, build_context, format_context


class CommunicationActor:
    def __init__(self, tx, rx, device, method='independent_ppo', sigma=2.,
                 mode='hard', greedy=False, shuffled_messages=None):
        if method not in ('independent_ppo', 'dial_ppo'):
            raise ValueError(f'Unknown communication method: {method}')
        if mode not in ('hard', 'continuous', 'constant', 'random', 'shuffle'):
            raise ValueError(f'Unknown channel mode: {mode}')
        if method == 'independent_ppo' and mode == 'continuous':
            raise ValueError('Continuous channel requires DIAL')
        if mode == 'shuffle' and not shuffled_messages:
            raise ValueError('Shuffle evaluation requires another episode message stream')
        self.tx, self.rx, self.device = tx, rx, device
        self.method, self.mode, self.greedy = method, mode, greedy
        self.channel = CommunicationChannel(sigma)
        self.messages, self.symbols = [], []
        self.shuffled_messages = shuffled_messages
        self.step = 0

    @torch.no_grad()
    def act(self, obs, triples, deltas):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        triples, deltas = triples.to(self.device).unsqueeze(0), deltas.to(self.device).unsqueeze(0)
        if self.method == 'independent_ppo':
            symbol, _, _ = self.tx.act(obs, triples, deltas)
            bits = symbols_to_bits(symbol[0])
        else:
            logits = self.tx(obs, triples, deltas)[0]
            bits = self.channel(logits, hard=self.mode != 'continuous')
        if self.mode == 'constant':
            bits = torch.zeros(2, device=self.device)
        elif self.mode == 'random':
            bits = symbols_to_bits(torch.randint(4, (), device=self.device))
        elif self.mode == 'shuffle':
            # Hold the last donor symbol if the recipient episode is longer.
            symbol = self.shuffled_messages[min(self.step, len(self.shuffled_messages)-1)]
            bits = symbols_to_bits(torch.tensor(symbol, device=self.device))
        self.step += 1
        sym = int(bits_to_symbols(bits))
        self.symbols.append(sym)
        self.messages = (self.messages + [bits])[-MAX_SYMBOLS:]
        if self.method == 'independent_ppo':
            ctx = build_context(self.symbols, self.device).unsqueeze(0)
            action, _, value = self.rx.act(ctx, greedy=self.greedy)
            context_text = format_context(ctx[0])
        else:
            context = bits.new_zeros((1, MAX_SYMBOLS, 2))
            valid = torch.zeros((1, MAX_SYMBOLS), dtype=torch.bool, device=self.device)
            context[0, :len(self.messages)] = torch.stack(self.messages)
            valid[0, :len(self.messages)] = True
            action, _, value = self.rx.act_bits(context, valid, greedy=self.greedy)
            context_text = (format_context(build_context(self.symbols, self.device))
                            if self.mode != 'continuous' else ' '.join(
                                f'({float(b[0]):.2f},{float(b[1]):.2f})' for b in self.messages))
        return int(action), sym, float(value), context_text


def episode_outcome(info, truncated):
    for key, name in (('lion_caught', 'death'), ('food_reached', 'food'),
                      ('water_reached', 'water'), ('poison_touched', 'poison')):
        if info.get(key):
            return name
    return 'timeout' if truncated else 'end'


@torch.no_grad()
def evaluate_episode(env, make_tx_input, actor, perceive_passes=1, seed=None):
    obs, _ = env.reset(seed=seed)
    tx_input = make_tx_input()
    tx_input.reset()
    tx_input.refresh(obs)
    total = 0.
    for step in range(1, env.max_steps+1):
        action, _, _, _ = actor.act(obs, tx_input.triples, tx_input.deltas(obs, env))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
        if step < perceive_passes:
            tx_input.refresh(obs)
    return {'return': total, 'length': step, 'outcome': episode_outcome(info, truncated),
            'lion_caged': bool(info['lion_caged']), 'scene_seed': seed,
            'messages': actor.symbols}


def summarize(rows, seed=0):
    """Within-seed Wilson rate intervals and episode-bootstrap mean intervals."""
    loose = [r for r in rows if not r['lion_caged']]
    values = {
        'food_rate': [r['outcome'] == 'food' for r in rows],
        'mean_return': [r['return'] for r in rows],
        'mean_length': [r['length'] for r in rows],
        'loose_death_rate': [r['outcome'] == 'death' for r in loose],
        'water_rate': [r['outcome'] == 'water' for r in rows],
        'poison_rate': [r['outcome'] == 'poison' for r in rows],
    }
    rng = np.random.default_rng(seed)
    result = {}
    for key, values_for_key in values.items():
        x = np.asarray(values_for_key, dtype=float)
        if not len(x):
            result[key] = {'mean': None, 'ci95': None, 'episodes': 0}
            continue
        if key.endswith('_rate'):
            # Unlike empirical bootstrap, Wilson intervals remain informative
            # when an evaluation observes zero failures or zero successes.
            n, p, z = len(x), float(x.mean()), 1.959963984540054
            denom = 1 + z*z/n
            center = (p + z*z/(2*n)) / denom
            half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
            interval = [max(0., center-half), min(1., center+half)]
        elif len(x) < 2:
            interval = None
        else:
            means = x[rng.integers(len(x), size=(2000, len(x)))].mean(axis=1)
            interval = np.quantile(means, [.025, .975]).tolist()
        result[key] = {'mean': float(x.mean()),
                       'ci95': interval, 'episodes': len(x)}
    return result

