"""Shared Gumbel inference, interventions, and episode evaluation."""
from representation.CommunicationCadence import perception_due
import numpy as np
import torch


class CommunicationActor:
    def __init__(self, tx, rx, device, mode='hard', greedy=False, shuffled_messages=None):
        if mode not in ('hard', 'sampled', 'constant', 'random', 'shuffle'):
            raise ValueError(f'Unknown channel mode: {mode}')
        if mode == 'shuffle' and not shuffled_messages:
            raise ValueError('Shuffle evaluation requires another episode message stream')
        self.tx, self.rx, self.device = tx, rx, device
        self.mode, self.greedy = mode, greedy
        self.symbols = []
        self.shuffled_messages = shuffled_messages
        self.step = 0

    @torch.no_grad()
    def act(self, obs, triples, deltas):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        triples = triples.to(self.device).unsqueeze(0)
        deltas = deltas.to(self.device).unsqueeze(0)
        logits = self.tx(obs, triples, deltas)[0]
        n_symbols = self.rx.token_embeddings.num_embeddings
        symbol = torch.distributions.Categorical(logits=logits).sample() if self.mode == 'sampled' else logits.argmax(-1)
        if self.mode == 'constant':
            symbol = torch.tensor(0, device=self.device)
        elif self.mode == 'random':
            symbol = torch.randint(n_symbols, (), device=self.device)
        elif self.mode == 'shuffle':
            symbol = torch.tensor(self.shuffled_messages[min(self.step, len(self.shuffled_messages)-1)], device=self.device)
        if not 0 <= int(symbol) < n_symbols:
            raise ValueError('Symbol outside checkpoint vocabulary')
        message = torch.nn.functional.one_hot(symbol, n_symbols).float().reshape(1, n_symbols)
        dist, value = self.rx.distribution_messages(message)
        action = dist.probs.argmax(-1) if self.greedy else dist.sample()
        self.step += 1
        self.symbols.append(int(symbol))
        return int(action), int(symbol), float(value), str(int(symbol))


def episode_outcome(info, truncated):
    for key, name in (('lion_caught', 'death'), ('food_reached', 'food'),
                      ('water_reached', 'water'), ('poison_touched', 'poison')):
        if info.get(key):
            return name
    return 'timeout' if truncated else 'end'


@torch.no_grad()
def evaluate_episode(env, make_tx_input, actor, perceive_passes=1, seed=None, recorder=None):
    obs, _ = env.reset(seed=seed)
    tx_input = make_tx_input()
    tx_input.reset()
    tx_input.refresh(obs)
    total = 0.
    if recorder is not None:
        recorder.add(env.render_panorama(), hud_lines=[
            f"Scene {seed} | {actor.mode} |  sampled receiver={not actor.greedy}",
            f"step 0 | lion_caged={env.lion_caged}"])
    for step in range(1, env.max_steps+1):
        action, symbol, _, _ = actor.act(obs, tx_input.triples, tx_input.deltas(obs, env))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if recorder is not None:
            outcome = episode_outcome(info, truncated) if terminated or truncated else 'running'
            frames = env.render_panorama()
            hud = [f"Scene {seed} | {actor.mode}",
                   f"step {step} | TX=s{symbol} | RX={('forward', 'backward', 'left', 'right')[action]}",
                   f"return={total:+.3f} | lion_caged={env.lion_caged} | {outcome}"]
            recorder.add(frames, hud_lines=hud)
            if terminated or truncated:
                for _ in range(12):
                    recorder.add(frames, hud_lines=hud)
        if terminated or truncated:
            break
        if perception_due(env, step, perceive_passes):
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
