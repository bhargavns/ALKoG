"""Mechanism checks: no renderer, pretrained weights, or SAM inference required."""
import io
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
from torch import nn

from lib.CommunicationChannel import CommunicationChannel, symbols_to_bits, bits_to_symbols
from lib.DIALPPOTrainer import DIALPPOTrainer, evaluate_windows
from lib.ReceiverPolicy import ReceiverActorCritic, build_context
from lib.SymbolPolicy import DIALTransmitter


torch.set_num_threads(1)


class TinyTX(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_head = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 2))

    def forward(self, obs, triples, deltas):
        return self.message_head(obs)


class ToyEnv:
    observation_space = SimpleNamespace(shape=(4,))
    action_space = SimpleNamespace(n=4)

    def __init__(self, length=5, seed=0):
        self.length = self.max_steps = length
        self.rng = np.random.default_rng(seed)
        self.resets = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.resets += 1
        self.target = int(self.rng.integers(4))
        self.step_count = 0
        return np.eye(4)[self.target], {}

    def step(self, action):
        self.step_count += 1
        correct = action == self.target
        end = self.step_count == self.length
        return np.eye(4)[self.target], float(correct), end, False, {
            'food_reached': bool(correct and end), 'lion_caught': False, 'lion_caged': True}


class ToyInput:
    triples = torch.tensor([[0, -1, -1]])

    def reset(self):
        pass

    def refresh(self, obs):
        pass

    def deltas(self, obs, env):
        return torch.zeros(1, 2, 2)


def trainer(length=5, horizon=8, **kwargs):
    torch.manual_seed(17)
    return DIALPPOTrainer(TinyTX(), ReceiverActorCritic(), ToyEnv(length), ToyInput,
        device='cpu', horizon=horizon, epochs=2, minibatch_size=4, **kwargs)


def test_binary_embedding_equivalence_and_padding():
    torch.manual_seed(3)
    rx = ReceiverActorCritic().eval()
    for symbols in ([0], [0, 1, 2, 3], list(range(4))*3):
        ctx = build_context(symbols).unsqueeze(0)
        retained = torch.tensor(symbols[-10:])
        bits = torch.zeros(1, 10, 2)
        valid = torch.zeros(1, 10, dtype=torch.bool)
        bits[0, :len(retained)] = symbols_to_bits(retained)
        valid[0, :len(retained)] = True
        with torch.no_grad():
            d, v = rx.distribution_bits(bits, valid)
            torch.testing.assert_close(d.probs, rx.dist(ctx).probs, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(v, rx.value(ctx), atol=1e-6, rtol=1e-5)
            # Invalid slots have no effect, even if their bit contents change.
            bits[~valid] = 123.
            torch.testing.assert_close(d.probs, rx.distribution_bits(bits, valid)[0].probs)
    assert bits_to_symbols(symbols_to_bits(torch.arange(4))).tolist() == [0, 1, 2, 3]


def test_channel_noise_reuse_and_hard_threshold():
    channel = CommunicationChannel(2)
    logits = torch.tensor([[-1., 2.]], requires_grad=True)
    noise = torch.tensor([[.2, -.3]])
    expected = torch.sigmoid(logits + 2*noise)
    torch.testing.assert_close(channel(logits, noise), expected)
    assert channel(logits, hard=True).tolist() == [[0., 1.]]
    channel(logits, noise).sum().backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad != 0).all()
    with pytest.raises(ValueError):
        CommunicationChannel(-1)


def test_replay_ratios_prefixes_pending_message_and_resets():
    t = trainer(length=13, horizon=8)
    b, _ = t.collect_rollout()
    d, _, _, _ = evaluate_windows(t.tx, t.rx, t.channel, b['records'], b['windows'])
    torch.testing.assert_close(d.log_prob(b['actions']), b['logp'], atol=1e-6, rtol=1e-5)
    assert b['windows'][0].tolist() == [0]+[-1]*9
    previous = t.last_window_record_ids[-1]
    pending_id = t._pending['id']
    pending_noise = t._pending['noise'].clone()
    # Perform an actual update before collecting the next horizon.
    t.update(b)
    b2, _ = t.collect_rollout()
    assert t.last_window_record_ids[0] == previous + [pending_id]
    local_pending_index = b2['windows'][0, len(previous)]
    torch.testing.assert_close(b2['records']['noise'][local_pending_index], pending_noise)
    # Episode ends after five steps in this second horizon; next window is clean.
    assert len(t.last_window_record_ids[5]) == 1
    for ids in t.last_window_record_ids:
        assert len(ids) == len(set(ids))
        assert len({t.last_record_episodes[i] for i in ids}) == 1
        assert len(ids) <= 10
    d2, _, _, _ = evaluate_windows(t.tx, t.rx, t.channel, b2['records'], b2['windows'])
    torch.testing.assert_close(d2.log_prob(b2['actions']), b2['logp'], atol=1e-6, rtol=1e-5)


def test_actor_gradient_reaches_earlier_messages_and_detached_control():
    t = trainer(length=20, horizon=12)
    b, _ = t.collect_rollout()
    records = {k: v.clone() for k, v in b['records'].items()}
    records['obs'].requires_grad_(True)
    d, _, _, _ = evaluate_windows(t.tx, t.rx, t.channel, records, b['windows'][-1:])
    (-d.log_prob(b['actions'][-1:]).mean()).backward()
    assert sum(float(p.grad.abs().sum()) for p in t.tx.parameters()) > 0
    referenced = b['windows'][-1]
    assert (records['obs'].grad[referenced].norm(dim=-1) > 0).all()
    assert (records['obs'].grad[:2] == 0).all()  # expired messages
    t.tx.zero_grad(set_to_none=True)
    d, _, _, _ = evaluate_windows(t.tx, t.rx, t.channel, records, b['windows'], detach_channel=True)
    (-d.log_prob(b['actions']).mean()).backward()
    assert all(p.grad is None for p in t.tx.parameters())


def test_joint_update_moves_both_models_and_detached_control_does_not_move_tx():
    for detach in (False, True):
        t = trainer(detach_channel=detach)
        before_tx = [p.detach().clone() for p in t.tx.parameters()]
        before_rx = [p.detach().clone() for p in t.rx.parameters()]
        metrics = t.collect_and_update()
        changed = any(not torch.equal(a, b) for a, b in zip(before_tx, t.tx.parameters()))
        assert changed == (not detach)
        assert any(not torch.equal(a, b) for a, b in zip(before_rx, t.rx.parameters()))
        assert metrics['optimizer_steps'] > 0
        assert (metrics['tx_actor_grad_norm'] > 0) == (not detach)
        assert np.isfinite(list(v for v in metrics.values() if isinstance(v, float))).all()


def test_critic_detach_only_stops_value_path():
    t = trainer()
    b, _ = t.collect_rollout()
    d, v, _, _ = evaluate_windows(t.tx, t.rx, t.channel, b['records'], b['windows'], critic_detach=True)
    v.sum().backward(retain_graph=True)
    assert all(p.grad is None for p in t.tx.parameters())
    (-d.log_prob(b['actions']).mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in t.tx.parameters())


def test_real_transmitter_and_checkpoint_roundtrip():
    kg = SimpleNamespace(symbols=torch.randn(4, 4), relation_symbols=torch.randn(2, 4))
    tx = DIALTransmitter(kg)
    obs, tri, delta = torch.randn(2, 4), torch.full((2, 7, 3), -1), torch.randn(2, 7, 2, 2)
    tri[:, 0] = torch.tensor([0, 0, 1])
    assert tx(obs, tri, delta).shape == (2, 2)
    assert not hasattr(tx, 'critic')
    stream = io.BytesIO()
    torch.save(tx.state_dict(), stream)
    stream.seek(0)
    copy = DIALTransmitter(kg)
    copy.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(copy(obs, tri, delta), tx(obs, tri, delta))


def test_kl_guard_rejects_update_before_parameters_change():
    t = trainer(target_kl=.01)
    batch, _ = t.collect_rollout()
    batch['logp'] = batch['logp'] + 5  # deliberately incompatible behavior policy
    before = [p.clone() for p in t.tx.parameters()]
    metrics = t.update(batch)
    assert metrics['kl_early_stop'] and metrics['stopped_kl'] > .01
    assert metrics['optimizer_steps'] == 0
    assert all(torch.equal(a, b) for a, b in zip(before, t.tx.parameters()))


def test_bootstrap_uses_pending_message_without_consuming_it():
    t = trainer(length=50)
    calls = []
    original = t._decision
    def capture(history):
        calls.append([r['id'] for r in history])
        return original(history)
    t._decision = capture
    batch, _ = t.collect_rollout()
    assert len(calls) == t.horizon + 1
    assert calls[-1] == t.last_window_record_ids[-1] + [t._pending['id']]
    assert t._pending['id'] not in t.last_window_record_ids[-1]
    calls.clear()
    t.collect_rollout()
    assert calls[0] == list(range(9))


def test_hard_inference_and_interventions():
    from lib.CommunicationEvaluation import CommunicationActor, evaluate_episode
    t = trainer()
    for mode in ('hard', 'continuous', 'constant', 'random', 'shuffle'):
        actor = CommunicationActor(t.tx, t.rx, 'cpu', method='dial_ppo', mode=mode,
                                   shuffled_messages=[3, 2] if mode == 'shuffle' else None)
        row = evaluate_episode(ToyEnv(length=3), ToyInput, actor, seed=10)
        assert row['length'] == 3 and len(row['messages']) == 3
        assert all(0 <= s < 4 for s in row['messages'])
        if mode == 'constant':
            assert row['messages'] == [0, 0, 0]
        if mode == 'shuffle':
            assert row['messages'] == [3, 2, 2]


def test_independent_baseline_can_still_update():
    from lib.CommunicationTrainer import CommunicationTrainer
    from lib.SymbolPolicy import TripleActorCritic
    kg = SimpleNamespace(symbols=torch.randn(4, 4), relation_symbols=torch.randn(2, 4))
    baseline = CommunicationTrainer(TripleActorCritic(kg, k_triples=1), ReceiverActorCritic(),
        ToyEnv(length=3), ToyInput, device='cpu', horizon=8, epochs=1, minibatch_size=4)
    stats = baseline.collect_and_update()
    assert stats['episodes'] == 2
    assert 'tx_explained_var' in stats and 'rx_explained_var' in stats


def test_evaluation_intervals_do_not_claim_certainty_from_zero_events():
    from lib.CommunicationEvaluation import summarize
    rows = [{'outcome': 'timeout', 'return': -3., 'length': 300, 'lion_caged': False}]
    stats = summarize(rows)
    assert stats['food_rate']['mean'] == 0
    assert stats['food_rate']['ci95'][1] > .5  # one episode cannot establish a zero rate
    assert stats['mean_return']['ci95'] is None
    assert summarize([])['food_rate']['mean'] is None
