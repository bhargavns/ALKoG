"""Joint optimization, checkpoint, bootstrap, and evaluation regression checks."""
import io
from types import SimpleNamespace
import torch
import pytest
from helpers import ToyEnv, ToyInput, Sender, trainer
from communication.GumbelPPOTrainer import GumbelPPOTrainer, evaluate_records
from communication.SymbolPolicy import GumbelTransmitter
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic


def test_pending_bootstrap_message_is_replayed_once_after_update():
    t = trainer(length=50)
    batch, _ = t.collect_rollout()
    pending = {k: v.clone() if torch.is_tensor(v) else v for k, v in t._pending_record.items()}
    assert pending['id'] not in t.last_record_ids
    t.update(batch)
    batch, _ = t.collect_rollout()
    assert t.last_record_ids[0] == pending['id']
    assert len(set(t.last_record_ids)) == t.horizon
    torch.testing.assert_close(batch['records']['noise'][0], pending['noise'], atol=0, rtol=0)
    dist, *_ = evaluate_records(t.transmitter, t.receiver, t.channel, batch['records'])
    torch.testing.assert_close(dist.log_prob(batch['actions']), batch['logp'])


def test_finite_episode_terminal_does_not_bootstrap():
    t = trainer(horizon=2)
    _, returns = t._gae(torch.tensor([1., 2.]), torch.tensor([0., 1.]),
                        torch.zeros(2), torch.tensor(1000.))
    torch.testing.assert_close(returns, torch.tensor([1 + t.gamma * t.advantage_trace_decay * 2, 2.]))


@pytest.mark.parametrize('mode', ['training', 'evaluation'])
def test_perception_cadence_across_rollouts_and_resets(mode):
    from evaluation.CommunicationEvaluation import CommunicationActor, evaluate_episode
    env = ToyEnv(length=151)
    env.colocated = True
    calls = []

    class CountingInput(ToyInput):
        def refresh(self, obs):
            calls.append((env.resets, env.step_count))

    tx, rx = Sender(), CurrentMessageReceiverActorCritic(n_symbols=16)
    if mode == 'evaluation':
        evaluate_episode(env, CountingInput, CommunicationActor(tx, rx, 'cpu'), perceive_passes=5)
        assert calls == [(1, 0), (1, 75), (1, 150)]
    else:
        t = GumbelPPOTrainer(tx, rx, env, CountingInput, device='cpu', horizon=80,
                            epochs=1, minibatch_size=80, perceive_passes=5)
        t.collect_and_update()
        t.collect_and_update()
        assert calls == [(1, 0), (1, 75), (1, 150), (2, 0)]


@pytest.mark.parametrize('detach', [False, True])
@pytest.mark.parametrize('loss_path', ['actor', 'critic'])
def test_receiver_losses_train_sender_unless_channel_detached(detach, loss_path):
    torch.manual_seed(42)
    t = trainer()
    batch, _ = t.collect_rollout()
    dist, value, _, _ = evaluate_records(
        t.transmitter, t.receiver, t.channel, batch['records'], detach_channel=detach)
    loss = value.sum() if loss_path == 'critic' else -dist.log_prob(batch['actions']).mean()
    loss.backward()
    sender_grads = [p.grad for p in t.transmitter.parameters()]
    if detach:
        assert all(g is None for g in sender_grads)
    else:
        assert any(g is not None and g.abs().sum() > 0 for g in sender_grads)
    assert t.receiver.token_embeddings.weight.grad.abs().sum() > 0
    head = t.receiver.critic if loss_path == 'critic' else t.receiver.actor
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())

def test_real_transmitter_and_checkpoint_roundtrip():
    vocabulary = SimpleNamespace(symbols=torch.randn(4, 4), relation_symbols=torch.randn(2, 4))
    tx = GumbelTransmitter(vocabulary)
    obs, tri, delta = torch.randn(2, 4), torch.full((2, 7, 3), -1), torch.randn(2, 7, 2, 2)
    tri[:, 0] = torch.tensor([0, 0, 1])
    assert tx(obs, tri, delta).shape == (2, 16)
    assert not hasattr(tx, 'critic')
    stream = io.BytesIO()
    torch.save(tx.state_dict(), stream)
    stream.seek(0)
    copy = GumbelTransmitter(vocabulary)
    copy.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(copy(obs, tri, delta), tx(obs, tri, delta))

def test_kl_guard_rejects_update_before_parameters_change():
    t = trainer(target_kl=.01)
    batch, _ = t.collect_rollout()
    batch['logp'] = batch['logp'] + 5  # deliberately incompatible behavior policy
    before = [p.clone() for p in t.transmitter.parameters()]
    metrics = t.update(batch)
    assert metrics['kl_early_stop'] and metrics['stopped_kl'] > .01
    assert metrics['optimizer_steps'] == 0
    assert all(torch.equal(a, b) for a, b in zip(before, t.transmitter.parameters()))

def test_hard_inference_and_interventions():
    from evaluation.CommunicationEvaluation import CommunicationActor, evaluate_episode
    t = trainer()
    for mode in ('hard', 'sampled', 'constant', 'random', 'shuffle'):
        actor = CommunicationActor(t.transmitter, t.receiver, 'cpu', mode=mode,
                                   shuffled_messages=[3, 2] if mode == 'shuffle' else None)
        row = evaluate_episode(ToyEnv(length=3), ToyInput, actor, seed=10)
        assert row['length'] == 3 and len(row['messages']) == 3
        assert all(0 <= s < 16 for s in row['messages'])
        if mode == 'constant':
            assert row['messages'] == [0, 0, 0]
        if mode == 'shuffle':
            assert row['messages'] == [3, 2, 2]

def test_evaluation_intervals_do_not_claim_certainty_from_zero_events():
    from evaluation.CommunicationEvaluation import summarize
    rows = [{'outcome': 'timeout', 'return': -3., 'length': 300, 'lion_caged': False}]
    stats = summarize(rows)
    assert stats['food_rate']['mean'] == 0
    assert stats['food_rate']['ci95'][1] > .5  # one episode cannot establish a zero rate
    assert stats['mean_return']['ci95'] is None
    assert summarize([])['food_rate']['mean'] is None

def test_colocated_video_uses_same_perception_schedule(monkeypatch):
    from observability import VideoRecorder
    env = ToyEnv(length=151)
    env.colocated = True
    env.render_size = 8
    env.lion_caged = True
    env.render_panorama = lambda: []
    calls = []

    class Input(ToyInput):
        def refresh(self, obs):
            calls.append(env.step_count)
            return [], []

    monkeypatch.setattr(VideoRecorder, 'PanoramaRecorder',
                        lambda *a, **k: SimpleNamespace(add=lambda *a, **k: None,
                                                       close=lambda: None))
    monkeypatch.setattr(VideoRecorder, 'format_triples', lambda *a: '')
    VideoRecorder.record_communication_episode('unused.mp4', env, Sender(),
        CurrentMessageReceiverActorCritic(n_symbols=16), None, Input, 5, 'cpu')
    assert calls == [0, 75, 150]
