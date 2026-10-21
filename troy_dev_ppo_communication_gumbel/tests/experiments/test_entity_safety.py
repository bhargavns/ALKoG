"""Behavioral contracts for the oracle composition diagnostic."""
import json

import pytest
import torch

from communication.CommunicationChannel import StraightThroughGumbelChannel
from communication.ReceiverPolicy import SelectionMessageReceiver
from communication.SymbolPolicy import OracleSequenceTransmitter
from experiments.entity_safety.evaluation import metrics
from experiments.entity_safety.run import Config, aggregate, train_one
from experiments.entity_safety.task import Episodes, matched_shuffle, sample_episodes
from experiments.entity_safety.trainer import decision


def episodes(count=96, matched=False):
    return sample_episodes(count, torch.Generator().manual_seed(123), matched_layouts=matched)


def test_balanced_meanings_and_safety_truth_table():
    batch = episodes()
    for entity in range(3):
        for caged in (False, True):
            mask = (batch.target_entity == entity) & (batch.target_caged == caged)
            assert mask.sum() == 16
            assert (batch.dangerous[mask] == (entity != 2 and not caged)).all()
    assert (batch.identities.sort(-1).values == torch.arange(3)).all()
    assert len(batch.target.unique()) == 3


def test_receiver_private_information_isolation():
    batch = episodes()
    changed = Episodes(batch.identities, batch.positions, ~batch.cages, (batch.target + 1) % 3)
    assert torch.equal(batch.receiver_input(), changed.receiver_input())
    assert not torch.equal(batch.sender_input(), changed.sender_input())
    assert batch.receiver_input().shape == (96, 3, 5)


def test_reward_requires_both_entity_and_response_including_warnings():
    batch = episodes()
    correct = batch.target * 2 + batch.dangerous.long()
    assert batch.reward(correct).eq(1).all()
    assert batch.reward(batch.target * 2 + 1 - batch.dangerous.long()).eq(0).all()
    assert batch.reward(((batch.target + 1) % 3) * 2 + batch.dangerous.long()).eq(0).all()
    assert batch.reward(((batch.target + 1) % 3) * 2 + 1).eq(0).all()


@pytest.mark.parametrize('length', [1, 2])
def test_complete_discrete_sequence_replay_and_sender_gradient(length):
    batch = episodes()
    sender = OracleSequenceTransmitter(input_dimension=5, message_length=length)
    receiver = SelectionMessageReceiver(candidate_dimension=5, message_length=length)
    channel = StraightThroughGumbelChannel(n_symbols=6)
    noise = channel.sample_noise((96, length, 6), 'cpu')
    dist, values, logits, message = decision(sender, receiver, channel, batch, noise)
    again = decision(sender, receiver, channel, batch, noise)
    assert message.shape == (96, length, 6)
    assert torch.equal(message, again[3])
    assert torch.equal(dist.logits, again[0].logits)
    assert ((message == 0) | (message == 1)).all()
    assert message.sum(-1).eq(1).all()
    assert logits.shape == noise.shape
    loss = -dist.log_prob(batch.target * 2 + batch.dangerous.long()).mean() + values.square().mean()
    loss.backward()
    assert sum(p.grad.abs().sum() for p in sender.parameters()) > 0
    assert not torch.equal(noise[:, 0], noise[:, -1]) if length > 1 else True


def test_receiver_candidate_permutation_equivariance():
    batch = episodes()
    receiver = SelectionMessageReceiver(candidate_dimension=5)
    messages = torch.nn.functional.one_hot(torch.randint(6, (96, 2)), 6).float()
    candidates = batch.receiver_input()
    perm = torch.tensor([2, 0, 1])
    original, value = receiver.distribution_messages(messages, candidates)
    shuffled, shuffled_value = receiver.distribution_messages(messages, candidates[:, perm])
    assert torch.allclose(original.probs.reshape(-1, 3, 2)[:, perm],
                          shuffled.probs.reshape(-1, 3, 2), atol=1e-6)
    assert torch.allclose(value, shuffled_value, atol=1e-6)


def test_matched_shuffle_preserves_public_layout_and_moves_meanings():
    batch = episodes(matched=True)
    indices = matched_shuffle(96, torch.Generator().manual_seed(42))
    assert torch.equal(batch.receiver_input(), batch.receiver_input()[indices])
    assert not (indices == torch.arange(96)).any()
    meaning = batch.target_entity * 2 + batch.target_caged.long()
    assert (meaning != meaning[indices]).all()
    assert torch.equal(indices.sort().values, torch.arange(96))


def test_unsafe_approach_denominator_and_sheep_cases():
    batch = episodes()
    actions = batch.target * 2  # Always APPROACH, correctly identify target.
    report = metrics(batch, actions)
    assert report['unsafe_approach_rate'] == 1
    assert report['dangerous_count'] == 32
    assert report['joint_accuracy'] == pytest.approx(2 / 3)
    assert metrics(batch, actions, batch.target_entity == 2)['unsafe_approach_rate'] is None


def test_update_and_all_evaluation_artifacts(tmp_path):
    config = Config(iterations=2, batch_size=12, evaluation_episodes=12, epochs=1, evaluation_seed=999)
    result = train_one(config, 1, 2, tmp_path / 'run', {'commit': 'test'})
    assert set(result) == {'normal', 'no_message', 'constant', 'shuffled', 'sampled', 'computation'}
    assert len(result['normal']['by_meaning']) == 6
    assert len((tmp_path / 'run' / 'evaluation.jsonl').read_text().splitlines()) == 60
    checkpoint = torch.load(tmp_path / 'run' / 'checkpoint.pt', weights_only=True)
    assert checkpoint['config']['message_length'] == 2
    record = torch.load(tmp_path / 'run' / 'rollouts' / '00001.pt', weights_only=True)
    assert record['noise'].shape == (12, 2, 6)
    assert record['scope'] == 'one_complete_utterance_then_one_terminal_response'
    summary = aggregate({2: [result]})
    assert summary['2']['normal']['metrics']['joint_accuracy']['std'] is None
    manifest = json.loads((tmp_path / 'run' / 'split_manifest.json').read_text())
    assert manifest['excluded_target_meanings'] == []


@pytest.mark.parametrize('count', [0, 5, 11])
def test_incomplete_balanced_batches_rejected(count):
    with pytest.raises(ValueError):
        sample_episodes(count, torch.Generator())


def test_exact_reward_matches_explicit_discrete_reward_sum_and_gradient():
    from experiments.entity_safety.exact import expected_reward
    batch = episodes(6)
    sender = OracleSequenceTransmitter(input_dimension=5, message_length=1, n_symbols=2)
    receiver = SelectionMessageReceiver(candidate_dimension=5, message_length=1, n_symbols=2)
    expected, _ = expected_reward(sender, receiver, batch)
    manual = torch.zeros(6)
    weights = sender(batch.sender_input()).softmax(-1)[:, 0]
    for token in range(2):
        message = torch.nn.functional.one_hot(torch.full((6, 1), token), 2).float()
        dist, _ = receiver.distribution_messages(message, batch.receiver_input())
        for action in range(6):
            manual += weights[:, token] * dist.probs[:, action] * batch.reward(torch.full((6,), action))
    assert torch.allclose(expected, manual, atol=1e-6)
    exact_grad = torch.autograd.grad(expected.mean(), sender.message_head[-1].weight, retain_graph=True)[0]
    manual_grad = torch.autograd.grad(manual.mean(), sender.message_head[-1].weight)[0]
    assert exact_grad.abs().sum() > 0
    assert torch.allclose(exact_grad, manual_grad, atol=1e-6)


def test_fixed_code_is_frozen_holistic_and_sheep_state_invariant():
    from experiments.entity_safety.exact import FixedCodeSender, message_probabilities, enumerate_messages
    batch = episodes(6, matched=True)
    sender = FixedCodeSender()
    tokens = sender(batch.sender_input()).argmax(-1)
    assert len(tokens.unique(dim=0)) == 5
    assert torch.equal(tokens[4], tokens[5])
    assert not list(sender.parameters())
    all_tokens, _ = enumerate_messages(2, 6)
    weights = message_probabilities(sender(batch.sender_input()), all_tokens)
    assert weights.sum(-1).eq(1).all()
    assert ((weights == 0) | (weights == 1)).all()


@pytest.mark.parametrize('learner', ['fixed', 'exact'])
def test_exact_training_artifacts_and_finite_gradients(tmp_path, learner):
    config = Config(iterations=2, batch_size=12, evaluation_episodes=12,
                    evaluation_seed=999, learner=learner)
    result = train_one(config, 0, 2, tmp_path / learner, {'commit': 'test'})
    assert result['normal']['metrics']['count'] == 12
    record = torch.load(tmp_path / learner / 'rollouts' / '00001.pt', weights_only=True)
    assert record['message_probabilities'].shape == (12, 36)
    assert torch.allclose(record['message_probabilities'].sum(-1), torch.ones(12), atol=1e-6)
    assert torch.isfinite(record['expected_rewards']).all()


def test_audit_scenes_cover_all_private_states_orders_and_targets():
    from experiments.entity_safety.audit import exhaustive_episodes
    batch = exhaustive_episodes(layout_count=2)
    assert len(batch.target) == 288
    signatures = torch.cat((batch.identities, batch.cages.long(), batch.target[:, None]), -1)
    assert len(signatures[:144].unique(dim=0)) == 144
    for entity in range(3):
        for caged in (False, True):
            assert ((batch.target_entity == entity) & (batch.target_caged == caged)).sum() == 48


def test_interval_bounds_reject_constant_warning_even_if_no_unsafe_approaches():
    from experiments.entity_safety.audit import response_bounds
    sender = OracleSequenceTransmitter()
    receiver = SelectionMessageReceiver()
    with torch.no_grad():
        receiver.actor[-1].weight.zero_()
        receiver.actor[-1].bias.copy_(torch.tensor([0., 10.]))
    rows = response_bounds(sender, receiver)
    assert sum(row['passes'] for row in rows) == 2
    assert all(not row['passes'] for row in rows if row['expected_response'] == 'APPROACH')


def test_interval_bounds_enclose_observed_actor_outputs():
    from experiments.entity_safety.audit import response_bounds
    # A sufficiently strong constant APPROACH head has a positive certified
    # margin for all four safe cases and negative margins for dangerous cases.
    sender = OracleSequenceTransmitter()
    receiver = SelectionMessageReceiver()
    with torch.no_grad():
        receiver.actor[-1].weight.zero_()
        receiver.actor[-1].bias.copy_(torch.tensor([10., 0.]))
    rows = response_bounds(sender, receiver)
    assert sum(row['passes'] for row in rows) == 4
    assert all(row['conservative_margin'] == 10 for row in rows if row['passes'])


def test_position_bounds_detect_a_counterexample_outside_center_layouts():
    from experiments.entity_safety.audit import response_bounds

    class SafetySender(torch.nn.Module):
        n_symbols, message_length = 2, 1

        def forward(self, evidence):
            return torch.nn.functional.one_hot(evidence[:, -1].long(), 2).to(evidence.dtype)[:, None] * 10

    sender = SafetySender()
    receiver = SelectionMessageReceiver(message_length=1, n_symbols=2,
                                        embedding_dimension=1, hidden_dimension=1)
    with torch.no_grad():
        receiver.token_embeddings.weight.copy_(torch.tensor([[-1.], [1.]]))
        receiver.actor[0].weight.zero_()
        receiver.actor[0].weight[0, 0] = 3
        receiver.actor[0].weight[0, -2] = 4  # Position can overwhelm the correct token.
        receiver.actor[0].bias.zero_()
        receiver.actor[-1].weight.copy_(torch.tensor([[-1.], [1.]]))
        receiver.actor[-1].bias.zero_()
    assert not any(row['passes'] for row in response_bounds(sender, receiver))
    with torch.no_grad():
        receiver.actor[0].weight[0, -2] = .1
    assert all(row['passes'] for row in response_bounds(sender, receiver))
