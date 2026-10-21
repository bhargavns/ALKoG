"""Exact single-decision reward baseline; no PPO, critic or Gumbel surrogate.

Every receiver forward input is a discrete utterance. Sender gradients flow
through its probabilities of those utterances, never through token identities.
The objective is the expected shared 0/1 task reward, not message supervision.
"""
import itertools

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from .task import N_ENTITIES


def enumerate_messages(length, vocabulary):
    tokens = torch.tensor(list(itertools.product(range(vocabulary), repeat=length)))
    return tokens, F.one_hot(tokens, vocabulary).float()


def message_probabilities(logits, tokens):
    logp = logits.log_softmax(-1)
    selected = logp[:, None].expand(-1, len(tokens), -1, -1).gather(
        -1, tokens[None, :, :, None].expand(len(logits), -1, -1, -1))
    return selected.squeeze(-1).sum(-1).exp()


class FixedCodeSender(nn.Module):
    """Frozen arbitrary holistic code, ONLY a decoder learnability check.

    Five meanings map to five distinct whole utterances; safe sheep shares one
    code across cage states. No recombination rule or typed slots are supplied.
    """

    def __init__(self, message_length=2, n_symbols=6, code_seed=734):
        super().__init__()
        tokens, _ = enumerate_messages(message_length, n_symbols)
        meanings = 2 * N_ENTITIES - 1
        if len(tokens) < meanings:
            raise ValueError('Not enough distinct messages for fixed code')
        order = torch.randperm(len(tokens), generator=torch.Generator().manual_seed(code_seed))
        self.register_buffer('code', tokens[order[:meanings]])
        self.message_length, self.n_symbols = message_length, n_symbols

    def forward(self, evidence):
        meaning = evidence[:, :N_ENTITIES].argmax(-1) * 2 + evidence[:, -1].long()
        tokens = self.code[meaning]
        return torch.zeros(len(evidence), self.message_length, self.n_symbols).scatter(
            -1, tokens[:, :, None], 1).sub(1).mul(1e9)


def expected_reward(sender, receiver, episodes):
    logits = sender(episodes.sender_input())
    tokens, messages = enumerate_messages(sender.message_length, sender.n_symbols)
    weights = message_probabilities(logits, tokens)
    candidates = episodes.receiver_input()
    batch, choices = weights.shape
    dist, _ = receiver.distribution_messages(
        messages[None].expand(batch, -1, -1, -1).reshape(batch * choices, *messages.shape[1:]),
        candidates[:, None].expand(-1, choices, -1, -1).reshape(batch * choices, *candidates.shape[1:]))
    action_prob = dist.probs.reshape(batch, choices, -1)
    correct = episodes.target * 2 + episodes.dangerous.long()
    success = action_prob.gather(-1, correct[:, None, None].expand(-1, choices, -1)).squeeze(-1)
    return (weights * success).sum(-1), (tokens, weights, action_prob, success)


def exact_update(sender, receiver, channel, optimizer, episodes):
    expected, (tokens, weights, action_prob, success) = expected_reward(sender, receiver, episodes)
    with torch.no_grad():
        selected = Categorical(probs=weights).sample()
        probabilities = action_prob[torch.arange(len(selected)), selected]
        actions = Categorical(probs=probabilities).sample()
        logits = sender(episodes.sender_input())
        rollout = dict(identities=episodes.identities, positions=episodes.positions,
                       cages=episodes.cages, target=episodes.target,
                       sender_input=episodes.sender_input(),
                       messages=F.one_hot(tokens[selected], channel.n_symbols).float(),
                       actions=actions, rewards=episodes.reward(actions),
                       enumerated_tokens=tokens, message_probabilities=weights.detach(),
                       receiver_correct_action_probability=success.detach(),
                       expected_rewards=expected.detach(),
                       scope='exact_expectation_of_one_utterance_then_one_terminal_response')
        entropy = Categorical(logits=logits).entropy().mean().item()
        unique = len(logits.argmax(-1).unique(dim=0))
    loss = -expected.mean()
    optimizer.zero_grad()
    loss.backward()
    parameters = list(sender.parameters()) + list(receiver.parameters())
    torch.nn.utils.clip_grad_norm_(parameters, 0.5)
    optimizer.step()
    return dict(joint_accuracy=rollout['rewards'].mean().item(),
                expected_joint_accuracy=-loss.item(), loss=loss.item(),
                sender_entropy=entropy, unique_greedy_utterances=unique), rollout
