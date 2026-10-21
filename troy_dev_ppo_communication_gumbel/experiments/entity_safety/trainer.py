"""One utterance / one joint action PPO diagnostic.

Replay reconstructs each full utterance from oracle inputs and saved [B,L,V]
Gumbel noise. This is the existing biased straight-through surrogate, NOT exact
marginal categorical-message PPO. Shared reward is joint correctness only.
"""
import torch


def decision(sender, receiver, channel, episodes, noise):
    logits = sender(episodes.sender_input())
    messages = channel(logits, noise)
    distribution, values = receiver.distribution_messages(messages, episodes.receiver_input())
    return distribution, values, logits, messages


def update(sender, receiver, channel, optimizer, episodes, noise, *, epochs=4,
           clip_ratio=0.2, entropy_coefficient=0.01):
    with torch.no_grad():
        dist, values, _, messages = decision(sender, receiver, channel, episodes, noise)
        actions = dist.sample()
        rewards = episodes.reward(actions)
        old_logp = dist.log_prob(actions)
        advantage = rewards - values
        advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    # Detached rollout records are the entire one-decision scope, not one record
    # per token. Save before optimization; replayed tokens can change on updates.
    rollout = dict(identities=episodes.identities, positions=episodes.positions,
                   cages=episodes.cages, target=episodes.target,
                   sender_input=episodes.sender_input(), noise=noise,
                   messages=messages.detach(), actions=actions, rewards=rewards,
                   old_log_prob=old_logp, old_value=values,
                   scope='one_complete_utterance_then_one_terminal_response')
    parameters = list(sender.parameters()) + list(receiver.parameters())
    for _ in range(epochs):
        dist, value, _, _ = decision(sender, receiver, channel, episodes, noise)
        ratio = (dist.log_prob(actions) - old_logp).exp()
        objective = torch.minimum(ratio * advantage,
                                  ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
        loss = (-objective.mean() + 0.5 * (value - rewards).square().mean()
                - entropy_coefficient * dist.entropy().mean())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 0.5)
        optimizer.step()
    return {'joint_accuracy': rewards.mean().item(), 'loss': loss.item()}, rollout
