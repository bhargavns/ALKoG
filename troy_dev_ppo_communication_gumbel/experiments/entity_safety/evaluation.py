"""Paired frozen evaluation with per-meaning metrics and descriptive frequencies."""
import json
from collections import Counter

import torch
import torch.nn.functional as F

from .task import ENTITIES, N_CASES, SHEEP, matched_shuffle


def metrics(episodes, actions, mask=None):
    if mask is None:
        mask = torch.ones_like(actions, dtype=torch.bool)
    entity_correct = actions // 2 == episodes.target
    response_correct = actions % 2 == episodes.dangerous.long()
    danger_mask = mask & episodes.dangerous
    selected_entity = episodes.identities.gather(1, (actions // 2)[:, None]).squeeze(1)
    selected_caged = episodes.cages.gather(1, (actions // 2)[:, None]).squeeze(1)
    actual_unsafe = (selected_entity != SHEEP) & ~selected_caged & (actions % 2 == 0)
    def mean(value, subset):
        return value[subset].float().mean().item() if subset.any() else None
    return dict(count=int(mask.sum()), dangerous_count=int(danger_mask.sum()),
                joint_accuracy=mean(entity_correct & response_correct, mask),
                entity_accuracy=mean(entity_correct, mask),
                response_accuracy=mean(response_correct, mask),
                unsafe_approach_rate=mean(actions % 2 == 0, danger_mask),
                selected_entity_unsafe_approach_rate=mean(actual_unsafe, mask))


@torch.no_grad()
def evaluate(sender, receiver, channel, episodes, output, seed):
    sender.eval()
    receiver.eval()
    message = channel(sender(episodes.sender_input()), hard=True)
    shuffle = matched_shuffle(len(message), torch.Generator().manual_seed(seed))
    controls = {
        'normal': message,
        'no_message': torch.zeros_like(message),
        'constant': F.one_hot(torch.zeros(message.shape[:2], dtype=torch.long), channel.n_symbols).float(),
        'shuffled': message[shuffle],
    }
    with torch.random.fork_rng():
        torch.manual_seed(seed + 1)
        sampled_message = channel(sender(episodes.sender_input()))
        sampled_dist, _ = receiver.distribution_messages(sampled_message, episodes.receiver_input())
        sampled_actions = sampled_dist.sample()
    controls['sampled'] = sampled_message
    result = {}
    with (output / 'evaluation.jsonl').open('w') as log:
        for control, received in controls.items():
            dist, _ = receiver.distribution_messages(received, episodes.receiver_input())
            actions = sampled_actions if control == 'sampled' else dist.logits.argmax(-1)
            by_meaning = {}
            frequencies = {}
            for entity, name in enumerate(ENTITIES):
                for caged in (False, True):
                    subset = (episodes.target_entity == entity) & (episodes.target_caged == caged)
                    safety = 'dangerous' if entity != SHEEP and not caged else 'safe'
                    key = f'{name}/{safety}/{"caged" if caged else "loose"}'
                    by_meaning[key] = metrics(episodes, actions, subset)
                    frequencies[key] = {
                        'utterances': dict(Counter(','.join(map(str, row)) for row in
                                                  received[subset].argmax(-1).tolist())) if control != 'no_message' else {},
                        'position_token_counts': received[subset].sum(0).int().tolist(),
                    }
            result[control] = dict(metrics=metrics(episodes, actions), by_meaning=by_meaning,
                                   message_frequencies=frequencies)
            rewards = episodes.reward(actions)
            target_entities = episodes.target_entity
            target_danger = episodes.dangerous
            for i in range(len(actions)):
                log.write(json.dumps(dict(
                    episode=i, layout=i // N_CASES, control=control,
                    candidate_entities=[ENTITIES[x] for x in episodes.identities[i].tolist()],
                    positions=episodes.positions[i].tolist(), cages=episodes.cages[i].tolist(),
                    target_candidate=int(episodes.target[i]),
                    target_entity=ENTITIES[int(target_entities[i])],
                    dangerous=bool(target_danger[i]),
                    message=None if control == 'no_message' else received[i].argmax(-1).tolist(),
                    message_source_episode=int(shuffle[i]) if control == 'shuffled' else i,
                    selected_candidate=int(actions[i] // 2),
                    response='WARN' if actions[i] % 2 else 'APPROACH',
                    reward=float(rewards[i]),
                )) + '\n')
    sender.train()
    receiver.train()
    return result
