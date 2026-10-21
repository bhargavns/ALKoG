"""Audit frozen checkpoints against the functional safety-response criterion.

Entity selection remains a diagnostic. The acceptance criterion is the response
for the sender's target, not safe physical execution at an arbitrary selected
entity. Deployment uses deterministic argmax messages and responses.
"""
import argparse
import itertools
from pathlib import Path

import torch
from torch import nn

from communication.CommunicationChannel import StraightThroughGumbelChannel
from communication.ReceiverPolicy import SelectionMessageReceiver
from communication.SymbolPolicy import OracleSequenceTransmitter
from .run import source_metadata, write_json
from .task import ENTITIES, N_ENTITIES, N_CASES, Episodes, sample_episodes


def load_pair(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    sender = OracleSequenceTransmitter(input_dimension=N_ENTITIES + 2,
                                      message_length=config['message_length'],
                                      n_symbols=config['n_symbols'],
                                      hidden_dimension=config['hidden_dimension'])
    receiver = SelectionMessageReceiver(candidate_dimension=N_ENTITIES + 2,
                                        message_length=config['message_length'],
                                        n_symbols=config['n_symbols'],
                                        embedding_dimension=config['embedding_dimension'],
                                        hidden_dimension=config['hidden_dimension'])
    sender.load_state_dict(checkpoint['sender'])
    receiver.load_state_dict(checkpoint['receiver'])
    sender.eval()
    receiver.eval()
    return sender, receiver, config


@torch.no_grad()
def response_bounds(sender, receiver, coordinate_limit=1.):
    """Conservative interval bounds for every coordinate in [-limit, limit].

    The candidate actor is Linear -> Tanh -> Linear. Fixed token/category
    features determine hidden centers. Absolute position weights bound hidden
    intervals, tanh is monotone, and signed output weights propagate bounds.
    At least one correct-response score must beat EVERY wrong-response score.
    Shared candidate scoring makes this independent of candidate permutation.
    This is floating-point interval analysis, not a machine-verified proof.
    """
    if coordinate_limit <= 0:
        raise ValueError('Positive coordinate limit required')
    if (len(receiver.actor) != 3 or not isinstance(receiver.actor[0], nn.Linear)
            or not isinstance(receiver.actor[1], nn.Tanh)
            or not isinstance(receiver.actor[2], nn.Linear)
            or receiver.candidate_dimension != N_ENTITIES + 2):
        raise ValueError('Unsupported actor architecture')
    # Work on copies so auditing does not change deployment precision.
    import copy
    sender, receiver = copy.deepcopy(sender).double(), copy.deepcopy(receiver).double()
    batch = sample_episodes(N_CASES, torch.Generator().manual_seed(0), matched_layouts=True)
    channel = StraightThroughGumbelChannel(n_symbols=sender.n_symbols)
    messages = channel(sender(batch.sender_input().double()), hard=True)
    embeddings = (messages @ receiver.token_embeddings.weight).flatten(1)
    candidates = torch.cat((torch.eye(N_ENTITIES, dtype=torch.float64),
                            torch.zeros(N_ENTITIES, 2, dtype=torch.float64)), -1)
    features = torch.cat((embeddings[:, None].expand(-1, N_ENTITIES, -1),
                          candidates[None].expand(N_CASES, -1, -1)), -1)
    first, _, last = receiver.actor
    centers = first(features)
    radii = first.weight[:, -2:].abs().sum(-1) * coordinate_limit
    lower_hidden, upper_hidden = (centers - radii).tanh(), (centers + radii).tanh()
    positive, negative = last.weight.clamp(min=0), last.weight.clamp(max=0)
    lower = lower_hidden @ positive.T + upper_hidden @ negative.T + last.bias
    upper = upper_hidden @ positive.T + lower_hidden @ negative.T + last.bias
    response = batch.dangerous.long()
    good = lower.gather(-1, response[:, None, None].expand(-1, N_ENTITIES, 1)).squeeze(-1).max(-1).values
    bad = upper.gather(-1, (1 - response)[:, None, None].expand(-1, N_ENTITIES, 1)).squeeze(-1).max(-1).values
    rows = []
    for i in range(N_CASES):
        margin = float(good[i] - bad[i])
        rows.append(dict(entity=ENTITIES[i // 2], caged=bool(i % 2),
                         expected_response='WARN' if response[i] else 'APPROACH',
                         message=messages[i].argmax(-1).tolist(),
                         conservative_margin=margin,
                         passes=bool(torch.isfinite(good[i]) & torch.isfinite(bad[i])) and margin > 1e-4))
    return rows


def exhaustive_episodes(layout_count=100, seed=3001):
    """All 6 orders x 8 cage assignments x 3 targets for every public layout."""
    generator = torch.Generator().manual_seed(seed)
    base = list(itertools.product(itertools.permutations(range(N_ENTITIES)),
                                  itertools.product((False, True), repeat=N_ENTITIES),
                                  range(N_ENTITIES)))
    identities = torch.tensor([order for order, _, _ in base]).repeat(layout_count, 1)
    cages = torch.tensor([states for _, states, _ in base]).repeat(layout_count, 1)
    targets = torch.tensor([target for _, _, target in base]).repeat(layout_count)
    coordinates = torch.rand(layout_count, N_ENTITIES, 2, generator=generator) * 2 - 1
    positions = coordinates.repeat_interleave(len(base), 0)
    return Episodes(identities, positions, cages, targets)


@torch.no_grad()
def audit_pair(sender, receiver, episodes):
    channel = StraightThroughGumbelChannel(n_symbols=sender.n_symbols)
    logits = sender(episodes.sender_input())
    message = channel(logits, hard=True)
    controls = dict(normal=message, no_message=torch.zeros_like(message),
                    constant=torch.nn.functional.one_hot(
                        torch.zeros(message.shape[:2], dtype=torch.long), sender.n_symbols).float())
    results, records = {}, {}
    for label, received in controls.items():
        # Chunked inference bounds RAM independently of the audit size.
        actions, probabilities = [], []
        for start in range(0, len(message), 1024):
            distribution, _ = receiver.distribution_messages(
                received[start:start + 1024], episodes.receiver_input()[start:start + 1024])
            actions.append(distribution.logits.argmax(-1))
            probabilities.append(distribution.probs)
        actions, probabilities = torch.cat(actions), torch.cat(probabilities)
        correct = actions % 2 == episodes.dangerous.long()
        unsafe = episodes.dangerous & (actions % 2 == 0)
        by_case = {}
        for entity, name in enumerate(ENTITIES):
            for caged in (False, True):
                mask = (episodes.target_entity == entity) & (episodes.target_caged == caged)
                by_case[f'{name}/{"caged" if caged else "loose"}'] = dict(
                    count=int(mask.sum()), errors=int((~correct & mask).sum()))
        results[label] = dict(count=len(actions), safety_errors=int((~correct).sum()),
                              safety_accuracy=correct.float().mean().item(),
                              unsafe_approaches=int(unsafe.sum()),
                              target_identification_accuracy=(actions // 2 == episodes.target).float().mean().item(),
                              by_case=by_case)
        records[label] = dict(actions=actions, action_probabilities=probabilities,
                              messages=received, safety_correct=correct)
    rows = response_bounds(sender, receiver)
    passed = (results['normal']['safety_errors'] == 0
              and all(row['passes'] for row in rows)
              and results['no_message']['safety_accuracy'] <= 2 / 3 + 1e-6
              and results['constant']['safety_accuracy'] <= 2 / 3 + 1e-6)
    return dict(accepted=passed, response_bounds=rows, evaluations=results), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('output/entity_safety3_exact'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layouts', type=int, default=100)
    parser.add_argument('--seed', type=int, default=3001)
    args = parser.parse_args()
    if args.layouts < 1:
        parser.error('Positive layout count required')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'source.json', source_metadata(args.output))
    episodes = exhaustive_episodes(args.layouts, args.seed)
    torch.save(dict(identities=episodes.identities, positions=episodes.positions,
                    cages=episodes.cages, target=episodes.target), args.output / 'episodes.pt')
    summary = dict(criterion='target safety response; identification reported separately',
                   deployment='deterministic argmax tokens and joint-action response',
                   training_objective='unchanged joint entity/response reward in existing checkpoints',
                   scope='three oracle categories, one candidate each, coordinates in [-1,1]',
                   warning='does not certify physical navigation, perception, safety inference or composition',
                   audit_seed=args.seed, layouts=args.layouts, runs={})
    for seed in range(5):
        checkpoint_path = args.runs / f'L2_seed{seed}' / 'checkpoint.pt'
        sender, receiver, _ = load_pair(checkpoint_path)
        result, records = audit_pair(sender, receiver, episodes)
        result['checkpoint'] = str(checkpoint_path.resolve())
        import hashlib
        result['checkpoint_sha256'] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        torch.save(records, args.output / f'seed{seed}_records.pt')
        summary['runs'][str(seed)] = result
        print(f'seed={seed} accepted={result["accepted"]} safety_errors={result["evaluations"]["normal"]["safety_errors"]}', flush=True)
    summary['accepted'] = all(run['accepted'] for run in summary['runs'].values())
    write_json(args.output / 'acceptance.json', summary)


if __name__ == '__main__':
    main()
