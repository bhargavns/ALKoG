"""Run Experiment 1 sequentially across seeds and message lengths (CPU only)."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from communication.CommunicationChannel import StraightThroughGumbelChannel
from communication.ReceiverPolicy import SelectionMessageReceiver
from communication.SymbolPolicy import OracleSequenceTransmitter
from .evaluation import evaluate
from .task import ENTITIES, N_ENTITIES, N_CASES, sample_episodes
from .trainer import update
from .exact import FixedCodeSender, exact_update


@dataclass(frozen=True)
class Config:
    iterations: int = 1000
    batch_size: int = 60
    evaluation_episodes: int = 600
    epochs: int = 4
    learning_rate: float = 0.001
    entropy_coefficient: float = 0.01
    clip_ratio: float = 0.2
    temperature: float = 1.2
    hidden_dimension: int = 64
    embedding_dimension: int = 8
    n_symbols: int = 6
    evaluation_seed: int = 2147483645
    learner: str = 'gumbel'

    def validate(self):
        if self.learner not in ('gumbel', 'exact', 'fixed'):
            raise ValueError('Unknown learner')
        if self.iterations < 1 or self.epochs < 1:
            raise ValueError('Iterations and epochs must be positive')
        if any(n < N_CASES or n % N_CASES for n in (self.batch_size, self.evaluation_episodes)):
            raise ValueError(f'Batch/evaluation sizes must be positive multiples of {N_CASES}')
        if self.learning_rate <= 0 or self.entropy_coefficient < 0:
            raise ValueError('Positive learning rate and nonnegative entropy required')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_metadata(output):
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True)
    commit = git('rev-parse', 'HEAD').strip()
    # Include tracked edits AND newly created experiment/test sources in provenance.
    (output / 'source.patch').write_text(git('diff', 'HEAD'))
    paths = [Path(__file__).resolve().parent,
             root / 'troy_dev_ppo_communication_gumbel' / 'communication',
             root / 'troy_dev_ppo_communication_gumbel' / 'representation',
             root / 'troy_dev_ppo_communication_gumbel' / 'tests' / 'experiments']
    hashes = {}
    for folder in paths:
        for path in folder.rglob('*.py'):
            relative = path.relative_to(root)
            content = path.read_bytes()
            hashes[str(relative)] = hashlib.sha256(content).hexdigest()
            snapshot = output / 'source' / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(content)
    return dict(commit=commit, status=git('status', '--short'), sha256=hashes,
                torch_version=str(torch.__version__), surrogate='biased_straight_through_gumbel_PPO',
                decision_scope='complete_utterance_then_one_terminal_joint_action')


def train_one(config, seed, length, output, source):
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    sender = OracleSequenceTransmitter(input_dimension=N_ENTITIES + 2, message_length=length, n_symbols=config.n_symbols,
                                      hidden_dimension=config.hidden_dimension)
    if config.learner == 'fixed':
        sender = FixedCodeSender(length, config.n_symbols)
    receiver = SelectionMessageReceiver(candidate_dimension=N_ENTITIES + 2, message_length=length, n_symbols=config.n_symbols,
                                       embedding_dimension=config.embedding_dimension,
                                       hidden_dimension=config.hidden_dimension)
    channel = StraightThroughGumbelChannel(config.temperature, config.n_symbols)
    optimizer = torch.optim.Adam(list(sender.parameters()) + list(receiver.parameters()),
                                 lr=config.learning_rate)
    exact = config.learner in ('exact', 'fixed')
    run_config = dict(**asdict(config), seed=seed, message_length=length, device='cpu',
                      reward_access='full_expected_reward' if exact else 'sampled_action_reward',
                      optimizer_steps_per_batch=1 if exact else config.epochs,
                      task='single_decision_oracle_selection',
                      fixed_code_seed=734 if config.learner == 'fixed' else None)
    write_json(output / 'config.json', run_config)
    write_json(output / 'split_manifest.json', dict(
        entities=ENTITIES, train_meanings=[(e, c) for e in ENTITIES for c in ('loose', 'caged')],
        excluded_target_meanings=[], training_layout_seed=seed,
        evaluation_layout_seed=config.evaluation_seed,
        split='independent continuous random layouts; all meanings in training and evaluation',
        oracle_safety_rule='predators dangerous iff loose; sheep always safe',
        receiver_features=['candidate_category_one_hot', 'candidate_xy'],
        evaluation_matched_group_size=N_CASES))
    rollouts = output / 'rollouts'
    rollouts.mkdir()
    start = time.perf_counter()
    with (output / 'learning_curve.jsonl').open('w') as curve:
        for iteration in range(1, config.iterations + 1):
            episodes = sample_episodes(config.batch_size, generator)
            if config.learner in ('exact', 'fixed'):
                stats, rollout = exact_update(sender, receiver, channel, optimizer, episodes)
            else:
                noise = channel.sample_noise((config.batch_size, length, config.n_symbols), 'cpu')
                stats, rollout = update(sender, receiver, channel, optimizer, episodes, noise,
                                        epochs=config.epochs, clip_ratio=config.clip_ratio,
                                        entropy_coefficient=config.entropy_coefficient)
            stats.update(iteration=iteration, training_contexts=iteration * config.batch_size,
                         elapsed_seconds=time.perf_counter() - start)
            curve.write(json.dumps(stats) + '\n')
            torch.save(rollout, rollouts / f'{iteration:05d}.pt')
            if iteration == 1 or iteration % 100 == 0 or iteration == config.iterations:
                curve.flush()
                print(f'L={length} seed={seed} iteration={iteration} accuracy={stats["joint_accuracy"]:.3f}', flush=True)
    training_seconds = time.perf_counter() - start
    checkpoint = dict(sender=sender.state_dict(), receiver=receiver.state_dict(),
                      optimizer=optimizer.state_dict(), config=run_config, source=source,
                      torch_rng_state=torch.get_rng_state(), layout_rng_state=generator.get_state())
    torch.save(checkpoint, output / 'checkpoint.pt')
    episodes = sample_episodes(config.evaluation_episodes,
                               torch.Generator().manual_seed(config.evaluation_seed), matched_layouts=True)
    torch.save(dict(identities=episodes.identities, positions=episodes.positions,
                    cages=episodes.cages, target=episodes.target), output / 'evaluation_episodes.pt')
    result = evaluate(sender, receiver, channel, episodes, output, config.evaluation_seed + 1)
    result['computation'] = dict(training_seconds=training_seconds,
                                 sender_parameters=sum(p.numel() for p in sender.parameters()),
                                 receiver_parameters=sum(p.numel() for p in receiver.parameters()))
    write_json(output / 'metrics.json', result)
    return result


def aggregate(results):
    """Seed-level, equally weighted estimates, including each entity/cage case."""
    def summarize(rows):
        output = {}
        for key in rows[0]:
            values = [row[key] for row in rows if row[key] is not None]
            output[key] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                               std=statistics.stdev(values) if len(values) > 1 else None)
        return output
    output = {}
    for length, runs in results.items():
        output[str(length)] = {}
        for control in ('normal', 'no_message', 'constant', 'shuffled', 'sampled'):
            output[str(length)][control] = dict(
                metrics=summarize([r[control]['metrics'] for r in runs]),
                by_meaning={key: summarize([r[control]['by_meaning'][key] for r in runs])
                            for key in runs[0][control]['by_meaning']})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--lengths', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    parser.add_argument('--learner', choices=['gumbel', 'exact', 'fixed'], default='gumbel')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=60)
    parser.add_argument('--evaluation-episodes', type=int, default=600)
    parser.add_argument('--evaluation-seed', type=int, default=2147483645)
    args = parser.parse_args()
    config = Config(iterations=args.iterations, batch_size=args.batch_size,
                    evaluation_episodes=args.evaluation_episodes, evaluation_seed=args.evaluation_seed, learner=args.learner)
    config.validate()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.lengths)) != len(args.lengths):
        parser.error('Seeds and lengths must be unique')
    if config.evaluation_seed in args.seeds:
        parser.error('Training seeds must differ from the reserved evaluation seed')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_metadata(args.output)
    source['learner'] = config.learner
    source['surrogate'] = ('biased_straight_through_gumbel_PPO' if config.learner == 'gumbel'
                           else 'exact_expected_joint_reward_over_discrete_messages_and_actions')
    write_json(args.output / 'source.json', source)
    write_json(args.output / 'study.json', dict(config=asdict(config), seeds=args.seeds,
               lengths=args.lengths, status='exploratory' if len(args.seeds) < 5 else 'five_or_more_seeds',
               comparisons='paired layout seeds and interaction budgets; parameter counts reported separately'))
    results = {}
    for length in args.lengths:
        results[length] = [train_one(config, seed, length, args.output / f'L{length}_seed{seed}', source)
                           for seed in args.seeds]
    write_json(args.output / 'aggregate.json', aggregate(results))


if __name__ == '__main__':
    main()
