"""Joint PPO with discrete Gumbel messages and a biased straight-through gradient.

Each decision stores scene inputs and parameter-independent Gumbel noise.
Replay recomputes messages with current weights; PPO ratios condition on this
noise and are not exact marginal categorical-message policy gradients.
Episode time limits are finite-task terminals. Warm starts restore weights only.
"""

import numpy as np
import torch

from communication.CommunicationChannel import StraightThroughGumbelChannel
from representation.CommunicationCadence import perception_due
from communication.SenderEntropy import entropy_coefficient, sender_entropy

# communication/train_communication.py:
#   initialize() -> train(settings) -> construct GumbelPPOTrainer
#   for each iteration: set trainer.iteration -> collect_and_update()
#
# This file's training cycle:
#   collect_rollout() -> _new_record() -> _decision() -> evaluate_records()
#     -> SymbolPolicy.GumbelTransmitter.forward(): scene rows/pose -> symbol logits
#     -> CommunicationChannel.StraightThroughGumbelChannel.forward(): discrete message
#     -> ReceiverPolicy.CurrentMessageReceiverActorCritic.distribution_messages():
#          action distribution + value estimate
#     -> environment/KGWorldEnv.py: CommunicationKGWorldEnv.step(): movement/reward
#   _gae(): completed rollout -> advantages and return targets
#   update() -> evaluate_records() again with gradients -> optimizer.step()
#
# Scene inputs enter through representation/TransmitterInput.py or
# representation/OracleTransmitterInput.py. The visual adapter invokes
# perception/ScenePerception.py; oracle inputs bypass image inference.
#
# Outputs return to train_communication.train(): metrics go to
# observability/Diagnostics.py, and updated policies are saved/recorded there.
# Standalone evaluation and videos use evaluation/CommunicationEvaluation.py's
# CommunicationActor/evaluate_episode(), not this trainer's collect/update loop.


# Parameters (B = batch size, K = scene rows, V = message categories):
#   records: List of individual _new_record() dictionaries to stack into the batched evaluate_records() format.
#   device: PyTorch device for policy execution/tensor storage, e.g. cpu or cuda:0.
def pack_records(records, device):
    # Called by _decision() for a single environment step and by collect_rollout()
    # for the whole rollout. Both produce the input dictionary evaluate_records()
    # expects; update() later selects minibatches from the whole-rollout dictionary.
    # Stack decisions along a batch axis: obs [B,4], triples [B,K,3],
    # deltas [B,K,2,2], noise [B,V]. K includes the reserved receiver row;
    # V is the communication vocabulary size. Record IDs are diagnostic only.
    return {field_name: torch.stack([record[field_name] for record in records]).to(device)
            for field_name in ('obs', 'triples', 'deltas', 'noise')}


# Parameters (B = batch size, K = scene rows, V = message categories):
#   transmitter: GumbelTransmitter: scene pose/rows/offsets -> categorical message logits.
#   receiver: CurrentMessageReceiverActorCritic: one message -> action distribution and value.
#   channel: StraightThroughGumbelChannel: logits plus saved noise -> discrete differentiable message.
#   records: Batched tensors: obs [B,4], triples [B,K,3], deltas [B,K,2,2], noise [B,V].
#   detach_channel: If True, cut receiver-loss gradients into the sender while preserving message values.
def evaluate_records(transmitter, receiver, channel, records, *, detach_channel=False):
    """Reconstruct each decision's categorical message using its original noise."""
    # Called without gradients by _decision() while playing episodes, then WITH
    # gradients by update() while learning from those same decisions. "evaluate"
    # here means a policy forward pass, not a separate held-out evaluation run.
    # transmitter invokes SymbolPolicy.GumbelTransmitter.forward(), which encodes triples
    # with SymbolTable and pools them with SlotAttention before producing logits.
    sender_logits = transmitter(records['obs'], records['triples'], records['deltas'])
    # Replay the random draw, not a cached message activation. Updated sender
    # weights can change the selected symbol even though the noise stays fixed.
    messages = channel(sender_logits, records['noise'])
    # This experimental control preserves the forward message but disconnects
    # both receiver loss paths from the sender. Receiver parameters still train.
    if detach_channel:
        messages = messages.detach()
    # ReceiverPolicy.distribution_messages() consumes one [B,V] message batch.
    # collect_rollout() samples actions from dist; update() evaluates saved actions
    # under dist. The receiver critic's value feeds both _gae() and value loss.
    action_distribution, value_estimates = receiver.distribution_messages(messages)
    return action_distribution, value_estimates, sender_logits, messages


class GumbelPPOTrainer:
    """Joint PPO for a categorical sender and a current-message receiver."""

    # Parameters:
    #   transmitter: GumbelTransmitter: scene pose/rows/offsets -> categorical message logits.
    #   receiver: CurrentMessageReceiverActorCritic: one message -> action distribution and value.
    #   environment: Simulation instance used for the training rollout or the separate recorded episode.
    #   make_transmitter_input: Zero-argument factory from train_communication.create_transmitter_input; owns fresh episode memory.
    #   device: PyTorch device for policy execution/tensor storage, e.g. cpu or cuda:0.
    #   horizon: Environment steps per rollout; episode boundaries do not end collection.
    #   gamma: Future-reward discount factor used by generalized advantage estimation.
    #   advantage_trace_decay: GAE lambda: discount on accumulated temporal-difference errors, usually in [0,1].
    #   clip_ratio: PPO probability-ratio clipping width around 1, e.g. 0.2.
    #   epochs: Optimization passes over each collected rollout.
    #   minibatch_size: Maximum number of saved decisions in one optimizer step.
    #   learning_rate: Adam learning rate shared by transmitter and receiver.
    #   receiver_entropy_coefficient: Weight of the receiver action-entropy bonus in the joint loss.
    #   value_loss_coefficient: Weight of the receiver critic squared-error loss.
    #   max_gradient_norm: Clipping threshold for the combined transmitter/receiver gradient norm.
    #   perceive_passes: Opening scene refresh passes in stationary mode; colocated cadence is defined separately.
    #   record_trajectories: If True, retain completed episode paths for plotting by train().
    #   normalize_value_loss: If True, divide critic errors by rollout return standard deviation.
    #   target_kl: Approximate action-policy KL threshold for early stopping; <=0 disables the guard.
    #   detach_channel: If True, cut receiver-loss gradients into the sender while preserving message values.
    #   temperature: Positive Gumbel surrogate temperature; changes gradients, not discrete sampling probabilities.
    #   n_symbols: Number of categories in the communication vocabulary (V), at least two.
    #   sender_entropy_beta: Nonnegative initial sender-entropy bonus coefficient; zero disables it.
    #   sender_entropy_schedule: constant keeps the bonus fixed; annealed decays it across the training budget.
    #   total_iterations: Full iteration budget used to locate the current point in the entropy schedule.
    def __init__(self, transmitter, receiver, environment, make_transmitter_input, device='cuda',
                 horizon=2048, gamma=0.99, advantage_trace_decay=0.95, clip_ratio=0.2,
                 epochs=10, minibatch_size=256, learning_rate=3e-4, receiver_entropy_coefficient=0.01,
                 value_loss_coefficient=0.5, max_gradient_norm=0.5, perceive_passes=1,
                 record_trajectories=False, normalize_value_loss=True,
                 target_kl=0.03, detach_channel=False,
                 temperature=1.2, n_symbols=16,
                 sender_entropy_beta=0., sender_entropy_schedule="annealed", total_iterations=800):
        if horizon < 2 or epochs < 1 or minibatch_size < 1 or perceive_passes < 1:
            raise ValueError('horizon >= 2 and positive epochs/minibatch/perceive_passes required')
        # Constructed in communication/train_communication.py:train(), after the
        # policies and environment exist and any warm-start weights have loaded.
        # make_transmitter_input is a bound create_transmitter_input() factory from that file;
        # it returns representation.TransmitterInput or OracleTransmitterInput.
        # This instance shares the policies with train(), but owns the training
        # adapter's episode memory. Optional video recording uses another adapter.
        self.transmitter, self.receiver = transmitter.to(device), receiver.to(device)
        self.environment, self.transmitter_input = environment, make_transmitter_input()
        self.device, self.horizon = device, horizon
        # gamma discounts future rewards; gae_lambda controls how far advantage
        # estimates accumulate temporal-difference errors. clip_ratio bounds the
        # policy-ratio term in PPO, not the raw model weights or gradients.
        self.gamma, self.advantage_trace_decay, self.clip_ratio = gamma, advantage_trace_decay, clip_ratio
        # horizon counts environment steps per rollout. epochs counts optimization
        # passes over that same rollout; each pass consists of several minibatches.
        self.epochs, self.minibatch_size = epochs, minibatch_size
        # receiver_entropy_coefficient weights RECEIVER action entropy; the separate sender coefficient
        # below weights uncertainty over communication symbols. value_loss_coefficient scales the
        # critic's squared prediction error relative to the policy objective.
        self.receiver_entropy_coefficient, self.value_loss_coefficient = receiver_entropy_coefficient, value_loss_coefficient
        self.max_gradient_norm, self.target_kl = max_gradient_norm, target_kl
        self.perceive_passes = perceive_passes
        self.record_trajectories = record_trajectories
        self.normalize_value_loss = normalize_value_loss
        self.n_symbols = n_symbols
        # A detached-channel control should leave the sender fixed. A nonzero
        # sender entropy objective would otherwise still update it directly.
        if detach_channel and sender_entropy_beta:
            raise ValueError('Detached control cannot have a sender entropy objective')
        entropy_coefficient(1, total_iterations, sender_entropy_beta, sender_entropy_schedule)
        self.sender_entropy_beta = sender_entropy_beta
        self.sender_entropy_schedule = sender_entropy_schedule
        self.total_iterations = total_iterations
        # train_communication.train() assigns trainer.iteration = it immediately
        # before collect_and_update(). update() passes this index to
        # communication/SenderEntropy.py:entropy_coefficient(); this class does
        # not advance the outer loop's iteration counter itself.
        self.iteration = 1
        self.channel = StraightThroughGumbelChannel(temperature, n_symbols)
        self.detach_channel = detach_channel
        # One optimizer updates both agents. Parameter groups identify ownership;
        # both groups use the same learning rate. No optimizer state is restored here.
        self.optimizer = torch.optim.Adam([
            {'params': self.transmitter.parameters(), 'name': 'transmitter'},
            {'params': self.receiver.parameters(), 'name': 'receiver'},
        ], lr=learning_rate)
        # Episode state persists across rollout/update boundaries. _pending holds
        # the next decision's inputs/noise, prepared for end-of-rollout bootstrap.
        # An episode reset clears it; an ordinary PPO update does not reset the env.
        self._observation, self._pending_record = None, None
        self._record_id = 0
        self._episode_id = -1
        self._trajectory = None
        self.trajectories = []
        self.last_symbols = self.last_actions = None

    def _reset_env(self):
        # Called only by collect_rollout(): on its first entry and after terminal
        # transitions. environment/KGWorldEnv.py resets the layout/agent state.
        # TransmitterInput.reset() clears representation/Grounding.py's AnchorMemory;
        # refresh() then invokes the ScenePerception callback and assembles triples.
        # OracleTransmitterInput.refresh() instead constructs exact simulator rows.
        # Neither train()'s iteration boundary nor update() triggers this reset.
        observation, _ = self.environment.reset()
        self._observation = observation
        self.transmitter_input.reset()
        self.transmitter_input.refresh(self._observation)
        self._episode_return = 0.0
        self._episode_length = 0
        self._pending_record = None
        self._episode_id += 1
        # Optional paths support plots only; they are not the PPO training buffer.
        # A trajectory remains open if its episode spans multiple rollouts.
        if self.record_trajectories:
            self._trajectory = {
                "positions": [self.environment.receiver_xy.tolist()],
                "transmitter": self.environment.data.qpos[0:2].copy().tolist(),
                "lion": self.environment.model.body("lion").pos[:2].copy().tolist(),
                "food": self.environment.model.body("food").pos[:2].copy().tolist(),
                "cage": self.environment.model.body("cage").pos[:2].copy().tolist(),
                "lion_caged": bool(self.environment.lion_caged),
                "symbols": [],
            }

    # Parameters:
    #   info: Environment outcome dictionary returned by step(), including food/lion termination indicators.
    #   truncated: Whether the environment ended the episode at its time limit.
    def _finish_trajectory(self, info, truncated):
        # collect_rollout() calls this just before _reset_env() on episode end.
        # train_communication.train() drains self.trajectories after each update,
        # retains the latest 200 paths, and passes them to
        # observability/Diagnostics.py:save_trajectory_plot() when training finishes.
        # This plotting label distinguishes food/death/timeout; other terminal
        # outcomes use "end" here. These paths are separate from PPO records.
        if self._trajectory is None:
            return
        if info.get("food_reached", False):
            outcome = "food"
        elif info.get("lion_caught", False):
            outcome = "death"
        else:
            outcome = "timeout" if truncated else "end"
        self._trajectory["outcome"] = outcome
        self._trajectory["episode_return"] = float(self._episode_return)
        self.trajectories.append(self._trajectory)
        self._trajectory = None

    def _new_record(self):
        # Called by collect_rollout() for each decision and once more to prepare
        # the rollout-boundary bootstrap. Its record goes to _decision() now and
        # to evaluate_records() inside update() later, without rerunning perception.
        # TransmitterInput.deltas() (also inherited by OracleTransmitterInput)
        # calls Grounding.deltas_from_anchors() to express positions forward/left
        # relative to the transmitter. CommunicationChannel.sample_noise() supplies
        # the per-symbol random draw used both now and during optimization.
        # Snapshot the exact input BEFORE taking an action. clone() prevents later
        # environment/input-adapter changes from altering stored rollout evidence.
        # deltas() also refreshes the moving receiver's position in its reserved row.
        # Gumbel noise is independent of model weights, so it can be reused in PPO.
        record = {
            'id': self._record_id, 'episode': self._episode_id, 'step': self._episode_length,
            'obs': torch.as_tensor(self._observation, dtype=torch.float32, device=self.device).clone(),
            'triples': self.transmitter_input.triples.to(self.device).clone(),
            'deltas': self.transmitter_input.deltas(self._observation, self.environment).to(self.device).clone(),
            'noise': self.channel.sample_noise((self.n_symbols,), self.device),
        }
        self._record_id += 1
        return record

    # Parameters:
    #   record: One saved decision from _new_record(): pose, triples, offsets, Gumbel noise, and diagnostic IDs.
    @torch.no_grad()
    def _decision(self, record):
        # collect_rollout() calls this before env.step() and for its final bootstrap
        # value. The path is pack_records() -> evaluate_records() -> sender/channel/
        # receiver. update() calls evaluate_records() directly to enable gradients.
        # Environment interaction needs predictions, not an autograd graph. Replay
        # builds a fresh graph later. Use a batch of one; logits/message are unbatched
        # on return, while the distribution and value retain their batch dimension.
        action_distribution, value_estimates, sender_logits, messages = evaluate_records(
            self.transmitter, self.receiver, self.channel, pack_records([record], self.device))
        return action_distribution, value_estimates, sender_logits[0], messages[0]

    def collect_rollout(self):
        # First half of collect_and_update(), called once per iteration of
        # train_communication.train(). THIS is where training episodes are played:
        # collect exactly settings.horizon steps, return a batch for update() and
        # metrics for train() to log. One call can finish several episodes or stop
        # partway through one; only the very first call resets on entry.
        if self._observation is None:
            self._reset_env()
        # Dropout is zero. Keep the same module mode for collection and replay.
        self.transmitter.train()
        self.receiver.train()
        # Lists are local to this rollout. Stored log-probabilities and values come
        # from the behavior policy (the weights before this iteration's update).
        decision_records = []
        actions, action_log_probabilities, rollout_values, rewards, terminal_flags, symbols = [], [], [], [], [], []
        rollout_logits, rollout_soft_messages, episodes = [], [], []
        for _ in range(self.horizon):
            # Reuse a pending bootstrap record once at the next rollout's start.
            # Recompute its message/action distribution under current weights;
            # the previous rollout cached inputs and noise, not a chosen action.
            record = self._pending_record if self._pending_record is not None else self._new_record()
            self._pending_record = None
            decision_records.append(record)
            action_distribution, value_estimates, sender_logits, message_vector = self._decision(record)
            # The receiver samples one movement from its categorical action policy.
            # PPO stores the log-probability of THIS action, not a sender-symbol action_log_probability.
            action = action_distribution.sample()
            action_log_probability = action_distribution.log_prob(action)
            symbol_index = int(message_vector.argmax(-1))
            # CommunicationKGWorldEnv.step() in environment/KGWorldEnv.py moves
            # the receiver using its egocentric action, computes task rewards and
            # outcomes, and returns the transmitter observation for the next decision.
            # This is the episode-playing operation hidden behind train()'s single
            # collect_and_update() call. Both task endings and time limits are terminal.
            next_observation, reward, terminated, truncated, info = self.environment.step(int(action))
            self._episode_return += reward
            self._episode_length += 1
            if self._trajectory is not None:
                self._trajectory['positions'].append(self.environment.receiver_xy.tolist())
                self._trajectory['symbols'].append(symbol_index)
            # Each list entry corresponds to the same decision in registry. Rewards
            # and terminal flags describe its resulting transition. No weights are
            # updated during collection; all horizon steps use the behavior policy.
            actions.append(action.squeeze(0))
            action_log_probabilities.append(action_log_probability.squeeze(0))
            rollout_values.append(value_estimates.squeeze(0))
            rewards.append(reward)
            terminal_flags.append(terminated or truncated)
            symbols.append(symbol_index)
            rollout_logits.append(sender_logits)
            # Keep soft surrogate values for saturation diagnostics only. The
            # receiver actually consumed a discrete one-hot message, even in training.
            rollout_soft_messages.append(self.channel.soft(sender_logits, record["noise"]))
            # Ending an episode does not end collection: reset immediately, then
            # spend the remaining horizon steps in the next episode. Report the
            # whole episode's return/length even if it began in an earlier rollout.
            if terminated or truncated:
                episodes.append({**info, 'return': self._episode_return, 'length': self._episode_length,
                                 'timeout': bool(truncated and not terminated)})
                self._finish_trajectory(info, truncated)
                self._reset_env()
            else:
                self._observation = next_observation
                # representation/CommunicationCadence.py:perception_due() defines
                # the same timing used by evaluation/CommunicationEvaluation.py:
                # opening passes if stationary, every 75 steps if colocated.
                # Scene perception refreshes on its configured episode cadence.
                # Receiver position and egocentric offsets still update each decision
                # through _new_record(), even when expensive perception is skipped.
                if perception_due(self.environment, self._episode_length, self.perceive_passes):
                    self.transmitter_input.refresh(self._observation)
        # Cache the next message's provenance/noise, not its activation. It will
        # be consumed exactly once, under the updated parameters next rollout.
        # Estimate value beyond the collection boundary without stepping the env.
        # If the last transition ended an episode, _gae's terminal mask ignores
        # this value, which then belongs to the freshly reset episode instead.
        self._pending_record = self._new_record()
        _, bootstrap_value, _, _ = self._decision(self._pending_record)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        terminal_flags = torch.tensor(terminal_flags, dtype=torch.float32, device=self.device)
        rollout_values = torch.stack(rollout_values)
        # _gae() consumes environment rewards/terminal_flags flags and receiver critic values.
        # update() uses its advantages in policy_loss and its returns in value_loss.
        advantages, return_targets = self._gae(rewards, terminal_flags, rollout_values, bootstrap_value.squeeze(0))
        # The rollout batch contains fixed targets and old policy predictions.
        # update() reuses it for several epochs but does not collect new experience.
        batch = {
            'records': pack_records(decision_records, self.device),
            'actions': torch.stack(actions), 'logp': torch.stack(action_log_probabilities),
            'values': rollout_values, 'advantages': advantages, 'returns': return_targets,
        }
        self.last_record_ids = [record['id'] for record in decision_records]
        self.last_record_episodes = {record['id']: record['episode'] for record in decision_records}
        self.last_symbols = np.asarray(symbols)
        self.last_actions = batch['actions'].cpu().numpy()
        symbol_frequencies = np.bincount(self.last_symbols, minlength=self.n_symbols) / self.horizon
        # Outcome averages cover only episodes COMPLETED in this rollout; ongoing
        # episodes are excluded. Zero when no episodes finish means "no samples",
        # not an observed zero success rate. loose_death_rate uses loose-lion episodes.
        caged_episodes = [episode for episode in episodes if episode['lion_caged']]
        loose_episodes = [episode for episode in episodes if not episode['lion_caged']]
        # Parameters:
        #   episode_records: Completed episode dictionaries for the outcome subgroup being summarized.
        #   field_name: Boolean environment outcome key whose occurrence rate is being computed.
        def rate(episode_records, field_name):
            # Avoid division by zero for an empty completed-episode subgroup.
            return sum(bool(episode.get(field_name)) for episode in episode_records) / max(1, len(episode_records))
        # collect_and_update() merges these rollout statistics with update()'s
        # loss/gradient statistics. train_communication.train() sends that merged
        # dictionary to Diagnostics.append_metrics() (metrics.jsonl), prints it,
        # and later calls Diagnostics.save_training_curves().
        # symbol_entropy measures diversity of actual sampled symbols across the
        # rollout, not per-input sender uncertainty. channel_saturation measures
        # the soft gradient surrogate's values near 0/1, not forward discreteness.
        stats = {
            'episodes': len(episodes),
            'mean_return': float(np.mean([episode['return'] for episode in episodes])) if episodes else 0.,
            'mean_length': float(np.mean([episode['length'] for episode in episodes])) if episodes else 0.,
            'food_rate': rate(episodes, 'food_reached'),
            'death_rate': rate(episodes, 'lion_caught'),
            'water_rate': rate(episodes, 'water_reached'),
            'poison_rate': rate(episodes, 'poison_touched'),
            'timeout_rate': rate(episodes, 'timeout'),
            'caged_food_rate': rate(caged_episodes, 'food_reached'),
            'loose_food_rate': rate(loose_episodes, 'food_reached'),
            'loose_death_rate': rate(loose_episodes, 'lion_caught'),
            'symbol_fracs': symbol_frequencies.tolist(), 'symbol_entropy': _entropy(symbol_frequencies),
            'action_fracs': (np.bincount(self.last_actions, minlength=4) / self.horizon).tolist(),
            'channel_logit_abs': float(torch.stack(rollout_logits).abs().mean()),
            'channel_saturation': float(((torch.stack(rollout_soft_messages) < .05) |
                                         (torch.stack(rollout_soft_messages) > .95)).float().mean()),
            'channel_temperature': self.channel.temperature,
            'symbol_metric': 'sampled_discrete',
        }
        return batch, stats

    # Parameters:
    #   batch: collect_rollout() output: records, actions, old logp/values, normalized advantages, and return targets.
    def update(self, batch):
        # Second half of collect_and_update(): consumes collect_rollout()'s batch
        # and mutates the SAME transmitter/receiver instances owned by train().
        # Consequently train()'s next iteration, checkpoint save, or record_video()
        # sees the updated weights without any explicit copying back.
        # Learn from the just-collected rollout. No env.step() calls occur here.
        # All epochs compare current predictions against the SAME saved behavior
        # log-probabilities, advantages, and returns, even as weights change.
        return_targets = batch['returns']
        # Normalize value ERRORS by return standard deviation, rather than changing
        # the return targets themselves. The floor avoids division by zero.
        value_error_scale = return_targets.std().clamp_min(1e-8) if self.normalize_value_loss else 1.
        parameters = list(self.transmitter.parameters()) + list(self.receiver.parameters())
        # Per-loss gradient diagnostics examine only the sender message head;
        # actual optimization and clipping include all parameters of both agents.
        message_head_parameters = tuple(self.transmitter.message_head.parameters())
        minibatch_metrics, stop_update, stopping_kl = [], False, 0.
        actor_gradient_norm = value_gradient_norm = entropy_gradient_norm = 0.
        # Compute the sender bonus once per outer iteration; it stays fixed across
        # this update's epochs/minibatches and anneals over the total run budget.
        entropy_bonus_coefficient = entropy_coefficient(self.iteration, self.total_iterations, self.sender_entropy_beta, self.sender_entropy_schedule)
        sender_entropies = []
        # ReceiverPolicy has no recurrent state: each saved decision is sufficient
        # for its own forward pass, even though neighboring observations can be
        # statistically correlated. Reshuffle decisions each epoch; replay needs
        # no histories. Perception/anchor-memory outputs were frozen into records
        # by _new_record(), so update() does not reconstruct past environment state.
        for _ in range(self.epochs):
            for indices in torch.randperm(len(return_targets), device=self.device).split(self.minibatch_size):
                action_distribution, value_estimates, sender_logits, _ = evaluate_records(
                    self.transmitter, self.receiver, self.channel, {field_name: field_value[indices] for field_name, field_value in batch['records'].items()},
                    detach_channel=self.detach_channel)
                # Probability ratio for the saved RECEIVER action, conditional on
                # the saved scene input and Gumbel draw. This is not a marginal
                # probability ratio over all possible discrete sender messages.
                action_log_probability = action_distribution.log_prob(batch['actions'][indices])
                action_log_probability_ratio = action_log_probability - batch['logp'][indices]
                action_probability_ratio = action_log_probability_ratio.exp()
                # Estimate how far the updated action policy has drifted. Check
                # BEFORE this minibatch's optimizer step; earlier steps are retained
                # if this threshold stops the rest of the update. <=0 disables it.
                approximate_kl = ((action_probability_ratio - 1) - action_log_probability_ratio).mean()
                if self.target_kl > 0 and float(approximate_kl.detach()) > self.target_kl:
                    stop_update = True
                    stopping_kl = float(approximate_kl.detach())
                    break
                advantages = batch['advantages'][indices]
                # Positive advantage favors increasing an action's probability;
                # negative advantage favors decreasing it. The clipped alternative
                # limits the objective's incentive for overly large ratio changes.
                # Negate because the optimizer minimizes rather than maximizes.
                policy_loss = -torch.minimum(action_probability_ratio * advantages,
                    action_probability_ratio.clamp(1-self.clip_ratio, 1+self.clip_ratio) * advantages).mean()
                # Critic learns rollout return targets. Actor and critic both share
                # the receiver embedding and can train the sender through the channel.
                value_loss = ((value_estimates-return_targets[indices]) / value_error_scale).square().mean()
                # dist comes from ReceiverPolicy's actor; its entropy encourages
                # action exploration. communication/SenderEntropy.py:sender_entropy()
                # takes SymbolPolicy's logits and encourages per-scene uncertainty.
                # Sender beta comes from train()'s sender_entropy_beta/schedule options;
                # the rollout's reporting-only symbol_entropy uses _entropy() instead.
                receiver_entropy = action_distribution.entropy().mean()
                transmitter_entropy = sender_entropy(sender_logits)
                # Measure separate gradient contributions on the first accepted
                # minibatch only. autograd.grad does not populate optimizer .grad
                # buffers; retain_graph permits the full backward pass afterward.
                if not minibatch_metrics:
                    # Parameters:
                    #   loss: Scalar loss component whose gradient norm on the sender message head is measured.
                    def grad_norm(loss):
                        # A detached sender has unused parameters in receiver losses;
                        # allow_unused reports those as None, contributing zero norm.
                        gradients = torch.autograd.grad(loss, message_head_parameters, retain_graph=True, allow_unused=True)
                        return sum(float(gradient.detach().square().sum()) for gradient in gradients if gradient is not None) ** .5
                    actor_gradient_norm = grad_norm(policy_loss - self.receiver_entropy_coefficient * receiver_entropy)
                    value_gradient_norm = grad_norm(self.value_loss_coefficient * value_loss)
                    entropy_gradient_norm = grad_norm(-entropy_bonus_coefficient * transmitter_entropy) if entropy_bonus_coefficient else 0.
                # Minimize policy/value errors and SUBTRACT entropy bonuses, so
                # larger entropy lowers the loss. beta=0 disables the sender bonus.
                loss = policy_loss + self.value_loss_coefficient*value_loss - self.receiver_entropy_coefficient*receiver_entropy - entropy_bonus_coefficient*transmitter_entropy
                sender_entropies.append(float(transmitter_entropy.detach()))
                # Backward traverses ReceiverPolicy's actor/critic and embeddings,
                # CommunicationChannel's soft surrogate, then SymbolPolicy's message
                # head, attention, and symbol encoder (unless the channel is detached).
                # perception/ models and representation/ inputs are not optimizer
                # parameters. The next collect_rollout() uses these updated agents.
                # Clear old gradients, differentiate, clip, and take an Adam step.
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient_norm)
                self.optimizer.step()
                minibatch_metrics.append([float(metric.detach()) for metric in (policy_loss, value_loss, receiver_entropy, approximate_kl,
                    ((action_probability_ratio-1).abs() > self.clip_ratio).float().mean())])
            if stop_update:
                break
        # Summaries weight accepted minibatches equally. If the KL guard rejects
        # the first batch, there are no optimizer steps and loss summaries are zero.
        mean_metrics = np.mean(minibatch_metrics, axis=0) if minibatch_metrics else np.zeros(5)
        stats = dict(zip(('pi_loss', 'v_loss', 'entropy', 'approx_kl', 'clip_frac'), mean_metrics.tolist()))
        # Explained variance uses the OLD rollout values, not a fresh post-update
        # critic pass: 1 is perfect, 0 is no better than predicting a constant,
        # and negative values indicate residual variance larger than target variance.
        return_variance = float(return_targets.var())
        stats['explained_var'] = 1-float((return_targets-batch['values']).var())/return_variance if return_variance > 1e-8 else 0.
        stats.update({'tx_actor_grad_norm': actor_gradient_norm, 'tx_value_grad_norm': value_gradient_norm,
                      'tx_entropy_grad_norm': entropy_gradient_norm, 'sender_entropy_beta': entropy_bonus_coefficient,
                      'sender_entropy': float(np.mean(sender_entropies)) if sender_entropies else None,
                      'kl_early_stop': stop_update, 'stopped_kl': stopping_kl,
                      'optimizer_steps': len(minibatch_metrics)})
        # Keep explicit receiver-prefixed aliases for the existing logging/plotting
        # interface. The underlying PPO loss is the single joint objective above.
        stats.update({f'rx_{field_name}': stats[field_name] for field_name in
                      ('pi_loss', 'v_loss', 'entropy', 'approx_kl', 'clip_frac', 'explained_var')})
        return stats

    def collect_and_update(self):
        # Public entry point used by communication/train_communication.py:train().
        # The caller sets self.iteration, calls here, then logs returned stats and
        # optionally saves weights or records a separate episode. First play horizon
        # steps, then optimize; an unfinished training episode survives into the next
        # call. This function does not run the optional video/evaluation environment.
        batch, stats = self.collect_rollout()
        stats.update(self.update(batch))
        return stats

    # Parameters:
    #   rewards: Immediate rewards [horizon] from environment transitions in collection order.
    #   terminal_flags: Terminal/time-limit indicators [horizon]; stop value bootstrapping across episode resets.
    #   value_estimates: Receiver critic predictions [horizon] saved before the PPO update.
    #   bootstrap_value: Scalar critic estimate at the next decision beyond the rollout boundary.
    def _gae(self, rewards, terminal_flags, value_estimates, bootstrap_value):
        # Called only at the end of collect_rollout(). value_estimates/bootstrap_value come from
        # ReceiverPolicy's critic through _decision(); rewards/terminal_flags come from
        # CommunicationKGWorldEnv.step(). update() consumes both returned tensors:
        # normalized advantages guide the actor; unnormalized returns train critic.
        # Generalized Advantage Estimation: sweep backward to accumulate discounted
        # temporal-difference errors. Advantage estimates how much better a decision
        # was than the critic expected; returns provide the critic's training target.
        advantages = torch.zeros_like(rewards)
        accumulated_advantage = 0.0
        for step_index in reversed(range(self.horizon)):
            next_value = bootstrap_value if step_index == self.horizon - 1 else value_estimates[step_index + 1]
            # A terminal/time-limit transition contributes its immediate reward
            # but neither next-episode value nor next-episode advantage. At a plain
            # rollout boundary, bootstrap_value supplies the continuing episode's bootstrap.
            continuation_mask = 1.0 - terminal_flags[step_index]
            # One-step prediction error, then its discounted multi-step extension.
            temporal_difference_error = rewards[step_index] + self.gamma * next_value * continuation_mask - value_estimates[step_index]
            accumulated_advantage = temporal_difference_error + self.gamma * self.advantage_trace_decay * continuation_mask * accumulated_advantage
            advantages[step_index] = accumulated_advantage
        # Form unnormalized value targets BEFORE standardizing policy advantages.
        # Standardization controls policy-gradient scale without rescaling rewards.
        return_targets = advantages + value_estimates
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, return_targets


# Parameters:
#   symbol_frequencies: Empirical sampled-symbol proportions [V] for reporting entropy.
def _entropy(symbol_frequencies):
    # Called by collect_rollout() for stats['symbol_entropy'], displayed as Hsym
    # by train_communication.train() and plotted by observability/Diagnostics.py.
    # The TRAINING entropy objective is SenderEntropy.sender_entropy() in update().
    # Reporting-only entropy of sampled symbol frequencies, in nats (natural log).
    # Ignore zero-probability entries to avoid log(0); this never contributes gradients.
    probabilities = np.asarray(symbol_frequencies, dtype=np.float64)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())
