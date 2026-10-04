#!/bin/python3
import sys, os, json
import hashlib
from functools import partial
from types import SimpleNamespace
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from shared.Logger import Logger
from shared.ArgumentParser import ArgumentParser
from environment.KGWorldEnv import CommunicationKGWorldEnv
from perception.Perception import PerceptionPipeline
from representation.SceneVocabulary import build_scene_vocabulary
from representation.SceneVocabulary import scene_bodies
from perception.ScenePerception import perceive_scene_softcat
from perception.CategoryModel import load_category_checkpoint
from communication.SymbolPolicy import GumbelTransmitter
from communication.GumbelPPOTrainer import GumbelPPOTrainer
from communication.SenderEntropy import entropy_coefficient
from communication.CommunicationChannel import StraightThroughGumbelChannel
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic
from representation.TransmitterInput import TransmitterInput
from representation.OracleTransmitterInput import OracleTransmitterInput
from observability.VideoRecorder import record_communication_episode, segment_start_iters
from observability import Diagnostics

logger = Logger(__name__)
print = logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT = os.path.join(_TROY_DEV, "output")

ACTION_LABELS = ("forward", "backward", "left", "right")

required_arguments = []
# CLI values use key=value syntax; defaults below are strings until parsed.
# Boolean switches use 0/1, not true/false. Except for colocated, their current
# parser treats any value other than "1" as false; use the documented 0/1 values.
optional_arguments = {
    # Scene input: "softcat" runs visual perception; "oracle_state" uses exact simulator state.
    "perception": "softcat",
    # Receiver heading: "shared" matches transmitter yaw; "independent" samples its own yaw.
    "receiver_yaw_mode": "shared",
    # Position mode: "0" keeps transmitter stationary; "1" follows receiver and refreshes every 75 steps.
    "colocated": "0",
    # Backward-surrogate temperature: finite float > 0; lower makes the soft surrogate sharper.
    # Forward messages remain discrete; temperature does not change their sampling probabilities.
    "channel_temperature": "1.2",
    # Communication vocabulary size: integer >= 2; must match warm-start agent weights.
    "n_symbols": "16",
    # Sender uncertainty bonus: finite float >= 0; 0 disables it. Must be 0 if detach_channel=1.
    "sender_entropy_beta": "0",
    # Bonus schedule: "constant" keeps beta fixed; "annealed" holds through 25%, decays to 0 at 75%.
    "sender_entropy_schedule": "annealed",
    # Gradient control: "0" allows receiver losses to train sender; "1" cuts that gradient path.
    "detach_channel": "0",
    # PPO update stopping threshold: use a finite float > 0; 0 disables the guard.
    # The implementation also treats negative values as disabled.
    "target_kl": "0.03",
    # Messages in optional videos: "hard" selects sender argmax; "sampled" samples categories.
    # Does not change training message sampling; receiver actions in videos remain sampled.
    "video_channel": "hard",
    # Number of collect-and-update iterations: integer >= 1; also sets the entropy schedule budget.
    "iterations": "800",
    # Environment steps collected per iteration: integer >= 2; a rollout may span multiple episodes.
    "horizon": "2048",
    # Scene-row capacity, excluding the extra receiver row: integer >= 1.
    # Oracle mode requires >= 3 without resources, or >= 5 with resources=1.
    "k_triples": "6",
    # Opening scene passes in stationary mode: integer >= 1, including the reset pass.
    # Colocated mode instead refreshes at reset and every 75 steps, regardless of this value.
    "perceive_passes": "1",
    # Adam learning rate for both agents: finite float >= 0; 0 leaves weights unchanged.
    "lr": "3e-4",
    # Intermediate checkpoint interval in iterations: integer >= 0; 0 disables intermediate saves.
    # Final agent weights are always saved.
    "checkpoint_every": "25",
    # Requested episode videos across the run: integer >= 0; 0 disables recording.
    # Recording iterations are deduplicated, so at most one video is made per iteration.
    "videos": "0",
    # Output parent directory: nonempty filesystem path; created if missing.
    # Each invocation creates a separate run directory beneath it.
    "out_dir": os.path.join(_OUTPUT, "runs"),
    # Optional run-directory label: string (use a simple filename label), or empty for no label.
    "run_name": "",
    # Available PyTorch device string, e.g. "cpu", "cuda", or "cuda:0".
    # The selected backend must support all models used by the chosen perception mode.
    "device": "cuda",
    # Random initialization/environment seed: integer from 0 through 2**64 - 1.
    "seed": "0",
    # Warm start: empty string for fresh weights, or a compatible existing run-directory path.
    # Requires config.json, transmitter_final.pt, and receiver_final.pt; restores weights only.
    "init_from": "",
    # Value-loss scaling: "1" divides value errors by rollout return standard deviation; "0" disables it.
    "normalize_value": "1",
    # Probability of a near-lion food layout: float in [0, 1]; 0 never, 1 always.
    # This probability range is expected by the environment but is not explicitly validated.
    "food_near_lion": "0.0",
    # Food-to-lion spawn distance for near-lion layouts: finite float >= 0 in world units.
    # 0 overlaps their positions; positive offsets use a random direction and clip to arena bounds.
    # Ignored when food_near_lion=0; negative values currently also produce overlap.
    "food_near_lion_offset": "1.5",
    # Category-head checkpoint: existing file path for perception=softcat; ignored in oracle_state mode.
    "category_model": os.path.join(_OUTPUT, "category_model.pt"),
    # Scene contents: "0" omits water/poison; "1" enables them in addition to food/lion/cage/receiver.
    "resources": "0",
}

USAGE = """
Gumbel communication PPO. Arguments use key=value syntax (unknown keys fail).

Training always uses straight-through Gumbel PPO and a current-message MLP receiver.
Defaults: n_symbols=16 channel_temperature=1.2 receiver_yaw_mode=shared
colocated=0 perception=softcat iterations=800 horizon=2048 device=cuda videos=0.

perception=softcat uses SAM, frozen ResNet features, and a category checkpoint.
perception=oracle_state skips visual inference for fast testing. Both include
an exact receiver-position slot; oracle_state also supplies exact scene state.
colocated=1 follows the receiver and refreshes scene perception every 75 steps.
Stationary mode perceives at reset and for the opening perceive_passes steps.

sender_entropy_beta=0.1 sender_entropy_schedule=annealed enables the sender bonus:
hold for the first quarter, decay to zero at three quarters, then zero.
detach_channel=1 is a no-channel-gradient control (requires zero sender bonus).
Value gradients reach the sender unless detach_channel=1 disconnects the channel.
target_kl=0.03 limits update drift; 0 disables the guard.
video_channel=hard or sampled; videos use a separate environment.
init_from=<run directory> restores compatible Gumbel weights only.

Fast smoke: python3 communication/train_communication.py perception=oracle_state \
    device=cpu iterations=2 horizon=32 videos=0
See README.md for perception setup, evaluation, and all default arguments below.
"""


# Parameters:
#   input_arguments: CLI strings in key=value form; main() also accepts --help/-h.
def initialize(input_arguments):
    """Parse CLI values and validate settings before creating any run artifacts."""
    print(f"ScriptName: {__file__}")
    parser = ArgumentParser()
    try:
        parser.setArguments(input_arguments, required_arguments, optional_arguments)
        raw_arguments = {key: parser.get(key) for key in optional_arguments}
        settings = SimpleNamespace(**raw_arguments, raw_arguments=raw_arguments)
        for key in ("n_symbols", "iterations", "horizon", "k_triples", "perceive_passes",
                    "checkpoint_every", "videos", "seed"):
            setattr(settings, key, int(raw_arguments[key]))
        for key in ("channel_temperature", "sender_entropy_beta", "lr", "target_kl",
                    "food_near_lion", "food_near_lion_offset"):
            setattr(settings, key, float(raw_arguments[key]))
        if raw_arguments["colocated"] not in ("0", "1"):
            raise ValueError("colocated must be 0 or 1")
        for key in ("colocated", "detach_channel", "normalize_value", "resources"):
            setattr(settings, key, raw_arguments[key] == "1")

        if settings.perception not in ("softcat", "oracle_state"):
            raise ValueError("perception must be softcat or oracle_state")
        if settings.video_channel not in ("hard", "sampled"):
            raise ValueError("video_channel must be hard or sampled")
        if settings.receiver_yaw_mode not in ("independent", "shared"):
            raise ValueError("receiver_yaw_mode must be independent or shared")
        if settings.detach_channel and settings.sender_entropy_beta:
            raise ValueError("Detached channel requires zero sender entropy bonus")
        StraightThroughGumbelChannel(settings.channel_temperature, settings.n_symbols)
        entropy_coefficient(1, settings.iterations, settings.sender_entropy_beta,
                            settings.sender_entropy_schedule)
        if (settings.iterations < 1 or settings.horizon < 2 or
                settings.k_triples < 1 or settings.perceive_passes < 1):
            raise ValueError("positive iterations/k_triples/perceive_passes and horizon >= 2 required")
        if settings.checkpoint_every < 0 or settings.videos < 0:
            raise ValueError("checkpoint_every and videos must be nonnegative")
        if settings.perception == "oracle_state" and settings.k_triples < (5 if settings.resources else 3):
            raise ValueError("oracle_state requires room for every task object")

        if settings.init_from:
            previous_configuration_path = os.path.join(settings.init_from, "config.json")
            if not os.path.exists(previous_configuration_path):
                raise ValueError("Gumbel warm starts require versioned config.json")
            with open(previous_configuration_path) as handle:
                previous_configuration = json.load(handle)
            if int(previous_configuration.get('n_symbols', 4)) != settings.n_symbols:
                raise ValueError('Warm start vocabulary must match')
            if previous_configuration['method'] != 'st_gumbel_ppo' or int(previous_configuration['k_triples']) != settings.k_triples:
                raise ValueError("Warm start method and triple count must match")
            if previous_configuration.get('receiver_arch') != 'current_mlp':
                raise ValueError("Warm start receiver architecture must match")
    except Exception as e:
        e.add_note(USAGE)
        raise
    parser.printArguments()
    return settings


# Parameters:
#   environment: Simulation instance used for the training rollout or the separate recorded episode.
#   settings: Typed options returned by initialize(); includes CLI values in raw_arguments for saved metadata.
#   vocabulary: SceneVocabulary containing object/relation IDs and initial embedding tensors.
#   body_to_node: Mapping from simulator body names to integer scene-object IDs.
#   pipeline: Visual PerceptionPipeline; unused/None for oracle inputs.
#   category_model: Frozen category classifier over visual proposal features; unused/None for oracle inputs.
def create_transmitter_input(environment, *, settings, vocabulary, body_to_node,
                             pipeline=None, category_model=None):
    """Build a fresh visual/oracle adapter for a training or video environment.

    The adapter owns its anchor memory. In visual mode, partial binds the scene
    reader's dependencies without running it; refresh() invokes it when needed.
    Both modes include the same reserved, live receiver-position slot.
    """
    if settings.perception == "oracle_state":
        return OracleTransmitterInput(vocabulary, environment, body_to_node, settings.k_triples)
    perceive = partial(perceive_scene_softcat, environment, pipeline, vocabulary,
                       category_model, body_to_node)
    return TransmitterInput(vocabulary, perceive, body_to_node["receiver"], settings.k_triples)


# Parameters:
#   model: PyTorch model whose named state tensors are fingerprinted.
def weight_hash(model):
    """Fingerprint actual initialized/loaded weights without consuming RNG."""
    weight_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        weight_digest.update(name.encode())
        weight_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return weight_digest.hexdigest()


# Parameters:
#   iteration: One-based outer training iteration (or direct-loss diagnostic update).
#   settings: Typed options returned by initialize(); includes CLI values in raw_arguments for saved metadata.
#   video_directory: Output directory for episode_iterNNNN.mp4 recordings.
#   video_environment: Separate simulator instance so recording cannot advance the training episode.
#   transmitter: GumbelTransmitter: scene pose/rows/offsets -> categorical message logits.
#   receiver: CurrentMessageReceiverActorCritic: one message -> action distribution and value.
#   vocabulary: SceneVocabulary containing object/relation IDs and initial embedding tensors.
#   make_video_input: Zero-argument factory creating fresh input/anchor memory for the video environment.
def record_video(iteration, *, settings, video_directory, video_environment, transmitter, receiver,
                 vocabulary, make_video_input):
    """Record a separate episode, restoring model modes and training RNG afterward."""
    path = os.path.join(video_directory, f"episode_iter{iteration:04d}.mp4")
    # Evaluation must not advance the training policy's random stream.
    # manual_seed seeds all CUDA devices, so preserve all their generators.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    transmitter_was_training, receiver_was_training = transmitter.training, receiver.training
    try:
        transmitter.eval()
        receiver.eval()
        # fork_rng restores the original CPU/CUDA generator states on exit.
        # The fixed recording seed makes video sampling reproducible without
        # consuming the random draws that the next training rollout would use.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(100_000 + settings.seed + iteration)
            outcome, episode_return = record_communication_episode(
                path, video_environment, transmitter, receiver, vocabulary, make_video_input,
                settings.perceive_passes, settings.device, tag=f"iter {iteration}", channel_mode=settings.video_channel)
    finally:
        transmitter.train(transmitter_was_training)
        receiver.train(receiver_was_training)
    print(f"Wrote {path} (outcome={outcome}, return={episode_return:+.2f})")


# Parameters:
#   settings: Typed options returned by initialize(); includes CLI values in raw_arguments for saved metadata.
def train(settings):
    """Run an experiment using settings already parsed and validated by initialize."""
    # These labels identify channel categories in logs; they do not assign
    # meanings such as "food" or "left". The agents learn how to use each symbol.
    symbol_names = tuple(f"s{index}" for index in range(settings.n_symbols))
    # Create an isolated run directory so checkpoints, metrics, and visualizations
    # belong to this invocation. metrics.jsonl stores one JSON record per update.
    os.makedirs(settings.out_dir, exist_ok=True)
    run_directory = Diagnostics.make_run_dir(settings.out_dir, "comm", settings.run_name)
    checkpoint_directory = os.path.join(run_directory, "checkpoints")
    plot_directory = os.path.join(run_directory, "plots")
    video_directory = os.path.join(run_directory, "videos")
    metrics_path = os.path.join(run_directory, "metrics.jsonl")
    for directory in (checkpoint_directory, plot_directory, video_directory):
        os.makedirs(directory, exist_ok=True)
    print(f"Run directory: {run_directory}")
    # Seed PyTorch before constructing the policies. The environment receives its
    # own seed below, while the scene vocabulary uses its own seeded generator.
    torch.manual_seed(settings.seed)
    # Preserve the CLI representation expected by existing evaluation loaders,
    # then add metadata needed to reconstruct this fixed policy architecture.
    # method/receiver_arch identify checkpoint compatibility; they select no code.
    # Episode time limits are treated as terminal states when computing returns.
    config = settings.raw_arguments.copy()
    config.update(method="st_gumbel_ppo", receiver_arch="current_mlp",
                  perception=settings.perception, perceive_every=75 if settings.colocated else 0,
                  format_version=2, symbol_names=list(symbol_names),
                  resume_semantics="weights_only", time_limit_terminal=True,
                  observation_dim=4, transmitter_hidden=38,
                  receiver_model_dim=4, receiver_hidden=4,
                  category_model=os.path.abspath(settings.category_model) if settings.perception == "softcat" else None)
    if settings.perception == "oracle_state":
        config['oracle_spec'] = 'all objects; exact world anchors; inside iff lion_caged; otherwise unary lion/cage; per-step egocentric offsets'
    config.update(message_categories=settings.n_symbols, gradient_estimator='straight_through',
                  training_messages='sampled_discrete', channel_temperature=settings.channel_temperature)
    with open(os.path.join(run_directory, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)

    # Choose iterations at which to record a separate episode after the update.
    # A set removes duplicate iteration indices when more videos than iterations
    # are requested. With videos=0, the set is empty and no video environment runs.
    # This is for human viewing. 
    recording_iterations = set(segment_start_iters(settings.iterations, settings.videos))
    print(f"Recording one episode after iterations: {sorted(recording_iterations)}")

    # This is the environment used to collect PPO experience. Its layout settings
    # control the task distribution; shared yaw aligns the two agents' headings,
    # while colocated mode makes the transmitter follow the receiver's position.
    environment = CommunicationKGWorldEnv(
        seed=settings.seed,
        food_near_lion_prob=settings.food_near_lion,
        food_near_lion_offset=settings.food_near_lion_offset,
        resources=settings.resources,
        receiver_yaw_mode=settings.receiver_yaw_mode, colocated=settings.colocated,
    )
    video_environment = None
    try:
        print(
            f"Env: food_near_lion_prob={settings.food_near_lion} offset={settings.food_near_lion_offset} "
            f"resources={settings.resources} receiver_yaw_mode={settings.receiver_yaw_mode} colocated={settings.colocated}"
        )

        # Object IDs identify scene entities, not communication symbols. Both
        # input adapters use this same mapping. Save the initial embedding tensors
        # and vocabulary so evaluation can reconstruct the transmitter's shapes.
        bodies = scene_bodies(environment)
        vocabulary, body_to_node = build_scene_vocabulary(bodies, device=settings.device, seed=settings.seed)
        print(f"Scene vocabulary: one ID per body {bodies} -> ids {body_to_node}")
        vocabulary.save(os.path.join(run_directory, "scene_vocabulary_init.pt"))

        # Visual mode uses a separately trained category head plus pretrained
        # SAM/ResNet features. PPO trains the communication policies, not these
        # perception models. Oracle mode constructs neither visual model.
        pipeline = category_model = None
        if settings.perception == "softcat":
            category_model, category_checkpoint = load_category_checkpoint(settings.category_model, device=settings.device)
            category_model.requires_grad_(False)
            print(f"Category head {settings.category_model} (epoch {category_checkpoint['epoch']})")
            print(f"Loading SAM + ResNet-18 ({settings.device})...")
            pipeline = PerceptionPipeline(device=settings.device)
        else:
            print("Oracle state: exact simulator triples; skipping SAM, ResNet and category model")

        # Bind shared input settings once, then bind the environment separately.
        # Calling either factory creates fresh episode-local memory; perception
        # itself remains deferred until the adapter requests a scene refresh.
        input_factory = partial(create_transmitter_input, settings=settings,
                                vocabulary=vocabulary, body_to_node=body_to_node,
                                pipeline=pipeline, category_model=category_model)
        make_transmitter_input = partial(input_factory, environment)

        # The transmitter encodes scene rows and pose into categorical logits.
        # Add one row beyond k_triples for the reserved receiver-position input.
        # The receiver consumes only the resulting message; its actor chooses a
        # movement and its critic estimates return. The trainer moves both models
        # to settings.device before creating their shared optimizer.
        transmitter = GumbelTransmitter(vocabulary, observation_dimension=environment.observation_space.shape[0],
                                         k_triples=settings.k_triples + 1, n_symbols=settings.n_symbols).to(settings.device)
        receiver = CurrentMessageReceiverActorCritic(
            n_actions=int(environment.action_space.n), n_symbols=settings.n_symbols)
        print(
            f"Transmitter: {sum(parameter.numel() for parameter in transmitter.parameters()):,} params, "
            f"{settings.n_symbols} symbols {','.join(symbol_names)}"
        )
        print(f"Receiver:    {sum(parameter.numel() for parameter in receiver.parameters()):,} params")

        # Keep a detached copy for the final embedding-change report. This snapshot
        # is taken BEFORE loading warm-start weights, so in a warm-start run the
        # report compares against fresh initialization, not the loaded checkpoint.
        initial_node_symbols = transmitter.symbol_table.node_symbols.detach().cpu().clone()
        # Restore only agent parameters. This invocation still starts a fresh
        # environment, optimizer, iteration counter, and entropy schedule.
        if settings.init_from:
            transmitter.load_state_dict(
                torch.load(os.path.join(settings.init_from, "transmitter_final.pt"),
                           map_location=settings.device, weights_only=True))
            receiver.load_state_dict(
                torch.load(os.path.join(settings.init_from, "receiver_final.pt"),
                           map_location=settings.device, weights_only=True))
            print(f"Warm-started both agents from {settings.init_from}")

        # Hash actual initialized/loaded parameters without consuming RNG.
        with open(os.path.join(run_directory, 'initialization.json'), 'w') as handle:
            json.dump({'transmitter_sha256': weight_hash(transmitter),
                       'receiver_sha256': weight_hash(receiver)}, handle, indent=2)

        # The trainer owns rollout collection, Gumbel-noise replay, advantage
        # computation, and joint optimization. Actor and critic losses can reach
        # the transmitter through the channel unless detach_channel is enabled.
        # total_iterations defines the sender-entropy annealing horizon; the
        # current iteration is supplied immediately before each update below.
        trainer = GumbelPPOTrainer(
            transmitter, receiver, environment, make_transmitter_input,
            device=settings.device, horizon=settings.horizon, learning_rate=settings.lr,
            perceive_passes=settings.perceive_passes, record_trajectories=True,
            normalize_value_loss=settings.normalize_value,
            target_kl=settings.target_kl,
            detach_channel=settings.detach_channel,
            temperature=settings.channel_temperature, n_symbols=settings.n_symbols, sender_entropy_beta=settings.sender_entropy_beta,
            sender_entropy_schedule=settings.sender_entropy_schedule, total_iterations=settings.iterations,
        )
        print(f"normalize_value_loss={settings.normalize_value}  perceive_passes={settings.perceive_passes}"
              f"  perceive_every={75 if settings.colocated else 0}")

        video_environment = None
        # Record against a separate environment so recording cannot reset or
        # advance the episode currently being used for PPO experience collection.
        if settings.videos:
            video_environment = CommunicationKGWorldEnv(
                seed=100_000 + settings.seed, food_near_lion_prob=settings.food_near_lion,
                food_near_lion_offset=settings.food_near_lion_offset, resources=settings.resources,
                receiver_yaw_mode=settings.receiver_yaw_mode, colocated=settings.colocated)

        # Reuse the same adapter construction as training, but bind it to video_env
        # and give the recorded episode its own perception/anchor-memory state.
        make_video_input = partial(input_factory, video_environment)

        # Keep update summaries for learning curves and a bounded set of completed
        # episode paths for the final trajectory plot. Neither is PPO replay data.
        metric_rows, trajectories = [], []
        for iteration in range(1, settings.iterations + 1):
            # One iteration collects horizon environment steps (possibly spanning
            # several episodes), then performs PPO minibatch updates on that rollout.
            # The one-based index determines this iteration's sender entropy bonus.
            # An iteration is one cycle of updating the agents. An episode is one attempt 
            # at the environment, ending in success, death, or timeout. 
            trainer.iteration = iteration
            # Collection stores scene inputs, actions, old action log-probabilities,
            # and Gumbel noise. Updating recomputes messages with that saved noise,
            # forms PPO/value/entropy losses, and updates both agents. Returned stats
            # combine rollout outcomes with optimization diagnostics.
            stats = trainer.collect_and_update() # once per iteration. This is the big boy, it does a lot. See related class definition for what is going on here.  
            # Append metrics immediately after each update so progress is available
            # during a running job; plotting later uses the same in-memory records.
            stats["iteration"] = iteration
            metric_rows.append(stats)
            Diagnostics.append_metrics(metrics_path, stats)
            # Drain newly completed episode paths into the plotting history.
            # An episode can cross update boundaries; its label is the iteration
            # when it finished. Clearing this list prevents counting it again.
            for trajectory in trainer.trajectories:
                trajectory["iteration"] = iteration
            trajectories.extend(trainer.trajectories)
            trajectories = trajectories[-200:]
            trainer.trajectories = []

            # Summarize observed message/action usage, not prescribed symbol meanings.
            # Hsym is entropy of rollout symbol frequencies; RX ent is action entropy.
            # Actor/value gradient norms show their separate influence on the sender
            # message head, and explained variance reports critic prediction quality.
            most_frequent_symbols = np.argsort(stats["symbol_fracs"])[::-1][:3]
            symbol_summary = " ".join(f"{symbol_names[index]}={stats['symbol_fracs'][index]:.2f}" for index in most_frequent_symbols)
            action_summary = " ".join(
                f"{action_name[0]}={frequency:.2f}" for action_name, frequency in zip(ACTION_LABELS, stats["action_fracs"])
            )
            transmitter_gradient_summary = (f"TX actor_grad={stats['tx_actor_grad_norm']:.3g} "
                         f"value_grad={stats['tx_value_grad_norm']:.3g}")
            print(
                f"iter {iteration}/{settings.iterations}: return={stats['mean_return']:+.2f} "
                f"len={stats['mean_length']:.0f} food={stats['food_rate']:.2f} "
                f"death={stats['death_rate']:.2f} "
                f"| caged_food={stats['caged_food_rate']:.2f} "
                f"loose_food={stats['loose_food_rate']:.2f} "
                f"loose_death={stats['loose_death_rate']:.2f} "
                f"| {transmitter_gradient_summary} "
                f"| RX ent={stats['rx_entropy']:.2f} ev={stats['rx_explained_var']:+.2f} "
                f"| Hsym={stats['symbol_entropy']:.2f} [{symbol_summary}] acts[{action_summary}]"
            )

            # Record the just-updated policy. Video experience never enters the
            # trainer's rollout buffer and contributes no gradients or rewards.
            if iteration in recording_iterations:
                record_video(iteration, settings=settings, video_directory=video_directory,
                             video_environment=video_environment, transmitter=transmitter, receiver=receiver,
                             vocabulary=vocabulary, make_video_input=make_video_input)

            # Intermediate saves contain agent state dictionaries only, not optimizer
            # or environment state. Skip the final iteration here because final
            # checkpoints are written separately after the loop; zero disables this.
            if settings.checkpoint_every and iteration % settings.checkpoint_every == 0 and iteration < settings.iterations:
                torch.save(transmitter.state_dict(),
                           os.path.join(checkpoint_directory, f"iter{iteration:04d}_transmitter.pt"))
                torch.save(receiver.state_dict(),
                           os.path.join(checkpoint_directory, f"iter{iteration:04d}_receiver.pt"))
                print(f"Checkpoint saved at iteration {iteration}")
                Diagnostics.save_training_curves(metric_rows, plot_directory)

        # Always save final agent weights, even when intermediate saves are disabled.
        # unique_path avoids overwriting an existing file. These files support
        # evaluation and weights-only warm starts, not exact continuation of a run.
        transmitter_path = Diagnostics.unique_path(os.path.join(run_directory, "transmitter_final.pt"))
        receiver_path = Diagnostics.unique_path(os.path.join(run_directory, "receiver_final.pt"))
        torch.save(transmitter.state_dict(), transmitter_path)
        torch.save(receiver.state_dict(), receiver_path)
        # The policy trained its own embedding parameters copied from the vocabulary.
        # Export their learned values back into the vocabulary container for inspection;
        # the complete transmitter checkpoint above also includes these embeddings.
        vocabulary.symbols = transmitter.symbol_table.node_symbols.detach().cpu().clone()
        vocabulary.relation_symbols = transmitter.symbol_table.relation_symbols.detach().cpu().clone()
        vocabulary.save(Diagnostics.unique_path(os.path.join(run_directory, "scene_vocabulary_trained.pt")))
        print(f"Saved {transmitter_path} and {receiver_path}")

        # The report combines final-rollout outcomes with mean symbol frequencies
        # over up to 25 recent iterations. These are training diagnostics, not a
        # separate held-out evaluation or an estimate across multiple training seeds.
        final_metrics = metric_rows[-1]
        symbol_frequencies = np.mean([metrics_row["symbol_fracs"] for metrics_row in metric_rows[-25:]], axis=0)
        lines = [
            f"Final iteration {final_metrics['iteration']}: return={final_metrics['mean_return']:+.2f} "
            f"food={final_metrics['food_rate']:.2f} death={final_metrics['death_rate']:.2f} "
            f"caged_food={final_metrics['caged_food_rate']:.2f} "
            f"loose_food={final_metrics['loose_food_rate']:.2f} "
            f"loose_death={final_metrics['loose_death_rate']:.2f}",
            "",
            "Symbol usage over the last 25 iterations "
            f"(uniform would be {1 / settings.n_symbols:.2f} each, max entropy "
            f"{np.log(settings.n_symbols):.2f}):",
        ]
        for index, name in enumerate(symbol_names):
            lines.append(f"  '{name}': {symbol_frequencies[index]:.4f}")
        lines.append(f"  symbol entropy: {final_metrics['symbol_entropy']:.3f}")
        lines.append("")
        # L2 change measures embedding displacement; cosine similarity measures
        # directional alignment with the fresh-initialization snapshot saved above.
        # detach()/cpu() keeps reporting outside autograd and off accelerator storage.
        for index in range(vocabulary.num_objects):
            initial_embedding = initial_node_symbols[index]
            trained_embedding = transmitter.symbol_table.node_symbols.detach().cpu()[index]
            cosine_similarity = float(torch.nn.functional.cosine_similarity(initial_embedding, trained_embedding, dim=0))
            lines.append(
                f"  node {index}: L2 delta={float((initial_embedding - trained_embedding).norm()):.3f} cosine={cosine_similarity:+.3f}"
            )
        report = "\n".join(lines)
        with open(os.path.join(run_directory, "report.txt"), "w") as frequency:
            frequency.write(report + "\n")
        print("\n" + report)

        # Render the accumulated metrics and recent completed trajectories only after
        # final weights have been saved, so plotting does not gate checkpoint creation.
        written = Diagnostics.save_training_curves(metric_rows, plot_directory)
        trajectory_plot = Diagnostics.save_trajectory_plot(trajectories, plot_directory)
        if trajectory_plot:
            written.append(trajectory_plot)
        for plot_path in written:
            print(f"Wrote plot {plot_path}")

    finally:
        # Release simulator/rendering resources on both normal completion and errors
        # raised inside this training block. The optional environment may not exist.
        if video_environment is not None:
            video_environment.close()
        environment.close()


# Parameters:
#   input_arguments: CLI strings in key=value form; main() also accepts --help/-h.
def main(input_arguments):
    if input_arguments in (["--help"], ["-h"]):
        print(USAGE)
        print("Defaults: " + " ".join(f"{argument_name}={default_value}" for argument_name, default_value in optional_arguments.items()))
        return
    settings = initialize(input_arguments)
    train(settings)
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        logger.logger.exception(e)
        exit(1)
