#!/bin/python3
import sys, os
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
from lib.KGWorldEnv import CommunicationKGWorldEnv
from lib.Perception import PerceptionPipeline
from lib.OracleGrounding import (
    build_oracle_kg,
    oracle_bodies,
    perceive_scene_softcat,
)
from lib.CategoryModel import load_category_checkpoint
from lib.SymbolPolicy import TripleActorCritic
from lib.ReceiverPolicy import N_SYMBOLS, SYMBOL_NAMES, ReceiverActorCritic
from lib.CommunicationTrainer import CommunicationTrainer, TransmitterInput
from lib.VideoRecorder import record_communication_episode, segment_start_iters
from lib import Diagnostics

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT = os.path.join(_TROY_DEV, "output")

ACTION_LABELS = ("forward", "backward", "left", "right")

required_arguments = []
optional_arguments = {
    "iterations": "450",
    "horizon": "2048",
    "k_triples": "6",         # SAM slots; the receiver gets a 7th, reserved
    "perceive_passes": "1",   # SAM passes at episode start, then frozen
    "lr": "3e-4",
    "checkpoint_every": "25",
    "videos": "10",           # one episode recorded per equal segment of the run
    "out_dir": os.path.join(_OUTPUT, "runs"),
    "run_name": "",
    "device": "cuda",
    "seed": "0",
    "init_from": "",          # dir holding transmitter.pt / receiver.pt
    "normalize_value": "1",
    "food_near_lion": "0.0",
    "food_near_lion_offset": "1.5",
    "category_model": os.path.join(_OUTPUT, "category_model.pt"),
    "resources": "0",
}

USAGE = """
train_communication.py -- two-agent PPO: a transmitter that emits symbols and a
blind receiver that acts on them.

TRANSMITTER. Immobile, keeps the 4-camera rig and the whole perception stack.
Its input is the familiar distance-gated triple vector, 4 + (k_triples+1)*12 =
88 dims at defaults: the env vector [x/8, y/8, cos yaw, sin yaw] (constant, it
never moves) plus k_triples SAM-derived slots and one reserved slot for the
receiver. Its head emits one of 10 symbols (a-j) per step instead of a move.

RECEIVER. Blind -- it never sees the world, only the symbol stream. Context is
11 tokens x 4 dims = 44: [CLS][s_1]..[s_n][NULL]x(10-n), learned embeddings plus
learned positional encodings, through 3 transformer blocks with 1 head each.
The contextualized CLS goes to a 4-unit hidden layer then 4 action logits
(forward/backward/left/right). After 10 symbols the window slides.

REWARD. Both agents receive the SAME reward, the ordinary task outcome, and it
is evaluated on the RECEIVER's position: +10 food, -50 loose lion, -0.01/step,
and the potential-based food shaping. Neither agent has a private objective, so
the transmitter can only score by emitting symbols that steer the receiver and
the receiver can only score by reading them. No gradient crosses the channel --
the symbol is a sampled discrete index.

PERCEPTION. SAM + the soft-category head run ONCE per episode, at reset, and
never again: the transmitter cannot move and the lion/cage/food do not move, so
their anchors stay exact for the episode. A second consecutive pass was measured
over 30 episodes and added a static object 0 times, so it was dropped. Only the
receiver's position updates, and it is read from the simulator every step -- a
deliberate, stated oracle shortcut, because SAM at ~1.1 s/frame cannot run
per-step and a frozen anchor for the one moving object would be uncorrelated
with the truth. The receiver is position-only and never appears in a relation.

Each invocation writes a fresh timestamped run directory under `out_dir`:
    checkpoints/iterNNNN_{transmitter,receiver}.pt
    transmitter_final.pt, receiver_final.pt, kg_trained.pt
    metrics.jsonl          per-iteration stats for both agents
    report.txt             symbol usage + final summary
    plots/                 training curves, symbol/action stats, trajectories
    videos/                one transmitter's-eye episode per segment

    Optional:
        iterations=450 horizon=2048 k_triples=6 perceive_passes=1 lr=3e-4
        checkpoint_every=25 videos=10 out_dir=.../runs run_name= device=cuda
        seed=0 init_from= normalize_value=1 food_near_lion=0.0
        food_near_lion_offset=1.5 category_model=.../category_model.pt
        resources=0

    `videos=N` records one episode after each iteration that BEGINS one of N
    equal segments of the run: at iterations=450, videos=10 that is iterations
    1, 46, 91, ... 406.

    Example Usage:
        train_communication.py iterations=450 run_name=comm_s0
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def train():
    iterations = int(g_ArgParse.get("iterations"))
    horizon = int(g_ArgParse.get("horizon"))
    k_triples = int(g_ArgParse.get("k_triples"))
    perceive_passes = int(g_ArgParse.get("perceive_passes"))
    lr = float(g_ArgParse.get("lr"))
    checkpoint_every = int(g_ArgParse.get("checkpoint_every"))
    n_videos = int(g_ArgParse.get("videos"))
    out_dir = g_ArgParse.get("out_dir")
    run_name = g_ArgParse.get("run_name")
    device = g_ArgParse.get("device")
    seed = int(g_ArgParse.get("seed"))
    init_from = g_ArgParse.get("init_from")
    normalize_value = g_ArgParse.get("normalize_value") == "1"
    food_near_lion = float(g_ArgParse.get("food_near_lion"))
    food_near_lion_offset = float(g_ArgParse.get("food_near_lion_offset"))
    category_model_path = g_ArgParse.get("category_model")
    resources = g_ArgParse.get("resources") == "1"

    os.makedirs(out_dir, exist_ok=True)
    run_dir = Diagnostics.make_run_dir(out_dir, "comm", run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    video_dir = os.path.join(run_dir, "videos")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    for d in (ckpt_dir, plot_dir, video_dir):
        os.makedirs(d, exist_ok=True)
    print(f"Run directory: {run_dir}")
    torch.manual_seed(seed)

    record_iters = set(segment_start_iters(iterations, n_videos))
    print(f"Recording one episode after iterations: {sorted(record_iters)}")

    env = CommunicationKGWorldEnv(
        seed=seed,
        food_near_lion_prob=food_near_lion,
        food_near_lion_offset=food_near_lion_offset,
        resources=resources,
    )
    print(
        f"Env: food_near_lion_prob={food_near_lion} offset={food_near_lion_offset} "
        f"resources={resources}"
    )

    bodies = oracle_bodies(env)
    kg, body_to_node = build_oracle_kg(bodies, device=device, seed=seed)
    receiver_node = body_to_node["receiver"]
    print(f"KG: one node per body {bodies} -> ids {body_to_node}")
    kg.save(os.path.join(run_dir, "kg_init.pt"))

    category_model, blob = load_category_checkpoint(category_model_path, device=device)
    category_model.requires_grad_(False)
    print(f"Category head {category_model_path} (epoch {blob['epoch']})")
    print("Loading SAM + ResNet-18 (GPU)...")
    pipeline = PerceptionPipeline(device=device)

    def per_frame_fn():
        return perceive_scene_softcat(env, pipeline, kg, category_model, body_to_node)

    # ONE factory, used by both the trainer and the video recorder, so the input
    # the videos show is assembled by the same code that produced it in training
    def make_tx_input():
        return TransmitterInput(kg, per_frame_fn, receiver_node, k_triples)

    transmitter = TripleActorCritic(
        kg,
        obs_dim=env.observation_space.shape[0],
        n_actions=N_SYMBOLS,
        k_triples=k_triples + 1,  # + the reserved receiver slot
    )
    receiver = ReceiverActorCritic(n_actions=int(env.action_space.n))
    print(
        f"Transmitter: {sum(p.numel() for p in transmitter.parameters()):,} params, "
        f"{N_SYMBOLS} symbols {''.join(SYMBOL_NAMES)}"
    )
    print(f"Receiver:    {sum(p.numel() for p in receiver.parameters()):,} params")

    initial_node_symbols = transmitter.symbol_table.node_symbols.detach().cpu().clone()
    if init_from:
        transmitter.load_state_dict(
            torch.load(os.path.join(init_from, "transmitter_final.pt"),
                       map_location=device, weights_only=True))
        receiver.load_state_dict(
            torch.load(os.path.join(init_from, "receiver_final.pt"),
                       map_location=device, weights_only=True))
        print(f"Warm-started both agents from {init_from}")

    trainer = CommunicationTrainer(
        transmitter, receiver, env, make_tx_input,
        device=device, horizon=horizon, lr=lr,
        perceive_passes=perceive_passes, record_trajectories=True,
        normalize_value_loss=normalize_value,
    )
    print(f"normalize_value_loss={normalize_value}  perceive_passes={perceive_passes}")

    def record_video(it):
        path = os.path.join(video_dir, f"episode_iter{it:04d}.mp4")
        outcome, ep_ret = record_communication_episode(
            path, env, transmitter, receiver, kg, make_tx_input, perceive_passes,
            device, tag=f"iter {it}",
        )
        trainer._obs = None  # recording used the env; force a fresh episode
        print(f"Wrote {path} (outcome={outcome}, return={ep_ret:+.2f})")

    metric_rows, trajectories = [], []
    for it in range(1, iterations + 1):
        stats = trainer.collect_and_update()
        stats["iteration"] = it
        metric_rows.append(stats)
        Diagnostics.append_metrics(metrics_path, stats)
        for traj in trainer.trajectories:
            traj["iteration"] = it
        trajectories.extend(trainer.trajectories)
        trajectories = trajectories[-200:]
        trainer.trajectories = []

        top = np.argsort(stats["symbol_fracs"])[::-1][:3]
        syms = " ".join(f"{SYMBOL_NAMES[i]}={stats['symbol_fracs'][i]:.2f}" for i in top)
        acts = " ".join(
            f"{n[0]}={f:.2f}" for n, f in zip(ACTION_LABELS, stats["action_fracs"])
        )
        print(
            f"iter {it}/{iterations}: return={stats['mean_return']:+.2f} "
            f"len={stats['mean_length']:.0f} food={stats['food_rate']:.2f} "
            f"death={stats['death_rate']:.2f} "
            f"| caged_food={stats['caged_food_rate']:.2f} "
            f"loose_food={stats['loose_food_rate']:.2f} "
            f"loose_death={stats['loose_death_rate']:.2f} "
            f"| TX ent={stats['tx_entropy']:.2f} ev={stats['tx_explained_var']:+.2f} "
            f"| RX ent={stats['rx_entropy']:.2f} ev={stats['rx_explained_var']:+.2f} "
            f"| Hsym={stats['symbol_entropy']:.2f} [{syms}] acts[{acts}]"
        )

        if it in record_iters:
            record_video(it)

        if checkpoint_every and it % checkpoint_every == 0 and it < iterations:
            torch.save(transmitter.state_dict(),
                       os.path.join(ckpt_dir, f"iter{it:04d}_transmitter.pt"))
            torch.save(receiver.state_dict(),
                       os.path.join(ckpt_dir, f"iter{it:04d}_receiver.pt"))
            print(f"Checkpoint saved at iteration {it}")
            Diagnostics.save_training_curves(metric_rows, plot_dir)

    tx_path = Diagnostics.unique_path(os.path.join(run_dir, "transmitter_final.pt"))
    rx_path = Diagnostics.unique_path(os.path.join(run_dir, "receiver_final.pt"))
    torch.save(transmitter.state_dict(), tx_path)
    torch.save(receiver.state_dict(), rx_path)
    kg.symbols = transmitter.symbol_table.node_symbols.detach().cpu().clone()
    kg.relation_symbols = transmitter.symbol_table.relation_symbols.detach().cpu().clone()
    kg.save(Diagnostics.unique_path(os.path.join(run_dir, "kg_trained.pt")))
    print(f"Saved {tx_path} and {rx_path}")

    last = metric_rows[-1]
    fracs = np.mean([r["symbol_fracs"] for r in metric_rows[-25:]], axis=0)
    lines = [
        f"Final iteration {last['iteration']}: return={last['mean_return']:+.2f} "
        f"food={last['food_rate']:.2f} death={last['death_rate']:.2f} "
        f"caged_food={last['caged_food_rate']:.2f} "
        f"loose_food={last['loose_food_rate']:.2f} "
        f"loose_death={last['loose_death_rate']:.2f}",
        "",
        "Symbol usage over the last 25 iterations "
        f"(uniform would be {1 / N_SYMBOLS:.2f} each, max entropy "
        f"{np.log(N_SYMBOLS):.2f}):",
    ]
    for i, name in enumerate(SYMBOL_NAMES):
        lines.append(f"  '{name}': {fracs[i]:.4f}")
    lines.append(f"  symbol entropy: {last['symbol_entropy']:.3f}")
    lines.append("")
    for i in range(kg.num_nodes):
        a = initial_node_symbols[i]
        b = transmitter.symbol_table.node_symbols.detach().cpu()[i]
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        lines.append(
            f"  node {i}: L2 delta={float((a - b).norm()):.3f} cosine={cos:+.3f}"
        )
    report = "\n".join(lines)
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)

    written = Diagnostics.save_training_curves(metric_rows, plot_dir)
    traj_plot = Diagnostics.save_trajectory_plot(trajectories, plot_dir)
    if traj_plot:
        written.append(traj_plot)
    for p in written:
        print(f"Wrote plot {p}")

    env.close()


def main(inputArguments):
    initialize(inputArguments)
    train()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
