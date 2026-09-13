#!/bin/python3
import sys, os, json
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from lib.Logger import Logger
from lib.ArgumentParser import ArgumentParser
from lib.KGWorldEnv import CommunicationKGWorldEnv
from lib.Perception import PerceptionPipeline
from lib.OracleGrounding import (
    build_oracle_kg,
    oracle_bodies,
    perceive_scene_softcat,
)
from lib.CategoryModel import load_category_checkpoint
from lib.SymbolPolicy import TripleActorCritic, DIALTransmitter
from lib.DIALPPOTrainer import DIALPPOTrainer
from lib.CommunicationChannel import CommunicationChannel
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
    "receiver_yaw_mode": "independent",
    "method": "independent_ppo",
    "channel_sigma": "2.0",
    "detach_channel": "0",
    "critic_detach": "0",
    "target_kl": "0.03",
    "video_channel": "hard",
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
Two-agent communication training. Arguments use key=value syntax.

method=independent_ppo preserves separate categorical-message PPO.
method=dial_ppo trains a two-bit noisy sigmoid channel jointly through receiver
PPO. Hard execution maps 00/01/10/11 to a/b/c/d. channel_sigma=2.0,
detach_channel=1 provides the matched no-cross-agent-gradient control;
critic_detach=1 stops only critic gradients at the channel. target_kl=0.03
limits DIAL PPO update drift (0 disables it).

Defaults: iterations=450 horizon=2048 k_triples=6 perceive_passes=1 lr=3e-4
checkpoint_every=25 videos=10 device=cuda seed=0 normalize_value=1
food_near_lion=0 food_near_lion_offset=1.5 resources=0.
receiver_yaw_mode=independent (default) or shared: shared copies the
transmitter's randomly sampled yaw at reset, fixing both for that episode.
video_channel=hard (or continuous for DIAL). Videos use a separate environment.
category_model defaults to output/category_model.pt. init_from is a directory
containing transmitter_final.pt and receiver_final.pt; this restores weights
only, not optimizer or environment state. Architecture must match.

Example: python3 scripts/train_communication.py method=dial_ppo run_name=dial_s0
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
    method = g_ArgParse.get("method")
    sigma = float(g_ArgParse.get("channel_sigma"))
    CommunicationChannel(sigma)  # validate before creating a run or loading models
    video_channel = g_ArgParse.get("video_channel")
    if method not in ("independent_ppo", "dial_ppo"):
        raise ValueError("method must be independent_ppo or dial_ppo")
    if video_channel not in ("hard", "continuous") or (method == "independent_ppo" and video_channel != "hard"):
        raise ValueError("continuous videos require dial_ppo; otherwise use hard")
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
    receiver_yaw_mode = g_ArgParse.get("receiver_yaw_mode")
    if receiver_yaw_mode not in ("independent", "shared"):
        raise ValueError("receiver_yaw_mode must be independent or shared")

    if iterations < 1 or horizon < 2 or k_triples < 1 or perceive_passes < 1:
        raise ValueError("positive iterations/k_triples/perceive_passes and horizon >= 2 required")
    if checkpoint_every < 0 or n_videos < 0:
        raise ValueError("checkpoint_every and videos must be nonnegative")
    if init_from:
        previous_config = os.path.join(init_from, "config.json")
        if os.path.exists(previous_config):
            with open(previous_config) as handle:
                previous = json.load(handle)
            if previous['method'] != method or int(previous['k_triples']) != k_triples:
                raise ValueError("Warm start method and triple count must match")
        elif method == "dial_ppo":
            raise ValueError("DIAL warm starts require versioned config.json")
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
    config = {key: g_ArgParse.get(key) for key in optional_arguments}
    config.update(format_version=1, symbol_names=list(SYMBOL_NAMES), context_length=10,
                  message_bits=2 if method == "dial_ppo" else None,
                  resume_semantics="weights_only", time_limit_terminal=True,
                  observation_dim=4, transmitter_hidden=38,
                  receiver_model_dim=4, receiver_blocks=3, receiver_heads=1,
                  category_model=os.path.abspath(category_model_path))
    with open(os.path.join(run_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)

    record_iters = set(segment_start_iters(iterations, n_videos))
    print(f"Recording one episode after iterations: {sorted(record_iters)}")

    env = CommunicationKGWorldEnv(
        seed=seed,
        food_near_lion_prob=food_near_lion,
        food_near_lion_offset=food_near_lion_offset,
        resources=resources,
        receiver_yaw_mode=receiver_yaw_mode,
    )
    video_env = None
    try:
        print(
            f"Env: food_near_lion_prob={food_near_lion} offset={food_near_lion_offset} "
            f"resources={resources} receiver_yaw_mode={receiver_yaw_mode}"
        )

        bodies = oracle_bodies(env)
        kg, body_to_node = build_oracle_kg(bodies, device=device, seed=seed)
        receiver_node = body_to_node["receiver"]
        print(f"KG: one node per body {bodies} -> ids {body_to_node}")
        kg.save(os.path.join(run_dir, "kg_init.pt"))

        category_model, blob = load_category_checkpoint(category_model_path, device=device)
        category_model.requires_grad_(False)
        print(f"Category head {category_model_path} (epoch {blob['epoch']})")
        print(f"Loading SAM + ResNet-18 ({device})...")
        pipeline = PerceptionPipeline(device=device)

        def per_frame_fn():
            return perceive_scene_softcat(env, pipeline, kg, category_model, body_to_node)

        # ONE factory, used by both the trainer and the video recorder, so the input
        # the videos show is assembled by the same code that produced it in training
        def make_tx_input():
            return TransmitterInput(kg, per_frame_fn, receiver_node, k_triples)

        model_cls = DIALTransmitter if method == "dial_ppo" else TripleActorCritic
        transmitter = model_cls(kg, obs_dim=env.observation_space.shape[0],
                                k_triples=k_triples + 1)
        transmitter.to(device)
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

        trainer_cls = DIALPPOTrainer if method == "dial_ppo" else CommunicationTrainer
        dial_options = dict(sigma=sigma, target_kl=float(g_ArgParse.get("target_kl")),
                            detach_channel=g_ArgParse.get("detach_channel") == "1",
                            critic_detach=g_ArgParse.get("critic_detach") == "1") if method == "dial_ppo" else {}
        trainer = trainer_cls(
            transmitter, receiver, env, make_tx_input,
            device=device, horizon=horizon, lr=lr,
            perceive_passes=perceive_passes, record_trajectories=True,
            normalize_value_loss=normalize_value, **dial_options,
        )
        print(f"normalize_value_loss={normalize_value}  perceive_passes={perceive_passes}")

        video_env = None
        if n_videos:
            video_env = CommunicationKGWorldEnv(
                seed=100_000 + seed, food_near_lion_prob=food_near_lion,
                food_near_lion_offset=food_near_lion_offset, resources=resources,
                receiver_yaw_mode=receiver_yaw_mode)

        def make_video_input():
            return TransmitterInput(kg, lambda: perceive_scene_softcat(
                video_env, pipeline, kg, category_model, body_to_node), receiver_node, k_triples)

        def record_video(it):
            path = os.path.join(video_dir, f"episode_iter{it:04d}.mp4")
            # Evaluation must not advance the training policy's random stream.
            # manual_seed seeds all CUDA devices, so preserve all their generators.
            cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            tx_training, rx_training = transmitter.training, receiver.training
            try:
                transmitter.eval()
                receiver.eval()
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(100_000 + seed + it)
                    outcome, ep_ret = record_communication_episode(
                        path, video_env, transmitter, receiver, kg, make_video_input,
                        perceive_passes, device, tag=f"iter {it}", method=method,
                        sigma=sigma, channel_mode=video_channel)
            finally:
                transmitter.train(tx_training)
                receiver.train(rx_training)
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
            tx_status = (f"TX actor_grad={stats['tx_actor_grad_norm']:.3g} "
                         f"value_grad={stats['tx_value_grad_norm']:.3g}" if method == "dial_ppo"
                         else f"TX ent={stats['tx_entropy']:.2f} ev={stats['tx_explained_var']:+.2f}")
            print(
                f"iter {it}/{iterations}: return={stats['mean_return']:+.2f} "
                f"len={stats['mean_length']:.0f} food={stats['food_rate']:.2f} "
                f"death={stats['death_rate']:.2f} "
                f"| caged_food={stats['caged_food_rate']:.2f} "
                f"loose_food={stats['loose_food_rate']:.2f} "
                f"loose_death={stats['loose_death_rate']:.2f} "
                f"| {tx_status} "
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

    finally:
        if video_env is not None:
            video_env.close()
        env.close()


def main(inputArguments):
    if inputArguments in (["--help"], ["-h"]):
        print(USAGE)
        return
    initialize(inputArguments)
    train()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
