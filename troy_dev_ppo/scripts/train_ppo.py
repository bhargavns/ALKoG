#!/bin/python3
import sys, os
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
from lib.KGWorldEnv import KinematicKGWorldEnv
from lib.Perception import PerceptionPipeline
from lib.SymbolicKG import SymbolicKG
from lib.Grounding import perceive_scene_geo, assemble_triples_geo
from lib.SymbolPolicy import TripleActorCritic
from lib.TransformerPolicy import TransformerActorCritic
from lib.PPOTrainer import PPOTrainer
from lib.VideoRecorder import record_points, record_policy_episode
from lib import Diagnostics

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OUTPUT = os.path.join(_TROY_DEV, "output")

ACTION_LABELS = ("forward", "backward", "left", "right")

required_arguments = []
optional_arguments = {
    "kg": "latest",           # path to phase-1 kg.pt, or 'latest' for newest run
    "iterations": "150",
    "horizon": "2048",
    "k_triples": "6",
    "perceive_every": "75",   # also re-perceive every N steps within an episode
    "lr": "3e-4",
    "checkpoint_every": "25", # save a policy checkpoint every N iterations
    "out_dir": os.path.join(_OUTPUT, "runs"),
    "run_name": "",           # optional suffix on the run directory name
    "device": "cuda",
    "seed": "0",
    "init_from": "",          # warm-start model weights from a saved policy .pt
    "model": "slot",          # policy architecture: slot (default) | transformer
    "normalize_value": "1",   # standardize value loss to the advantage scale
    "food_near_lion": "0.0",  # prob. food spawns near the lion (conflict layout)
    "food_near_lion_offset": "1.5",  # gap (units) so both stay perceivable to SAM
}

USAGE = """
train_ppo.py -- Phase 2: discrete PPO over distance-gated KG symbol triples.

Loads the phase-1 knowledge graph (structure and visual embeddings frozen),
then trains separate actor and critic MLPs (single hidden layer of 38) on a
kinematic, physics-free arena: 4 egocentric unit moves (forward/backward/
left/right), no momentum.

Network input (76 = 4 + k_triples*12 at defaults):
  env vector (4):  [x/8, y/8, cos(yaw), sin(yaw)]  -- no object positions
  kg vector (72):  up to k_triples=6 perceived triples; per triple
                   [src || rel || dst], each token = 4-dim symbol + slot
                   positional embedding, node tokens gated (elementwise *)
                   by a shared learned 2->4 projection (identity init) of
                   the node's agent-frame (forward, left) ground-plane
                   offset, estimated from the union box's bottom edge
                   (vision-only range; no oracle positions).

Perception (SAM + CNN + relation heuristics) re-runs at every episode reset
and every perceive_every steps. Between passes each node's offset is
re-derived every step from a world-frame anchor (agent position at the pass
plus the measured offset) and rotated into the agent's egocentric frame, so
the distance input tracks the agent's own movement instead of going stale
and aligns with the action set. Anchors persist across passes within an
episode (objects are static): a concept the current pass missed is kept in
the input as a lone entry with its remembered anchor, so an intermittently
detected object stays visible to the policy from its first sighting.
Anchor memory clears at reset. When more than k_triples relations are seen,
the top k by mean node-match cosine are kept; lone concepts fill remaining
slots as unary (concept, PAD, PAD) entries. PPO gradients update the node
and relation symbols, the positional embeddings, and the distance projection
alongside the networks.

Each invocation writes to a fresh timestamped run directory under `out_dir`
(existing runs and checkpoints are never overwritten):
    checkpoints/policy_iterNNNN.pt   periodic checkpoints
    policy_final.pt, kg_trained.pt   final artifacts
    metrics.jsonl                    per-iteration stats (incl. action fracs)
    report.txt                       verbose symbol drift + final summary
    plots/                           training curves, action stats,
                                     agent trajectories, symbol evolution
    videos/                          agent's-eye mp4 (4-cam panorama, live
                                     perception overlay, triples HUD) of one
                                     policy rollout at 5 evenly spaced
                                     iterations: 0 (untrained) ... final

    Required:

    Optional:
        kg=latest iterations=150 horizon=2048 k_triples=6 perceive_every=75
        lr=3e-4 checkpoint_every=25 out_dir=.../output/runs run_name=
        device=cuda seed=0 init_from=
            (init_from: path to a saved policy .pt to warm-start the model
            weights from -- e.g. a previous run's policy_final.pt, to extend
            training beyond that run's iteration budget. Starts a fresh run
            dir/metrics/optimizer; only the weights carry over. The symbol
            drift report still measures against the untouched phase-1 KG
            init, not the warm-start checkpoint.)

    Example Usage:
        train_ppo.py iterations=200 run_name=long_run
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
    kg_path = g_ArgParse.get("kg")
    iterations = int(g_ArgParse.get("iterations"))
    horizon = int(g_ArgParse.get("horizon"))
    k_triples = int(g_ArgParse.get("k_triples"))
    perceive_every = int(g_ArgParse.get("perceive_every"))
    lr = float(g_ArgParse.get("lr"))
    checkpoint_every = int(g_ArgParse.get("checkpoint_every"))
    out_dir = g_ArgParse.get("out_dir")
    run_name = g_ArgParse.get("run_name")
    device = g_ArgParse.get("device")
    seed = int(g_ArgParse.get("seed"))
    init_from = g_ArgParse.get("init_from")
    model_kind = g_ArgParse.get("model")
    normalize_value = g_ArgParse.get("normalize_value") == "1"
    food_near_lion = float(g_ArgParse.get("food_near_lion"))
    food_near_lion_offset = float(g_ArgParse.get("food_near_lion_offset"))

    if kg_path == "latest":
        kg_path = Diagnostics.latest_kg_path(_OUTPUT)
    os.makedirs(out_dir, exist_ok=True)
    run_dir = Diagnostics.make_run_dir(out_dir, "ppo", run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    plot_dir = os.path.join(run_dir, "plots")
    video_dir = os.path.join(run_dir, "videos")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    for d in (ckpt_dir, plot_dir, video_dir):
        os.makedirs(d, exist_ok=True)
    print(f"Run directory: {run_dir}")
    torch.manual_seed(seed)

    # agent's-eye videos at 5 evenly spaced points: iter 0 (untrained) ... final
    record_iters = set(record_points(iterations))
    print(f"Recording agent's-eye video at iterations: {sorted(record_iters)}")

    kg = SymbolicKG.load(kg_path, device=device)
    print(f"Loaded KG from {kg_path}: {kg.num_nodes} nodes, {len(kg.edges)} distinct edges")
    print("\n" + Diagnostics.kg_report(kg))

    env = KinematicKGWorldEnv(
        seed=seed,
        food_near_lion_prob=food_near_lion,
        food_near_lion_offset=food_near_lion_offset,
    )
    print(
        f"Env: food_near_lion_prob={food_near_lion} offset={food_near_lion_offset}"
    )
    print("Loading SAM + ResNet-18 (GPU)...")
    pipeline = PerceptionPipeline(device=device)

    def triple_fn():
        per_frame = perceive_scene_geo(env, pipeline, kg)
        return assemble_triples_geo(kg, per_frame, k=k_triples)

    ModelCls = TransformerActorCritic if model_kind == "transformer" else TripleActorCritic
    print(f"Policy architecture: {model_kind} ({ModelCls.__name__})")
    model = ModelCls(
        kg,
        obs_dim=env.observation_space.shape[0],
        n_actions=int(env.action_space.n),
        k_triples=k_triples,
    )
    # captured before any init_from load, so the drift report always measures
    # against the untouched phase-1 KG init, not a warm-start checkpoint
    initial_node_symbols = model.symbol_table.node_symbols.detach().cpu().clone()
    initial_rel_symbols = model.symbol_table.relation_symbols.detach().cpu().clone()

    if init_from:
        model.load_state_dict(torch.load(init_from, map_location=device, weights_only=True))
        print(f"Warm-started model weights from {init_from}")

    trainer = PPOTrainer(
        model, env, triple_fn, device=device, horizon=horizon, lr=lr,
        perceive_every_steps=perceive_every, record_trajectories=True,
        normalize_value_loss=normalize_value,
    )
    print(f"normalize_value_loss={normalize_value}")

    def record_video(it):
        path = os.path.join(video_dir, f"policy_iter{it:04d}.mp4")
        outcome, ep_ret = record_policy_episode(
            path, env, model, pipeline, kg, k_triples, perceive_every, device,
            tag=f"iter {it}",
        )
        # recording used the env; force the trainer to start a fresh episode
        trainer._obs = None
        print(f"Wrote agent's-eye video {path} (outcome={outcome}, return={ep_ret:+.2f})")

    if 0 in record_iters:
        record_video(0)

    metric_rows = []
    node_snaps = [initial_node_symbols.numpy().copy()]
    rel_snaps = [initial_rel_symbols.numpy().copy()]
    trajectories = []
    for it in range(1, iterations + 1):
        stats = trainer.collect_and_update()
        stats["iteration"] = it
        metric_rows.append(stats)
        Diagnostics.append_metrics(metrics_path, stats)
        node_snaps.append(model.symbol_table.node_symbols.detach().cpu().numpy().copy())
        rel_snaps.append(model.symbol_table.relation_symbols.detach().cpu().numpy().copy())
        for traj in trainer.trajectories:
            traj["iteration"] = it
        trajectories.extend(trainer.trajectories)
        trajectories = trajectories[-200:]
        trainer.trajectories = []

        fracs = " ".join(
            f"{name[0]}={frac:.2f}" for name, frac in zip(ACTION_LABELS, stats["action_fracs"])
        )
        near = (
            f" | NEAR caged_food={stats['near_caged_food_rate']:.2f} "
            f"loose_food={stats['near_loose_food_rate']:.2f} "
            f"loose_death={stats['near_loose_death_rate']:.2f}"
            if food_near_lion > 0 else ""
        )
        print(
            f"iter {it}/{iterations}: return={stats['mean_return']:+.2f} "
            f"len={stats['mean_length']:.0f} food={stats['food_rate']:.2f} "
            f"death={stats['death_rate']:.2f} timeout={stats['timeout_rate']:.2f} "
            f"| caged_food={stats['caged_food_rate']:.2f} "
            f"loose_food={stats['loose_food_rate']:.2f} "
            f"loose_death={stats['loose_death_rate']:.2f} "
            f"| pi={stats['pi_loss']:.3f} v={stats['v_loss']:.3f} "
            f"ent={stats['entropy']:.2f} kl={stats['approx_kl']:.4f} "
            f"clip={stats['clip_frac']:.2f} ev={stats['explained_var']:+.2f}"
            f"{near} | acts[{fracs}]"
        )

        if it in record_iters:
            record_video(it)

        if checkpoint_every and it % checkpoint_every == 0 and it < iterations:
            ckpt = Diagnostics.unique_path(os.path.join(ckpt_dir, f"policy_iter{it:04d}.pt"))
            torch.save(model.state_dict(), ckpt)
            print(f"Checkpoint saved to {ckpt}")
            Diagnostics.save_training_curves(metric_rows, plot_dir)
            Diagnostics.save_trajectory_plot(trajectories, plot_dir)

    policy_path = Diagnostics.unique_path(os.path.join(run_dir, "policy_final.pt"))
    torch.save(model.state_dict(), policy_path)

    final_node_symbols = model.symbol_table.node_symbols.detach().cpu()
    final_rel_symbols = model.symbol_table.relation_symbols.detach().cpu()
    kg.symbols = final_node_symbols.clone()
    kg.relation_symbols = final_rel_symbols.clone()
    kg_out = Diagnostics.unique_path(os.path.join(run_dir, "kg_trained.pt"))
    kg.save(kg_out)
    print(f"Saved policy to {policy_path} and updated KG to {kg_out}")

    drift_lines = ["Symbol drift (init -> trained):"]
    for i in range(kg.num_nodes):
        a, b = initial_node_symbols[i], final_node_symbols[i]
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        drift_lines.append(
            f"  node {i} (merged {kg.counts[i]} detections): "
            f"L2 delta={float((a - b).norm()):.3f} cosine={cos:+.3f}"
        )
        drift_lines.append(
            f"    init    = [" + ", ".join(f"{float(v):+.4f}" for v in a) + "]"
        )
        drift_lines.append(
            f"    trained = [" + ", ".join(f"{float(v):+.4f}" for v in b) + "]"
        )
    for i, name in enumerate(kg.relation_names):
        a, b = initial_rel_symbols[i], final_rel_symbols[i]
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        drift_lines.append(
            f"  relation '{name}': L2 delta={float((a - b).norm()):.3f} cosine={cos:+.3f}"
        )
        drift_lines.append(
            f"    init    = [" + ", ".join(f"{float(v):+.4f}" for v in a) + "]"
        )
        drift_lines.append(
            f"    trained = [" + ", ".join(f"{float(v):+.4f}" for v in b) + "]"
        )
    pos_emb = model.symbol_table.pos_embeddings.detach().cpu()
    for i, name in enumerate(("src", "rel", "dst")):
        drift_lines.append(
            f"  pos embedding '{name}' = ["
            + ", ".join(f"{float(v):+.4f}" for v in pos_emb[i]) + "]"
        )
    drift_report = "\n".join(drift_lines)
    print("\n" + drift_report)

    last = metric_rows[-1]
    summary = (
        f"Final iteration {last['iteration']}: return={last['mean_return']:+.2f} "
        f"food={last['food_rate']:.2f} death={last['death_rate']:.2f} "
        f"caged_food={last['caged_food_rate']:.2f} loose_food={last['loose_food_rate']:.2f} "
        f"loose_death={last['loose_death_rate']:.2f}"
    )
    with open(os.path.join(run_dir, "report.txt"), "w") as f:
        f.write(f"kg: {kg_path}\n{summary}\n\n{drift_report}\n\n")
        f.write(Diagnostics.kg_report(kg) + "\n")
    print(summary)

    written = []
    written += Diagnostics.save_training_curves(metric_rows, plot_dir)
    written += Diagnostics.save_action_plots(
        metric_rows, trainer.last_actions, plot_dir, act_labels=ACTION_LABELS
    )
    traj_plot = Diagnostics.save_trajectory_plot(trajectories, plot_dir)
    if traj_plot:
        written.append(traj_plot)
    written += Diagnostics.save_symbol_evolution(
        np.stack(node_snaps), np.stack(rel_snaps), kg.relation_names, kg.counts, plot_dir
    )
    np.savez(
        os.path.join(run_dir, "symbol_history.npz"),
        node_symbols=np.stack(node_snaps),
        relation_symbols=np.stack(rel_snaps),
    )
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
