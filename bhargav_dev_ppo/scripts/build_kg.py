#!/bin/python3
import sys, os
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from lib.Logger import Logger
from lib.ArgumentParser import ArgumentParser
from lib.KGWorldEnv import KGWorldEnv
from lib.Perception import PerceptionPipeline
from lib.SymbolicKG import SymbolicKG
from lib.Grounding import perceive_scene, draw_debug_frame
from lib.VideoRecorder import PanoramaRecorder, record_points
from lib import Diagnostics

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

required_arguments = []
optional_arguments = {
    "episodes": "12",           # cage state alternates caged/loose each episode
    "steps": "120",             # env steps per episode
    "perceive_every": "15",     # run SAM+CNN every N steps
    "action_hold": "15",        # resample the random action every N steps
    "threshold": "0.80",        # cosine similarity for merging into a concept node
    "consolidate_threshold": "0.85",  # post-pass: merge splintered nodes above this
    "prune_frac": "0.01",       # post-pass: prune nodes below this fraction of detections
    "min_count": "5",           # floor for the exposure-scaled prune threshold
    "out_dir": os.path.join(_TROY_DEV, "output", "runs"),
    "run_name": "",             # optional suffix on the run directory name
    "debug_images": "8",        # save this many annotated perception passes
    "verbose": "1",             # 1 = log every perception pass in detail
    "device": "cuda",
    "seed": "0",
}

USAGE = """
build_kg.py -- Phase 1: build the symbolic knowledge graph.

Drives the agent with a persistent random policy, periodically runs the
panoramic perception stack (SAM boxes -> frozen ResNet-18 embeddings), merges
detections into concept nodes by cosine similarity, and records geometric
relation edges ('inside', 'near'). Every node and relation type gets a random
4-dim symbol.

Each invocation writes to a fresh timestamped run directory under `out_dir`
(existing runs are never overwritten):
    kg.pt            the graph
    metrics.jsonl    per-episode growth metrics
    kg_report.txt    verbose dump of nodes/symbols/edges
    plots/           similarity heatmap, symbol tables, graph diagram, growth
    exemplars/       one representative crop per concept node
    debug/           annotated perception frames
    videos/          agent's-eye mp4 (4-cam panorama + perception overlay)
                     for 5 evenly spaced episodes (first ... last)

    Required:

    Optional:
        episodes=12 steps=120 perceive_every=15 action_hold=15
        threshold=0.80 consolidate_threshold=0.85 prune_frac=0.01 min_count=5
        out_dir=.../output/runs run_name= debug_images=8 verbose=1
        device=cuda seed=0

The consolidation prune is exposure-scaled: nodes with fewer than
max(min_count, prune_frac * total_detections) detections are dropped, so the
surviving node set is stable with respect to episode count.

    Example Usage:
        build_kg.py episodes=20 threshold=0.85 run_name=high_thresh
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def build_kg():
    episodes = int(g_ArgParse.get("episodes"))
    steps = int(g_ArgParse.get("steps"))
    perceive_every = int(g_ArgParse.get("perceive_every"))
    action_hold = int(g_ArgParse.get("action_hold"))
    threshold = float(g_ArgParse.get("threshold"))
    consolidate_threshold = float(g_ArgParse.get("consolidate_threshold"))
    prune_frac = float(g_ArgParse.get("prune_frac"))
    min_count = int(g_ArgParse.get("min_count"))
    out_dir = g_ArgParse.get("out_dir")
    run_name = g_ArgParse.get("run_name")
    debug_images = int(g_ArgParse.get("debug_images"))
    verbose = int(g_ArgParse.get("verbose"))
    device = g_ArgParse.get("device")
    seed = int(g_ArgParse.get("seed"))

    os.makedirs(out_dir, exist_ok=True)
    run_dir = Diagnostics.make_run_dir(out_dir, "kg", run_name)
    debug_dir = os.path.join(run_dir, "debug")
    exemplar_dir = os.path.join(run_dir, "exemplars")
    plot_dir = os.path.join(run_dir, "plots")
    video_dir = os.path.join(run_dir, "videos")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    for d in (debug_dir, exemplar_dir, plot_dir, video_dir):
        os.makedirs(d, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # agent's-eye videos for 5 evenly spaced episodes (first ... last)
    record_eps = set(record_points(episodes - 1)) if episodes > 0 else set()
    print(f"Recording agent's-eye video for episodes: {sorted(record_eps)}")

    rng = np.random.default_rng(seed)
    env = KGWorldEnv(seed=seed)
    print("Loading SAM + ResNet-18 (GPU)...")
    pipeline = PerceptionPipeline(device=device)
    kg = SymbolicKG(similarity_threshold=threshold, device=device, seed=seed)

    # best (largest-area) crop per concept, kept for "what did this node see" diagnostics
    exemplars = {}  # concept_id -> (area, BGR crop)

    passes_saved = 0
    total_detections = 0
    metric_rows = []
    for ep in range(episodes):
        env.fixed_cage_state = (ep % 2 == 0)  # guarantee both cage states are seen
        env.reset()
        nodes_at_ep_start = kg.num_nodes
        ep_detections = 0
        ep_passes = 0
        recorder = None
        if ep in record_eps:
            recorder = PanoramaRecorder(
                os.path.join(video_dir, f"episode_{ep:03d}.mp4"),
                frame_size=env.render_size, fps=20,
            )
        action = rng.uniform(-1, 1, size=3)
        for step in range(steps):
            if step % action_hold == 0:
                action = rng.uniform(-1, 1, size=3)
            env.step(action)

            if step % perceive_every != 0:
                if recorder is not None:
                    recorder.add(env.render_panorama(), hud_lines=[
                        f"phase1 ep{ep} step {step}  lion_caged={env.lion_caged}",
                        f"nodes={kg.num_nodes} edges={len(kg.edges)}",
                    ])
                continue
            ep_passes += 1
            nodes_before = kg.num_nodes
            per_frame = perceive_scene(env, pipeline, kg, allow_add=True)
            if recorder is not None:
                # reuse the frames perception just rendered; overlays are live
                recorder.add(
                    [pf[2] for pf in per_frame],
                    [pf[0] for pf in per_frame],
                    [pf[1] for pf in per_frame],
                    hud_lines=[
                        f"phase1 ep{ep} step {step}  lion_caged={env.lion_caged}"
                        "  [perception pass]",
                        f"nodes={kg.num_nodes} edges={len(kg.edges)}",
                    ],
                )
            for cam_idx, (concept_boxes, relations, frame) in enumerate(per_frame):
                total_detections += len(concept_boxes)
                ep_detections += len(concept_boxes)
                for src, rel_name, dst in relations:
                    kg.add_edge(src, rel_name, dst)
                for cid, (x0, y0, x1, y1) in concept_boxes.items():
                    area = (x1 - x0) * (y1 - y0)
                    if area > exemplars.get(cid, (0, None))[0]:
                        crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
                        if crop.size:
                            exemplars[cid] = (area, crop.copy())
                if verbose and (concept_boxes or relations):
                    rel_str = ", ".join(f"n{s} {r} n{d}" for s, r, d in relations) or "-"
                    print(
                        f"  ep{ep} step{step} cam{cam_idx}: concepts "
                        f"{sorted(concept_boxes.keys())} relations [{rel_str}]"
                    )
                if passes_saved < debug_images and concept_boxes:
                    img = draw_debug_frame(frame, concept_boxes, relations)
                    cv2.imwrite(
                        os.path.join(debug_dir, f"ep{ep}_step{step}_cam{cam_idx}.png"), img
                    )
                    passes_saved += 1
            if verbose and kg.num_nodes > nodes_before:
                print(
                    f"  ep{ep} step{step}: {kg.num_nodes - nodes_before} new concept "
                    f"node(s) created -> {kg.num_nodes} total"
                )
        if recorder is not None:
            recorder.close()
            print(f"Wrote agent's-eye video {recorder.path}")
        row = {
            "episode": ep,
            "lion_caged": bool(env.lion_caged),
            "nodes": kg.num_nodes,
            "new_nodes": kg.num_nodes - nodes_at_ep_start,
            "distinct_edges": len(kg.edges),
            "edge_observations": int(sum(kg.edges.values())),
            "detections": ep_detections,
            "perception_passes": ep_passes,
        }
        metric_rows.append(row)
        Diagnostics.append_metrics(metrics_path, row)
        print(
            f"episode {ep + 1}/{episodes} (lion_caged={env.lion_caged}): "
            f"{kg.num_nodes} nodes (+{row['new_nodes']}), {len(kg.edges)} distinct edges, "
            f"{ep_detections} detections in {ep_passes} passes"
        )

    print(f"\nBefore consolidation: {kg.num_nodes} nodes, {len(kg.edges)} distinct edges")
    # exposure-scaled prune: junk recurs slowly but indefinitely, so an absolute
    # threshold lets it accrete legitimacy in long runs; a detection-share
    # threshold keeps the surviving set stable regardless of episode count
    min_count_eff = max(min_count, round(prune_frac * total_detections))
    print(
        f"Prune threshold: {min_count_eff} detections "
        f"(max of floor {min_count} and {prune_frac:.1%} of {total_detections})"
    )
    remap = kg.consolidate(merge_threshold=consolidate_threshold, min_count=min_count_eff)
    merged = sum(1 for old, new in remap.items() if new is not None and old != new)
    pruned = sum(1 for new in remap.values() if new is None)
    print(f"Consolidated: merged/renumbered {merged}, pruned {pruned} rare nodes")
    if verbose:
        for old in sorted(remap):
            fate = f"-> node {remap[old]}" if remap[old] is not None else "pruned"
            print(f"  pre-consolidation node {old} {fate}")

    # remap exemplars through the consolidation and keep the largest per new id
    final_exemplars = {}
    for old, (area, crop) in exemplars.items():
        new = remap.get(old)
        if new is not None and area > final_exemplars.get(new, (0, None))[0]:
            final_exemplars[new] = (area, crop)
    for cid, (_, crop) in final_exemplars.items():
        cv2.imwrite(os.path.join(exemplar_dir, f"node{cid}.png"), crop)
    print(f"Saved {len(final_exemplars)} exemplar crops to {exemplar_dir}")

    kg_path = os.path.join(run_dir, "kg.pt")
    kg.save(kg_path)
    print(f"Saved KG to {kg_path} ({total_detections} concept detections processed)")

    report = Diagnostics.kg_report(kg, exemplar_dir=exemplar_dir)
    with open(os.path.join(run_dir, "kg_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)

    written = Diagnostics.save_kg_plots(kg, plot_dir)
    written.append(Diagnostics.save_phase1_curves(metric_rows, plot_dir))
    for p in written:
        print(f"Wrote plot {p}")
    env.close()


def main(inputArguments):
    initialize(inputArguments)
    build_kg()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
