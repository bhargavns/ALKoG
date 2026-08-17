#!/bin/python3
import sys, os
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
from lib.SymbolicKG import SymbolicKG
from lib.Oracle import (
    NodeIdentityScorer,
    VERDICT_MERGE,
    VERDICT_SPLIT,
    BACKGROUND,
    AMBIGUOUS,
)
from lib import Diagnostics

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_TROY_DEV = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REAL = lambda label: label not in (BACKGROUND, AMBIGUOUS)  # noqa: E731
_DIST_KEY = {"lion": "dist_lion", "food": "dist_food", "cage": "dist_cage"}

required_arguments = ["run_dir"]
optional_arguments = {
    "thresholds": "0.40:0.96:0.02",  # start:stop:step, or a comma-separated list
    "color_weight": "0.5",      # weight on the colour-histogram half (0.5 = as recorded)
    "consolidate_threshold": "0.85",
    "min_count": "5",
    "device": "cuda",
    "seed": "0",
}

USAGE = """
eval_kg_oracle.py -- replay KG clustering offline against ground truth.

Reads the detection log a build_kg.py run wrote with oracle=1 and re-runs the
exact same online clustering (SymbolicKG.match_or_add + consolidate) over the
stored embeddings at many similarity thresholds, scoring every assignment
against MuJoCo's segmentation ground truth. No SAM, no GPU rendering -- so a
threshold sweep costs seconds instead of a full rebuild.

Writes into <run_dir>/oracle/:
    sweep.jsonl        one row per threshold
    sweep_report.txt   the readable version, with a recommended threshold
    plots/             threshold sweep, separability, distance, confusion

    Required:
        run_dir=...     a build_kg run directory (or its oracle/ subdirectory)

    Optional:
        thresholds=0.40:0.96:0.02 consolidate_threshold=0.85 min_count=5
        device=cuda seed=0

    Example Usage:
        eval_kg_oracle.py run_dir=output/runs/kg_20260730_101500
        eval_kg_oracle.py run_dir=... thresholds=0.45,0.50,0.55
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def parse_thresholds(spec):
    if ":" in spec:
        start, stop, step = (float(v) for v in spec.split(":"))
        return [round(v, 4) for v in np.arange(start, stop + 1e-9, step)]
    return [round(float(v), 4) for v in spec.split(",")]


def reweight(embeddings, color_weight, cnn_dim=512):
    """Rebalance the CNN and colour-histogram halves of a stored embedding.

    Detections are recorded as [sqrt(.5)*cnn, sqrt(.5)*hist] with each half unit
    norm, so cosine similarity is an equal blend. Renormalizing each half and
    rescaling by sqrt(1-w) / sqrt(w) makes the cosine (1-w)*cnn + w*colour --
    letting the weighting be swept offline without re-running perception.
    """
    cnn, his = embeddings[:, :cnn_dim], embeddings[:, cnn_dim:]
    cnn = cnn / np.maximum(1e-8, np.linalg.norm(cnn, axis=1, keepdims=True))
    his = his / np.maximum(1e-8, np.linalg.norm(his, axis=1, keepdims=True))
    return np.concatenate(
        [np.sqrt(1.0 - color_weight) * cnn, np.sqrt(color_weight) * his], axis=1
    )


def load_run(run_dir):
    oracle_dir = run_dir if os.path.basename(run_dir) == "oracle" else os.path.join(run_dir, "oracle")
    det_path = os.path.join(oracle_dir, "detections.jsonl")
    emb_path = os.path.join(oracle_dir, "embeddings.npy")
    if not os.path.exists(det_path):
        raise FileNotFoundError(f"{det_path} not found -- was build_kg.py run with oracle=1?")
    with open(det_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"{emb_path} not found -- cannot replay without embeddings")
    embeddings = np.load(emb_path).astype(np.float32)
    if len(embeddings) != len(records):
        raise ValueError(
            f"{len(embeddings)} embeddings vs {len(records)} detections -- logs are out of sync"
        )
    return oracle_dir, records, embeddings


# ------------------------------------------------------------------ replay


def replay(embeddings, labels, threshold, consolidate_threshold, min_count, device, seed):
    """Re-run online clustering at one threshold; score every assignment.

    Uses the real SymbolicKG so the replay cannot drift from the online path.
    """
    kg = SymbolicKG(
        embedding_dim=embeddings.shape[1],
        similarity_threshold=threshold,
        device=device,
        seed=seed,
    )
    scorer = NodeIdentityScorer()
    per_detection = []
    for emb, label in zip(embeddings, labels):
        node_id, created, best_sim, sims = kg.match_or_add(emb)
        verdict, rival, rival_sim = scorer.judge(label, node_id, created, best_sim, sims)
        scorer.update(node_id, label, verdict)
        per_detection.append(
            {"node": node_id, "created": created, "best_sim": best_sim, "verdict": verdict}
        )

    nodes_before = kg.num_nodes
    remap = kg.consolidate(merge_threshold=consolidate_threshold, min_count=min_count)
    _, groups_before = scorer.label_groups()
    _, groups_after = scorer.label_groups(remap)

    def extra(groups):
        # nodes beyond the one each real object deserves; missing objects count as 1 error
        real = {k: v for k, v in groups.items() if _REAL(k)}
        return sum(len(v) - 1 for v in real.values())

    t = scorer.totals()
    row = {
        "threshold": threshold,
        "nodes": nodes_before,
        "nodes_after_consolidation": kg.num_nodes,
        "false_split": t["false_split"],
        "false_merge": t["false_merge"],
        "ok": t["ok"],
        "scored": t["scored"],
        "objects_found": sum(1 for k in groups_before if _REAL(k)),
        "extra_nodes": extra(groups_before),
        "extra_nodes_after_consolidation": extra(groups_after),
        "impure_nodes": sum(
            1
            for n, h in scorer.node_labels.items()
            if h.most_common(1)[0][1] < 0.9 * sum(h.values())
        ),
    }
    return row, kg, scorer, per_detection, remap


# ------------------------------------------------------------------ plots


def plot_sweep(rows, n_objects, plot_dir):
    Diagnostics.apply_style()
    thr = [r["threshold"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    scored = np.maximum(1, np.array([r["scored"] for r in rows]))
    ax.plot(thr, 100 * np.array([r["false_split"] for r in rows]) / scored,
            color=Diagnostics.SERIES[0], label="false split")
    ax.plot(thr, 100 * np.array([r["false_merge"] for r in rows]) / scored,
            color=Diagnostics.STATUS_CRITICAL, label="false merge")
    ax.set_xlabel("similarity threshold")
    ax.set_ylabel("% of scored detections")
    ax.set_title("Splitting vs merging errors")
    ax.legend()

    ax = axes[1]
    ax.plot(thr, [r["nodes"] for r in rows], color=Diagnostics.SERIES[0], label="nodes")
    ax.plot(thr, [r["nodes_after_consolidation"] for r in rows],
            color=Diagnostics.SERIES[1], label="after consolidation")
    ax.axhline(n_objects, color=Diagnostics.BASELINE, ls="--",
               label=f"ideal ({n_objects} objects)")
    ax.set_yscale("symlog")
    ax.set_xlabel("similarity threshold")
    ax.set_ylabel("node count")
    ax.set_title("Graph size vs ground truth")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(plot_dir, "oracle_threshold_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_separability(scorer, threshold, plot_dir):
    same = np.asarray(scorer.label_sims["same"])
    diff = np.asarray(scorer.label_sims["diff"])
    if same.size == 0 and diff.size == 0:
        return None
    Diagnostics.apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(same.min() if same.size else 1.0, diff.min() if diff.size else 1.0), 1.0, 60
    )
    if same.size:
        ax.hist(same, bins=bins, color=Diagnostics.SERIES[0], alpha=0.75, label="same object")
    if diff.size:
        ax.hist(diff, bins=bins, color=Diagnostics.STATUS_CRITICAL, alpha=0.75,
                label="different object")
    ax.axvline(threshold, color=Diagnostics.INK, ls="--", label=f"threshold {threshold}")
    ax.set_xlabel("best-match cosine similarity")
    ax.set_ylabel("detections")
    ax.set_title("Can one threshold separate same from different? (overlap = no)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plot_dir, "oracle_separability.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_distance(records, per_detection, plot_dir):
    """Is splinter a distance/apparent-size effect? Colour by verdict."""
    pts = defaultdict(lambda: ([], [], []))
    for rec, det in zip(records, per_detection):
        label = rec["label"]
        key = _DIST_KEY.get(label)
        if key is None or key not in rec or det["best_sim"] < 0:
            continue  # best_sim -1.0 is the "no nodes existed yet" sentinel
        bucket = det["verdict"] if det["verdict"] in (VERDICT_SPLIT, VERDICT_MERGE) else "ok"
        d, s, a = pts[bucket]
        d.append(rec[key])
        s.append(det["best_sim"])
        a.append(rec["mask_px"])
    if not pts:
        return None
    Diagnostics.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {
        "ok": Diagnostics.SERIES[1],
        VERDICT_SPLIT: Diagnostics.SERIES[0],
        VERDICT_MERGE: Diagnostics.STATUS_CRITICAL,
    }
    for bucket, (d, s, a) in pts.items():
        axes[0].scatter(d, s, s=10, alpha=0.5, color=colors.get(bucket), label=bucket)
        axes[1].scatter(a, s, s=10, alpha=0.5, color=colors.get(bucket), label=bucket)
    axes[0].set_xlabel("agent distance to the true object (m)")
    axes[1].set_xlabel("detection mask size (px)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel("best-match similarity")
        ax.legend()
    axes[0].set_title("Similarity vs distance")
    axes[1].set_title("Similarity vs apparent size")
    fig.tight_layout()
    path = os.path.join(plot_dir, "oracle_similarity_vs_distance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion(scorer, plot_dir):
    nodes = sorted(scorer.node_labels)
    labels = sorted({l for h in scorer.node_labels.values() for l in h})
    if not nodes or not labels:
        return None
    mat = np.zeros((len(nodes), len(labels)))
    for i, n in enumerate(nodes):
        for j, l in enumerate(labels):
            mat[i, j] = scorer.node_labels[n][l]
    Diagnostics.apply_style()
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 3, 0.32 * len(nodes) + 2.2))
    ax.imshow(mat / np.maximum(1, mat.sum(axis=1, keepdims=True)),
              cmap=Diagnostics.SEQ_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(nodes)), [f"n{n}" for n in nodes])
    ax.set_title("Node composition by true object\n(a clean KG is one bright cell per row,\n"
                 "and one row per object)")
    ax.grid(False)
    fig.tight_layout()
    path = os.path.join(plot_dir, "oracle_node_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ main


def evaluate():
    run_dir = g_ArgParse.get("run_dir")
    thresholds = parse_thresholds(g_ArgParse.get("thresholds"))
    consolidate_threshold = float(g_ArgParse.get("consolidate_threshold"))
    min_count = int(g_ArgParse.get("min_count"))
    device = g_ArgParse.get("device")
    seed = int(g_ArgParse.get("seed"))
    color_weight = float(g_ArgParse.get("color_weight"))

    oracle_dir, records, embeddings = load_run(run_dir)
    if abs(color_weight - 0.5) > 1e-9:
        embeddings = reweight(embeddings, color_weight)
        print(f"Reweighted embeddings: colour {color_weight:.2f} / CNN {1 - color_weight:.2f}")
    plot_dir = os.path.join(oracle_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Loaded {len(records)} detections ({embeddings.shape[1]}-dim) from {oracle_dir}")

    labels = [r["label"] for r in records]
    label_counts = Counter(labels)
    n_objects = sum(1 for l in label_counts if _REAL(l))
    print(f"Ground truth objects seen: {n_objects} ({dict(label_counts)})")

    embeddings_t = torch.from_numpy(embeddings).to(device)
    sweep_path = os.path.join(oracle_dir, "sweep.jsonl")
    open(sweep_path, "w").close()

    rows, best = [], None
    for threshold in thresholds:
        row, kg, scorer, per_detection, remap = replay(
            embeddings_t, labels, threshold, consolidate_threshold, min_count, device, seed
        )
        rows.append(row)
        Diagnostics.append_metrics(sweep_path, row)
        print(
            f"  threshold {threshold:.2f}: {row['nodes']:>4d} nodes "
            f"({row['nodes_after_consolidation']:>3d} after consolidation), "
            f"split {row['false_split']:>4d}, merge {row['false_merge']:>4d}, "
            f"extra nodes {row['extra_nodes_after_consolidation']:>3d}"
        )
        # objective: get one node per object, breaking ties on total error
        key = (
            row["extra_nodes_after_consolidation"] + 10 * row["false_merge"] / max(1, row["scored"]),
            row["false_split"] + row["false_merge"],
        )
        if best is None or key < best[0]:
            best = (key, threshold, kg, scorer, per_detection, remap)

    _, best_threshold, best_kg, best_scorer, best_dets, best_remap = best
    written = [
        plot_sweep(rows, n_objects, plot_dir),
        plot_separability(best_scorer, best_threshold, plot_dir),
        plot_distance(records, best_dets, plot_dir),
        plot_confusion(best_scorer, plot_dir),
    ]

    online = Counter(r["verdict"] for r in records)
    lines = [
        "=" * 78,
        "OFFLINE THRESHOLD SWEEP vs GROUND TRUTH",
        "=" * 78,
        f"source: {oracle_dir}",
        f"detections: {len(records)}   real objects seen: {n_objects}",
        f"consolidate_threshold={consolidate_threshold}  min_count={min_count}",
        "",
        f"online run recorded: false_split={online[VERDICT_SPLIT]} "
        f"false_merge={online[VERDICT_MERGE]}",
        "",
        f"{'thresh':>7s} {'nodes':>6s} {'consol':>7s} {'split':>6s} {'merge':>6s} "
        f"{'extra':>6s} {'extra_c':>8s} {'impure':>7s}",
    ]
    for r in rows:
        lines.append(
            f"{r['threshold']:>7.2f} {r['nodes']:>6d} {r['nodes_after_consolidation']:>7d} "
            f"{r['false_split']:>6d} {r['false_merge']:>6d} {r['extra_nodes']:>6d} "
            f"{r['extra_nodes_after_consolidation']:>8d} {r['impure_nodes']:>7d}"
        )
    lines += [
        "",
        f"RECOMMENDED threshold: {best_threshold:.2f} "
        f"(fewest surplus nodes after consolidation, then fewest total errors)",
        "",
    ]
    lines += best_scorer.purity_lines(header=f"-- NODE PURITY at threshold {best_threshold:.2f}")
    lines.append("")
    lines += best_scorer.separability_lines()
    lines += ["", "-- AT THE RECOMMENDED THRESHOLD, AFTER CONSOLIDATION"]
    lines += best_scorer.purity_lines(remap=best_remap, header="")[1:]
    lines.append("=" * 78)
    report = "\n".join(lines)

    report_path = os.path.join(oracle_dir, "sweep_report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    for p in written:
        if p:
            print(f"Wrote plot {p}")
    print(f"Wrote {sweep_path}")
    print(f"Wrote {report_path}")


def main(inputArguments):
    initialize(inputArguments)
    evaluate()
    print(f"Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
