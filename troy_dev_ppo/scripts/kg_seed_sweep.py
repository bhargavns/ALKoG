#!/bin/python3
import sys, os
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import glob
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
from lib.SymbolicKG import SymbolicKG
from kg_metrics import accuracy, bcubed_scores, node_labels, pairwise_scores

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_REAL = ("lion", "cage", "food")


def effective_min_count(total_detections, min_count, prune_frac):
    """build_kg scales the prune floor with exposure; mirror it exactly."""
    return max(min_count, round(prune_frac * total_detections))

required_arguments = ["runs"]
optional_arguments = {
    "thresholds": "0.50,0.55,0.60,0.65,0.70",
    "consolidate_threshold": "0.85",
    "min_count": "5",
    "prune_frac": "0.01",       # matches build_kg: prune below this share of detections
    "device": "cuda",
    "seed": "0",
    "out": "",
}

USAGE = """
kg_seed_sweep.py -- average KG quality over seeds, for each merge threshold.

The similarity threshold only affects how stored detections are clustered, so it
is swept offline by replaying each run's embeddings. The SEED changes the
trajectory and therefore the detections themselves, so each seed needs its own
build_kg run; point `runs` at them.

Reports mean +/- std across seeds of accuracy, BCubed precision/recall/F1 and
node count, before and after consolidation.

    Required:
        runs=<glob>   e.g. 'output/runs/kg_*_seed*'

    Optional:
        thresholds=0.50,0.55,0.60,0.65,0.70 consolidate_threshold=0.85
        min_count=5 prune_frac=0.01 device=cuda seed=0 out=<jsonl of per-run rows>
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def load(run_dir):
    oracle = os.path.join(run_dir, "oracle")
    recs = [json.loads(l) for l in open(os.path.join(oracle, "detections.jsonl"))]
    emb = np.load(os.path.join(oracle, "embeddings.npy")).astype(np.float32)
    return recs, emb


def score(nodes, labels):
    bp, br, bf = bcubed_scores(nodes, labels)
    pp, pr, pf = pairwise_scores(nodes, labels)
    return {
        "accuracy": accuracy(nodes, labels),
        "precision": bp,
        "recall": br,
        "f1": bf,
        "pairwise_precision": pp,
        "pairwise_recall": pr,
        "pairwise_f1": pf,
        "nodes": len(set(nodes)),
    }


def evaluate_run(recs, emb, threshold, consolidate_threshold, min_count, prune_frac,
                 device, seed):
    kg = SymbolicKG(
        embedding_dim=emb.shape[1],
        similarity_threshold=threshold,
        device=device,
        seed=seed,
    )
    assigned = [kg.match_or_add(e)[0] for e in torch.from_numpy(emb).to(device)]
    remap = kg.consolidate(
        merge_threshold=consolidate_threshold,
        min_count=effective_min_count(len(emb), min_count, prune_frac),
    )

    keep = [i for i, r in enumerate(recs) if r["label"] in _REAL]
    before = score([assigned[i] for i in keep], [recs[i]["label"] for i in keep])
    survived = [i for i in keep if remap.get(assigned[i]) is not None]
    after = score(
        [remap[assigned[i]] for i in survived], [recs[i]["label"] for i in survived]
    )
    after["dropped"] = len(keep) - len(survived)
    return before, after


def summarize(rows, key):
    values = np.array([r[key] for r in rows], dtype=float)
    return values.mean(), values.std()


def table(title, per_threshold, thresholds):
    lines = [
        "",
        title,
        "-" * 92,
        f"{'threshold':>9s} {'accuracy':>16s} {'F1':>16s} {'recall':>16s} "
        f"{'precision':>16s} {'nodes':>12s}",
    ]
    for t in thresholds:
        rows = per_threshold[t]
        cells = []
        for key in ("accuracy", "f1", "recall", "precision", "nodes"):
            mean, std = summarize(rows, key)
            fmt = "%6.2f +-%5.2f" if key == "nodes" else "%6.4f +-%6.4f"
            cells.append((fmt % (mean, std)).rjust(16 if key != "nodes" else 12))
        lines.append(f"{t:>9.2f} " + " ".join(cells))
    lines.append("-" * 92)
    return "\n".join(lines)


def main(inputArguments):
    initialize(inputArguments)
    run_dirs = sorted(glob.glob(g_ArgParse.get("runs")))
    if not run_dirs:
        raise FileNotFoundError(f"no run directories match {g_ArgParse.get('runs')}")
    thresholds = [float(v) for v in g_ArgParse.get("thresholds").split(",")]
    consolidate_threshold = float(g_ArgParse.get("consolidate_threshold"))
    min_count = int(g_ArgParse.get("min_count"))
    prune_frac = float(g_ArgParse.get("prune_frac"))
    device = g_ArgParse.get("device")
    seed = int(g_ArgParse.get("seed"))
    out = g_ArgParse.get("out")

    print(f"{len(run_dirs)} runs x {len(thresholds)} thresholds")
    for d in run_dirs:
        print(f"  {d}")

    loaded = [(d, *load(d)) for d in run_dirs]
    for d, recs, emb in loaded:
        print(f"  {os.path.basename(d)}: {len(recs)} detections")

    before_by_t, after_by_t, rows = defaultdict(list), defaultdict(list), []
    for threshold in thresholds:
        for d, recs, emb in loaded:
            before, after = evaluate_run(
                recs, emb, threshold, consolidate_threshold, min_count, prune_frac,
                device, seed
            )
            before_by_t[threshold].append(before)
            after_by_t[threshold].append(after)
            rows.append(
                {"run": os.path.basename(d), "threshold": threshold,
                 "before": before, "after": after}
            )
        b = after_by_t[threshold][-1]
        print(f"  threshold {threshold:.2f} done "
              f"(last run: {b['nodes']} nodes, F1 {b['f1']:.4f})")

    text = "\n".join(
        [
            f"KG QUALITY vs MERGE THRESHOLD, mean +- std over {len(run_dirs)} seeds",
            f"consolidate_threshold={consolidate_threshold}  min_count={min_count}  "
            f"prune_frac={prune_frac}",
            "BCubed precision/recall/F1; nodes and metrics cover real objects only "
            "(lion/cage/food).",
            table("BEFORE CONSOLIDATION", before_by_t, thresholds),
            table("AFTER CONSOLIDATION", after_by_t, thresholds),
        ]
    )
    print("\n" + text)
    if out:
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote per-run rows to {out}")
    print("Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
