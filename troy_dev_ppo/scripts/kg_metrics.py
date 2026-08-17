#!/bin/python3
import sys, os
sys.path.append(os.environ['CORE'])
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
from collections import Counter, defaultdict

import numpy as np
import torch

from lib.python.Logger import Logger
from lib.python.ArgumentParser import ArgumentParser
from lib.SymbolicKG import SymbolicKG

g_Logger = Logger(__name__)
g_ArgParse = ArgumentParser()
print = g_Logger.print

_REAL = ("lion", "cage", "food")


def effective_min_count(total_detections, min_count, prune_frac):
    """build_kg scales the prune floor with exposure; mirror it exactly."""
    return max(min_count, round(prune_frac * total_detections))

required_arguments = ["run_dir"]
optional_arguments = {
    "threshold": "0.50",        # must match the build_kg run being scored
    "consolidate_threshold": "0.85",
    "min_count": "5",
    "prune_frac": "0.01",       # matches build_kg: prune below this share of detections
    "device": "cuda",
    "seed": "0",
}

USAGE = """
kg_metrics.py -- accuracy / precision / recall for a built KG vs ground truth.

Grouping detections into concept nodes is a clustering problem, so the useful
scores are the clustering ones:

  pairwise  over every pair of detections, did the KG put them in the same node
            and are they in fact the same object?
              precision = same-node pairs that are truly the same object
              recall    = same-object pairs the KG actually put together
            A false merge costs precision; a false split costs recall.

  BCubed    the same question asked per detection rather than per pair, so
            large objects do not dominate (Bagga & Baldwin, 1998).

  accuracy  the simple per-detection rate: was this detection assigned to a
            node that stands for its true object?

Reported before consolidation (what the online run built) and after.

    Required:
        run_dir=...   a build_kg run directory written with oracle=1

    Optional:
        threshold=0.50 consolidate_threshold=0.85 min_count=5 prune_frac=0.01
            (threshold must match the value the run was built with, or the
            replay below will not reproduce the run's node assignments)
        device=cuda seed=0
"""


def initialize(inputArguments):
    print(f"ScriptName: {__file__}")
    try:
        g_ArgParse.setArguments(inputArguments, required_arguments, optional_arguments)
    except Exception as e:
        e.add_note(USAGE)
        raise
    g_ArgParse.printArguments()


def pairwise_scores(nodes, labels):
    """Precision/recall/F1 over all detection pairs, computed by contingency."""
    table = defaultdict(int)
    for n, l in zip(nodes, labels):
        table[(n, l)] += 1
    node_totals, label_totals = Counter(), Counter()
    for (n, l), c in table.items():
        node_totals[n] += c
        label_totals[l] += c

    def pairs(c):
        return c * (c - 1) // 2

    tp = sum(pairs(c) for c in table.values())
    same_node = sum(pairs(c) for c in node_totals.values())
    same_label = sum(pairs(c) for c in label_totals.values())
    precision = tp / same_node if same_node else 0.0
    recall = tp / same_label if same_label else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def bcubed_scores(nodes, labels):
    """Per-detection precision/recall (Bagga & Baldwin)."""
    table = defaultdict(int)
    node_totals, label_totals = Counter(), Counter()
    for n, l in zip(nodes, labels):
        table[(n, l)] += 1
        node_totals[n] += 1
        label_totals[l] += 1
    p = np.mean([table[(n, l)] / node_totals[n] for n, l in zip(nodes, labels)])
    r = np.mean([table[(n, l)] / label_totals[l] for n, l in zip(nodes, labels)])
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return float(p), float(r), float(f1)


def node_labels(nodes, labels):
    hist = defaultdict(Counter)
    for n, l in zip(nodes, labels):
        hist[n][l] += 1
    return {n: h.most_common(1)[0][0] for n, h in hist.items()}


def accuracy(nodes, labels):
    majority = node_labels(nodes, labels)
    return float(np.mean([majority[n] == l for n, l in zip(nodes, labels)]))


def report(tag, nodes, labels):
    pp, pr, pf = pairwise_scores(nodes, labels)
    bp, br, bf = bcubed_scores(nodes, labels)
    majority = node_labels(nodes, labels)
    per_object = Counter(majority.values())
    print(
        f"  {tag:<22s} acc {accuracy(nodes, labels):.4f}   "
        f"pairwise P {pp:.4f} R {pr:.4f} F1 {pf:.4f}   "
        f"BCubed P {bp:.4f} R {br:.4f} F1 {bf:.4f}   "
        f"nodes {len(set(nodes))} ({dict(per_object)})"
    )
    return {"pairwise": (pp, pr, pf), "bcubed": (bp, br, bf)}


def main(inputArguments):
    initialize(inputArguments)
    run_dir = g_ArgParse.get("run_dir")
    oracle_dir = os.path.join(run_dir, "oracle")
    recs = [json.loads(l) for l in open(os.path.join(oracle_dir, "detections.jsonl"))]
    emb = np.load(os.path.join(oracle_dir, "embeddings.npy")).astype(np.float32)

    keep = [i for i, r in enumerate(recs) if r["label"] in _REAL and r["node"] is not None]
    labels = [recs[i]["label"] for i in keep]
    print(f"{len(recs)} detections logged; {len(keep)} of real objects "
          f"({dict(Counter(labels))})")

    print("\nAS BUILT (node assignments recorded during the run):")
    report("before consolidation", [recs[i]["node"] for i in keep], labels)

    # consolidation is not recorded per detection, so replay it -- the replay is
    # deterministic and reproduces the online node count exactly
    device = g_ArgParse.get("device")
    kg = SymbolicKG(
        embedding_dim=emb.shape[1],
        similarity_threshold=float(g_ArgParse.get("threshold")),
        device=device,
        seed=int(g_ArgParse.get("seed")),
    )
    assigned = []
    for e in torch.from_numpy(emb).to(device):
        assigned.append(kg.match_or_add(e)[0])
    min_count_eff = effective_min_count(
        len(emb), int(g_ArgParse.get("min_count")), float(g_ArgParse.get("prune_frac"))
    )
    remap = kg.consolidate(
        merge_threshold=float(g_ArgParse.get("consolidate_threshold")),
        min_count=min_count_eff,
    )
    print(f"\nREPLAY (reproduces the run: {len(set(assigned))} nodes before "
          f"consolidation, {kg.num_nodes} after):")
    report("before consolidation", [assigned[i] for i in keep], labels)
    survived = [i for i in keep if remap.get(assigned[i]) is not None]
    dropped = len(keep) - len(survived)
    report(
        "after consolidation",
        [remap[assigned[i]] for i in survived],
        [recs[i]["label"] for i in survived],
    )
    print(f"  ({dropped} detections belonged to nodes pruned by min_count and are "
          "excluded from the post-consolidation row)")
    print("Success! Exiting...")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        g_Logger.logger.exception(e)
        exit(1)
