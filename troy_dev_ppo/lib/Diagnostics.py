import json
import os
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Validated light-mode dataviz palette (see dataviz skill reference instance).
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
STATUS_SERIOUS = "#ec835a"

# blue sequential ramp 100->700 (magnitude), blue<->red diverging w/ gray midpoint
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "alkog_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "alkog_div", ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f0a5a5", "#e34948", "#8f1d1d"]
)


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "sans-serif",
            "font.size": 9,
            "lines.linewidth": 1.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ------------------------------------------------------------------ run dirs


def make_run_dir(base_dir, prefix, run_name=""):
    """Create a unique timestamped run directory; never reuses an existing one."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{stamp}" + (f"_{run_name}" if run_name else "")
    path = os.path.join(base_dir, name)
    suffix = 0
    while os.path.exists(path):
        suffix += 1
        path = os.path.join(base_dir, f"{name}_{suffix}")
    os.makedirs(path)
    return path


def unique_path(path):
    """Return `path` if free, else path with _1/_2/... inserted before the extension."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{root}_{i}{ext}"):
        i += 1
    return f"{root}_{i}{ext}"


def latest_kg_path(output_dir):
    """Newest kg.pt among output/runs/kg_*/, falling back to legacy output/kg.pt."""
    candidates = []
    runs = os.path.join(output_dir, "runs")
    if os.path.isdir(runs):
        for d in os.listdir(runs):
            p = os.path.join(runs, d, "kg.pt")
            if d.startswith("kg_") and os.path.exists(p):
                candidates.append(p)
    legacy = os.path.join(output_dir, "kg.pt")
    if os.path.exists(legacy):
        candidates.append(legacy)
    if not candidates:
        raise FileNotFoundError(f"no kg.pt found under {runs} or {output_dir}")
    return max(candidates, key=os.path.getmtime)


def append_metrics(path, row):
    """Append one dict as a JSON line (the run's machine-readable metrics log)."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------ KG report


def kg_report(kg, exemplar_dir=None):
    """Verbose human-readable dump of everything the KG stores."""
    lines = [
        "=" * 78,
        "SYMBOLIC KG DIAGNOSTIC REPORT",
        "=" * 78,
        f"nodes: {kg.num_nodes}   distinct edges: {len(kg.edges)}   "
        f"relation vocabulary: {kg.relation_names}",
        f"embedding dim: {kg.embeddings.shape[1] if kg.num_nodes else '?'}   "
        f"symbol dim: {kg.symbols.shape[1] if kg.num_nodes else '?'}   "
        f"match threshold: {kg.similarity_threshold}",
        "",
        "-- NODES (concept = EMA of merged visual detections; symbol = trainable 4-dim code)",
    ]
    total = max(1, sum(kg.counts))
    for i in range(kg.num_nodes):
        emb = kg.embeddings[i]
        sym = kg.symbols[i]
        lines.append(
            f"  node {i}: {kg.counts[i]} detections merged ({100 * kg.counts[i] / total:.1f}% "
            f"of all), |embedding|={float(emb.norm()):.3f}"
        )
        lines.append(
            "           symbol = ["
            + ", ".join(f"{float(v):+.4f}" for v in sym)
            + f"]  |symbol|={float(sym.norm()):.3f}"
        )
        if exemplar_dir is not None:
            ex = os.path.join(exemplar_dir, f"node{i}.png")
            lines.append(
                f"           exemplar crop: {ex if os.path.exists(ex) else '(none saved)'}"
            )
    if kg.num_nodes > 1:
        sims = (kg.embeddings @ kg.embeddings.T).cpu()
        lines.append("")
        lines.append("-- PAIRWISE VISUAL COSINE SIMILARITY (merge threshold "
                     f"{kg.similarity_threshold}; close pairs risk aliasing)")
        header = "        " + " ".join(f"  n{j:<4d}" for j in range(kg.num_nodes))
        lines.append(header)
        for i in range(kg.num_nodes):
            row = " ".join(f"{float(sims[i, j]):+.3f}" for j in range(kg.num_nodes))
            lines.append(f"  n{i:<4d} {row}")
        tri = [
            (float(sims[i, j]), i, j)
            for i in range(kg.num_nodes)
            for j in range(i + 1, kg.num_nodes)
        ]
        for sim, i, j in sorted(tri, reverse=True)[:3]:
            flag = "  <-- above merge threshold!" if sim >= kg.similarity_threshold else ""
            lines.append(f"  closest pair: node {i} vs node {j}: {sim:+.3f}{flag}")
    lines.append("")
    lines.append("-- RELATION SYMBOLS")
    for i, name in enumerate(kg.relation_names):
        sym = kg.relation_symbols[i]
        lines.append(
            f"  '{name}': [" + ", ".join(f"{float(v):+.4f}" for v in sym) + "]"
        )
    lines.append("")
    lines.append(f"-- EDGES ({len(kg.edges)} distinct, sorted by observation count)")
    total_obs = max(1, sum(kg.edges.values()))
    for (src, rel, dst), count in sorted(kg.edges.items(), key=lambda kv: -kv[1]):
        lines.append(
            f"  node {src} --{kg.relation_names[rel]}--> node {dst}: seen {count}x "
            f"({100 * count / total_obs:.1f}% of observations)"
        )
    lines.append("")
    lines.append("-- SYMBOL GEOMETRY (cosine between node symbols; PPO shapes these in phase 2)")
    if kg.num_nodes > 1:
        import torch.nn.functional as F

        norm = F.normalize(kg.symbols, dim=1)
        ssims = norm @ norm.T
        for i in range(kg.num_nodes):
            row = " ".join(f"{float(ssims[i, j]):+.3f}" for j in range(kg.num_nodes))
            lines.append(f"  n{i:<4d} {row}")
    lines.append("=" * 78)
    return "\n".join(lines)


def save_kg_plots(kg, plot_dir):
    """Visual diagnostics of KG contents. Returns list of files written."""
    apply_style()
    os.makedirs(plot_dir, exist_ok=True)
    written = []
    n = kg.num_nodes
    if n == 0:
        return written

    # 1. visual similarity heatmap (diverging: 0 = unrelated gray midpoint)
    sims = (kg.embeddings @ kg.embeddings.T).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(sims, cmap=DIV_CMAP, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sims[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    color=INK if abs(sims[i, j]) < 0.6 else SURFACE)
    ax.set_xticks(range(n), [f"n{i}" for i in range(n)])
    ax.set_yticks(range(n), [f"n{i}" for i in range(n)])
    ax.set_title("Concept visual similarity (cosine of stored embeddings)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)
    path = os.path.join(plot_dir, "kg_similarity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # 2. symbol tables (node + relation) as diverging heatmaps
    node_sym = kg.symbols.detach().cpu().numpy()
    rel_sym = kg.relation_symbols.detach().cpu().numpy()
    vmax = max(1e-6, np.abs(node_sym).max(), np.abs(rel_sym).max())
    fig, axes = plt.subplots(
        1, 2, figsize=(8.4, 0.55 * max(n, len(kg.relation_names)) + 1.6),
        width_ratios=[max(n, 1), max(len(kg.relation_names), 1)],
    )
    for ax, mat, labels, title in (
        (axes[0], node_sym, [f"n{i}" for i in range(n)], "Node symbols"),
        (axes[1], rel_sym, kg.relation_names, "Relation symbols"),
    ):
        im = ax.imshow(mat, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                        color=INK if abs(mat[i, j]) < 0.6 * vmax else SURFACE)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(mat.shape[1]), [f"d{j}" for j in range(mat.shape[1])])
        ax.set_title(title)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8)
    path = os.path.join(plot_dir, "kg_symbols.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # 3. graph structure diagram (node size ~ detections, edge width ~ log count)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    try:
        import networkx as nx

        g = nx.MultiDiGraph()
        for i in range(n):
            g.add_node(i)
        for (src, rel, dst), count in kg.edges.items():
            g.add_edge(src, dst, rel=rel, count=count)
        pos = nx.spring_layout(g, seed=0, k=1.6)
    except ImportError:
        angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {i: (np.cos(a), np.sin(a)) for i, a in zip(range(n), angle)}
    max_count = max(kg.edges.values()) if kg.edges else 1
    rel_colors = {i: SERIES[i % len(SERIES)] for i in range(len(kg.relation_names))}
    for (src, rel, dst), count in kg.edges.items():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>", color=rel_colors[rel], alpha=0.75,
                lw=0.8 + 2.4 * np.log1p(count) / np.log1p(max_count),
                connectionstyle=f"arc3,rad={0.18 if rel % 2 == 0 else -0.18}",
                shrinkA=14, shrinkB=14,
            ),
        )
    counts = np.array(kg.counts, dtype=float)
    sizes = 300 + 900 * counts / counts.max()
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=sizes, c=SURFACE, edgecolors=INK_2, linewidths=1.4, zorder=3)
    for i in range(n):
        ax.annotate(f"n{i}\n({kg.counts[i]})", pos[i], ha="center", va="center",
                    fontsize=8, zorder=4)
    for i, name in enumerate(kg.relation_names):
        ax.plot([], [], color=rel_colors[i], label=name)
    ax.legend(title="relation", loc="upper left", fontsize=8)
    ax.set_title("KG structure (node size = detections merged, edge width = times seen)")
    ax.set_axis_off()
    path = os.path.join(plot_dir, "kg_graph.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # 4. embedding PCA scatter + detection count bars
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    emb = kg.embeddings.cpu().numpy()
    centered = emb - emb.mean(axis=0, keepdims=True)
    if n > 1:
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        xy = u[:, :2] * s[:2]
        var = s**2 / max(1e-9, (s**2).sum())
    else:
        xy = np.zeros((n, 2))
        var = [0.0, 0.0]
    for i in range(n):
        axes[0].scatter(xy[i, 0], xy[i, 1], s=90, color=SERIES[i % len(SERIES)], zorder=3)
        axes[0].annotate(f"n{i}", (xy[i, 0], xy[i, 1]), textcoords="offset points",
                         xytext=(7, 4), fontsize=8)
    axes[0].set_title(f"Concept embeddings, PCA ({100 * var[0]:.0f}% + {100 * var[1]:.0f}% var)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].bar(range(n), kg.counts, color=SERIES[0], width=0.62)
    axes[1].set_xticks(range(n), [f"n{i}" for i in range(n)])
    axes[1].set_title("Detections merged per concept node")
    axes[1].set_ylabel("count")
    path = os.path.join(plot_dir, "kg_embeddings.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


# ------------------------------------------------------------ phase-1 curves


def save_phase1_curves(rows, plot_dir):
    """Per-episode KG growth curves from build_kg metrics rows."""
    apply_style()
    os.makedirs(plot_dir, exist_ok=True)
    eps = [r["episode"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(eps, [r["nodes"] for r in rows], color=SERIES[0], marker="o", ms=4)
    axes[0].set_title("Concept nodes after episode")
    axes[1].plot(eps, [r["distinct_edges"] for r in rows], color=SERIES[1], marker="o", ms=4)
    axes[1].set_title("Distinct edges after episode")
    axes[2].plot(eps, [r["detections"] for r in rows], color=SERIES[5], marker="o", ms=4)
    axes[2].set_title("Concept detections in episode")
    for ax in axes:
        ax.set_xlabel("episode")
    path = os.path.join(plot_dir, "kg_growth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------ phase-2 curves


def save_training_curves(rows, plot_dir):
    """PPO training curves from per-iteration stats rows."""
    apply_style()
    os.makedirs(plot_dir, exist_ok=True)
    it = [r["iteration"] for r in rows]
    written = []

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    axes[0, 0].plot(it, [r["mean_return"] for r in rows], color=SERIES[0])
    axes[0, 0].set_title("Mean episode return")
    axes[0, 1].plot(it, [r["mean_length"] for r in rows], color=SERIES[6])
    axes[0, 1].set_title("Mean episode length")
    axes[1, 0].plot(it, [r["food_rate"] for r in rows], color=STATUS_GOOD, label="food")
    axes[1, 0].plot(it, [r["death_rate"] for r in rows], color=STATUS_CRITICAL, label="death")
    axes[1, 0].plot(it, [r["timeout_rate"] for r in rows], color=MUTED, label="timeout")
    axes[1, 0].set_title("Episode outcome rates")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(it, [r["caged_food_rate"] for r in rows], color=SERIES[0],
                    label="food | caged")
    axes[1, 1].plot(it, [r["loose_food_rate"] for r in rows], color=SERIES[3],
                    label="food | loose")
    axes[1, 1].plot(it, [r["loose_death_rate"] for r in rows], color=STATUS_CRITICAL,
                    label="death | loose")
    axes[1, 1].set_title("Caged vs loose lion (the KG-symbol question)")
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("iteration")
    path = os.path.join(plot_dir, "training_progress.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    panels = [
        ("pi_loss", "Policy loss", SERIES[0]),
        ("v_loss", "Value loss", SERIES[6]),
        ("entropy", "Policy entropy", SERIES[4]),
        ("approx_kl", "Approx KL per update", SERIES[3]),
        ("clip_frac", "PPO clip fraction", SERIES[5]),
        ("explained_var", "Value explained variance", SERIES[1]),
    ]
    for ax, (key, title, color) in zip(axes.flat, panels):
        if key in rows[0]:
            ax.plot(it, [r[key] for r in rows], color=color)
        ax.set_title(title)
        ax.set_xlabel("iteration")
    path = os.path.join(plot_dir, "training_losses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def save_action_plots(rows, last_actions, plot_dir,
                      act_labels=("forward", "backward", "left", "right")):
    """Discrete-action diagnostics: selection fractions over training + final counts."""
    apply_style()
    os.makedirs(plot_dir, exist_ok=True)
    it = [r["iteration"] for r in rows]
    n_act = len(act_labels)
    written = []

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for d in range(n_act):
        axes[0].plot(it, [r["action_fracs"][d] for r in rows],
                     color=SERIES[d % len(SERIES)], label=act_labels[d])
    axes[0].axhline(1.0 / n_act, color=BASELINE, lw=0.8)
    axes[0].set_title("Action selection fraction per iteration")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8)

    if last_actions is not None:
        counts = np.bincount(np.asarray(last_actions, dtype=int), minlength=n_act)
        axes[1].bar(range(n_act), counts,
                    color=[SERIES[d % len(SERIES)] for d in range(n_act)], width=0.62)
        axes[1].set_xticks(range(n_act), act_labels)
        axes[1].set_title("Actions sampled (final iteration)")
        axes[1].set_ylabel("count")
    path = os.path.join(plot_dir, "action_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written


def save_trajectory_plot(trajectories, plot_dir, filename="trajectories.png", arena_half=7.0):
    """Grid of top-down episode paths with object layout and outcome."""
    apply_style()
    os.makedirs(plot_dir, exist_ok=True)
    if not trajectories:
        return None
    trajectories = trajectories[-12:]
    cols = min(4, len(trajectories))
    trows = int(np.ceil(len(trajectories) / cols))
    fig, axes = plt.subplots(trows, cols, figsize=(3.1 * cols, 3.1 * trows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, traj in zip(axes.flat, trajectories):
        ax.set_axis_on()
        pos = np.asarray(traj["positions"])
        outcome = traj["outcome"]
        color = {"food": STATUS_GOOD, "death": STATUS_CRITICAL}.get(outcome, MUTED)
        # path shading: light early -> dark late so direction is readable
        for k in range(1, len(pos)):
            ax.plot(pos[k - 1 : k + 1, 0], pos[k - 1 : k + 1, 1], color=SERIES[0],
                    alpha=0.15 + 0.85 * k / len(pos), lw=1.4, solid_capstyle="round")
        ax.scatter(*pos[0], marker="o", s=45, color=SERIES[0], zorder=4)
        ax.scatter(*traj["food"], marker="*", s=130, color=STATUS_GOOD, zorder=4)
        ax.scatter(*traj["cage"], marker="s", s=80, facecolors="none",
                   edgecolors=SERIES[6], linewidths=1.6, zorder=4)
        ax.scatter(*traj["lion"], marker="X", s=90,
                   color=STATUS_SERIOUS if traj["lion_caged"] else STATUS_CRITICAL, zorder=4)
        lim = arena_half + 0.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(
            f"it{traj.get('iteration', '?')} {'caged' if traj['lion_caged'] else 'LOOSE'} "
            f"-> {outcome} (R={traj['episode_return']:+.1f})",
            fontsize=8, color=color,
        )
    fig.suptitle("Agent trajectories  (o start, * food, X lion, square cage; path darkens "
                 "over time)", fontsize=10)
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_symbol_evolution(node_snaps, rel_snaps, relation_names, counts, plot_dir):
    """Symbol drift over training. node_snaps: [T, N, D]; rel_snaps: [T, R, D]."""
    apply_style()
    os.makedirs(plot_dir, exist_ok=True)
    node_snaps = np.asarray(node_snaps)
    rel_snaps = np.asarray(rel_snaps)
    t = np.arange(node_snaps.shape[0])
    n, d = node_snaps.shape[1], node_snaps.shape[2]
    written = []

    cols = min(4, max(1, n))
    trows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(trows, cols, figsize=(3.2 * cols, 2.7 * trows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for i in range(n):
        ax = axes.flat[i]
        ax.set_axis_on()
        for j in range(d):
            ax.plot(t, node_snaps[:, i, j], color=SERIES[j % len(SERIES)],
                    label=f"d{j}" if i == 0 else None)
        ax.set_title(f"node {i} ({counts[i]} det.)", fontsize=8)
        ax.set_xlabel("iteration")
    if n:
        axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Node symbol components during PPO", fontsize=10)
    path = os.path.join(plot_dir, "symbol_evolution_nodes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, max(2, len(relation_names)),
                             figsize=(3.2 * max(2, len(relation_names)), 2.9), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for i, name in enumerate(relation_names):
        ax = axes.flat[i]
        ax.set_axis_on()
        for j in range(rel_snaps.shape[2]):
            ax.plot(t, rel_snaps[:, i, j], color=SERIES[j % len(SERIES)], label=f"d{j}")
        ax.set_title(f"relation '{name}'", fontsize=9)
        ax.set_xlabel("iteration")
        if i == 0:
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Relation symbol components during PPO", fontsize=10)
    path = os.path.join(plot_dir, "symbol_evolution_relations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # drift magnitude summary (L2 from start)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i in range(n):
        drift = np.linalg.norm(node_snaps - node_snaps[0:1], axis=2)[:, i]
        ax.plot(t, drift, color=SERIES[i % len(SERIES)], label=f"n{i}")
    for i, name in enumerate(relation_names):
        drift = np.linalg.norm(rel_snaps - rel_snaps[0:1], axis=2)[:, i]
        ax.plot(t, drift, color=INK_2, ls="--" if i % 2 else ":", label=name)
    ax.set_title("Symbol L2 drift from initialization")
    ax.set_xlabel("iteration")
    ax.set_ylabel("‖s_t − s_0‖")
    ax.legend(fontsize=7, ncol=4)
    path = os.path.join(plot_dir, "symbol_drift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    return written
