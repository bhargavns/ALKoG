import json
import os
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared plot palette.
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


def append_metrics(path, row):
    """Append one dict as a JSON line (the run's machine-readable metrics log)."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


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
        else:
            for prefix in ("tx", "rx"):
                prefixed = f"{prefix}_{key}"
                if prefixed in rows[0]:
                    ax.plot(it, [r[prefixed] for r in rows], label=prefix)
            if ax.lines:
                ax.legend()
        ax.set_title(title)
        ax.set_xlabel("iteration")
    path = os.path.join(plot_dir, "training_losses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)
    if "tx_actor_grad_norm" in rows[0]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for key in ("tx_actor_grad_norm", "tx_value_grad_norm"):
            axes[0].plot(it, [r[key] for r in rows], label=key)
        axes[0].legend(fontsize=7)
        axes[0].set_title("Message-head gradient norms")
        axes[1].plot(it, [r["channel_saturation"] for r in rows])
        axes[1].set_title("Soft components below .05 or above .95")
        axes[2].plot(it, [r["channel_logit_abs"] for r in rows])
        axes[2].set_title("Mean absolute message logit")
        for ax in axes:
            ax.set_xlabel("iteration")
        path = os.path.join(plot_dir, "communication_channel.png")
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
