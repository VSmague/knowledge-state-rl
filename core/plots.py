"""
core/plots.py — Generic plots that need no game-specific visual style:
eval success curve, curriculum training loss, and the ks plots (PCA
trajectory, step distance). Built entirely from stats.pt (+ ks_*.pt),
never touches an env instance.

Per-group breakdowns (ALP/entropy/success/competence) are deliberately
NOT here: they'd need genuinely different visual treatment per game (a
handful of spatially-organized rooms for minigrid_game vs. 22 unordered
achievements for crafter_game), so a single shared "one style fits both"
version isn't the right fit — a game module that wants one can add its
own plots.py instead.

Usage:
    python -m core.plots --run_dir exp/<your_run>/
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import median_filter, uniform_filter1d


# ---------------------------------------------------------------------------
# Smoothing -- the dense per-rollout loss histories (ppo_loss_history/
# loss_history are recorded every rollout instead of every log_every
# episodes) have enough points that individual-rollout noise/spikes
# become visible, so plot_loss/plot_ppo_loss smooth them before drawing.
# ---------------------------------------------------------------------------

def smooth_curve(y, window: int = 15):
    """Two passes: a rolling MEDIAN (robust to isolated spikes, unlike a
    rolling mean which just dampens a spike into a smaller bump), then a
    light rolling MEAN on top for a visually smooth line. Both passes use
    edge-replicated padding so the output stays the same length as the
    input. window <= 1 or fewer than 3 points -> returned unchanged (no
    smoothing possible/meaningful)."""
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or len(y) < 3:
        return y
    w = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if w <= 1:
        return y
    med = median_filter(y, size=w, mode="nearest")
    mean_w = max(1, w // 3)
    return uniform_filter1d(med, size=mean_w, mode="nearest") if mean_w > 1 else med


# ---------------------------------------------------------------------------
# Eval success curve
# ---------------------------------------------------------------------------

def plot_success_rate(eval_history: list, out_path: str) -> None:
    if not eval_history:
        print("  [skip] no eval_history")
        return
    steps = [e["step"] for e in eval_history]
    means = np.array([float(np.mean(list(e["results"].values()))) for e in eval_history])
    stds  = np.array([float(np.std(list(e["results"].values())))  for e in eval_history])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    ax.plot(steps, means, color="#2ca25f", linewidth=2, marker="o", markersize=4, zorder=3)
    ax.fill_between(steps, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1),
                    color="#2ca25f", alpha=0.2, zorder=2, label="± std across goals")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Step"); ax.set_ylabel("Mean success rate (all goals)")
    ax.set_title("Evaluation success rate over training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Real eval success rate per group (room for minigrid_game, goal for crafter_game)
# ---------------------------------------------------------------------------

def plot_eval_success_by_group(eval_history: list, out_path: str) -> None:
    """Real eval success rate per group over training — mean ± std across
    the goals that fall into each group (std collapses to 0 for
    crafter_game's identity grouping, one goal per group). Small
    multiples, one subplot per group; same mean/std treatment as
    plot_success_rate, just broken out per group instead of pooled across
    all goals. Game-agnostic: results_by_group is already computed
    generically in core/train.py via group_fn (room_id for minigrid_game,
    identity for crafter_game)."""
    if not eval_history:
        print("  [skip] no eval_history")
        return
    steps = [e["step"] for e in eval_history]
    all_groups = sorted(
        {g for e in eval_history for g in e["results_by_group"].keys()}, key=str)
    if not all_groups:
        print("  [skip] no group data in eval_history")
        return

    ncols = min(4, len(all_groups)) or 1
    nrows = int(np.ceil(len(all_groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.2 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for i, group in enumerate(all_groups):
        ax = axes[i]
        means = np.array([e["results_by_group"].get(group, {}).get("mean", np.nan)
                          for e in eval_history])
        stds  = np.array([e["results_by_group"].get(group, {}).get("std", 0.0)
                          for e in eval_history])
        ax.plot(steps, means, color="#2ca25f", linewidth=1.6, marker="o", markersize=3, zorder=3)
        ax.fill_between(steps, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1),
                        color="#2ca25f", alpha=0.2, zorder=2)
        ax.set_title(str(group), fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(len(all_groups), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Evaluation success rate per group (mean ± std across goals in group)",
                 fontsize=11)
    fig.supxlabel("Step", fontsize=9)
    fig.supylabel("Success rate", fontsize=9)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Training loss (magellan / magellan_ks only)
# ---------------------------------------------------------------------------

def _plot_raw_and_smoothed(ax, steps, values, color, label, smooth_window):
    """Faint raw line + bold smoothed line on top (see smooth_curve above)
    -- shows the true data isn't hidden, while keeping the trend readable
    once loss histories have hundreds of per-rollout points."""
    steps = np.asarray(steps)
    values = np.asarray(values, dtype=np.float64)
    ax.plot(steps, values, color=color, linewidth=0.6, alpha=0.25, zorder=1)
    smoothed = smooth_curve(values, window=smooth_window)
    ax.plot(steps, smoothed, color=color, linewidth=1.6, label=label, zorder=3)


def plot_loss(loss_history: list, out_path: str, smooth_window: int = 15) -> None:
    """loss_history entries are (step, loss) for old runs (train_rnn.py,
    or core/train.py before the aux/cl breakdown was added) or (step,
    loss, aux_loss_or_None, cl_loss_or_None) for current core/train.py
    runs -- handles both so old exp/ dirs don't break.

    smooth_window: rolling-median-then-mean window (see smooth_curve
    above) applied on top of the raw curve, which is still drawn faintly
    underneath -- recorded every rollout, so loss histories have enough
    points that per-rollout noise/spikes are worth smoothing over for
    readability. Pass 0/1 to disable."""
    if not loss_history:
        print("  [skip] no loss_history (uniform/online have no competence MLP)")
        return
    steps  = [h[0] for h in loss_history]
    losses = [h[1] for h in loss_history]
    has_breakdown = len(loss_history[0]) >= 4
    aux = [h[2] for h in loss_history] if has_breakdown else []
    cl  = [h[3] for h in loss_history] if has_breakdown else []
    has_aux = has_breakdown and any(v is not None for v in aux)
    has_cl  = has_breakdown and any(v is not None for v in cl)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    _plot_raw_and_smoothed(ax, steps, losses, "#e34a33", "main (BCE)", smooth_window)
    if has_aux:
        aux_steps = [s for s, v in zip(steps, aux) if v is not None]
        aux_vals  = [v for v in aux if v is not None]
        _plot_raw_and_smoothed(ax, aux_steps, aux_vals, "#2980b9",
                               "aux (ks-recovery)", smooth_window)
    if has_cl:
        cl_steps = [s for s, v in zip(steps, cl) if v is not None]
        cl_vals  = [v for v in cl if v is not None]
        _plot_raw_and_smoothed(ax, cl_steps, cl_vals, "#27ae60",
                               "contrastive (NT-Xent)", smooth_window)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Curriculum competence-estimator training loss")
    ax.grid(True, alpha=0.3)
    if has_aux or has_cl:
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# PPO training loss (pg_loss / v_loss / entropy) -- every curriculum
# ---------------------------------------------------------------------------

def plot_ppo_loss(ppo_loss_history: list, out_path: str, smooth_window: int = 15) -> None:
    """Illustrates the 'entropy dominance' mechanism found on Crafter: for
    a goal whose advantage stays near zero (rarely/never solved), pg_loss
    and v_loss contribute almost nothing to the total PPO loss, so the
    -entropy_coef * entropy term can dominate the gradient and pin the
    policy near-uniform. Three panels (pg_loss, v_loss, entropy) sharing
    the step axis, so a reader can see all three move (or not) together.

    smooth_window: see plot_loss's docstring -- same rolling
    median-then-mean, raw curve still drawn faintly underneath. Pass
    0/1 to disable."""
    if not ppo_loss_history:
        print("  [skip] no ppo_loss_history")
        return
    steps    = [h[0] for h in ppo_loss_history]
    pg_loss  = [h[1] for h in ppo_loss_history]
    v_loss   = [h[2] for h in ppo_loss_history]
    entropy  = [h[3] for h in ppo_loss_history]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor("white")
    panels = [("pg_loss", pg_loss, "#8e44ad"), ("v_loss", v_loss, "#d35400"),
             ("entropy", entropy, "#16a085")]
    for ax, (name, ys, color) in zip(axes, panels):
        _plot_raw_and_smoothed(ax, steps, ys, color, name, smooth_window)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.suptitle("PPO training loss components", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# ks plots (magellan_ks only)
# ---------------------------------------------------------------------------

def _pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def plot_ks_trajectory_pca(ks_history: list, out_path: str) -> None:
    if not ks_history or len(ks_history) < 2:
        print("  [skip] not enough ks_history points for a PCA trajectory")
        return
    pcts  = [h[0] for h in ks_history]
    steps = [h[1] for h in ks_history]
    X = np.stack([h[2] for h in ks_history])
    if X.shape[1] < 2:
        print(f"  [skip] knowledge_dim={X.shape[1]} < 2, can't PCA to 2D")
        return
    proj = _pca_2d(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=steps, cmap="viridis", s=60, zorder=3)
    ax.plot(proj[:, 0], proj[:, 1], color="gray", linewidth=0.8, alpha=0.5, zorder=2)
    for i, pct in enumerate(pcts):
        ax.annotate(f"{pct}%", (proj[i, 0], proj[i, 1]), fontsize=7,
                   xytext=(4, 4), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="Step")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Knowledge-state trajectory (2D PCA)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


def plot_competence_by_ks_index(history: list, goal_names: list, out_dir: str) -> None:
    """Heatmap of C(g | ks) for every goal x position in the live ks_buffer
    (col 0 = oldest trained snapshot still in the buffer, last col = the
    live ks_now, not yet appended to the buffer). One PNG per logged
    checkpoint (competence_by_ks_index_<pct>pct.png) — the buffer's
    content changes over training, so a single end-of-run snapshot would
    hide that evolution."""
    if not history:
        print("  [skip] no ks_buffer_competence_history")
        return
    for pct, step, comps in history:
        S, G = comps.shape
        if G != len(goal_names):
            print(f"  [skip] goal_names ({len(goal_names)}) doesn't match comps G={G}")
            continue

        order = np.argsort([str(g) for g in goal_names])
        labels = [str(goal_names[i]) for i in order]
        grid = comps[:, order].T                 # (G, S) -> goals as rows

        fig, ax = plt.subplots(figsize=(max(6, 0.6 * S + 2), max(4, 0.22 * G)))
        fig.patch.set_facecolor("white")
        im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(S))
        ax.set_xticklabels([str(i) for i in range(S - 1)] + ["now"], fontsize=7)
        ax.set_yticks(range(G))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Position in ks_buffer (0 = oldest, \"now\" = live ks)")
        ax.set_ylabel("Goal")
        ax.set_title(f"Competence C(g|ks) across the ks buffer [{pct}% steps]")
        fig.colorbar(im, ax=ax, label="Estimated competence", fraction=0.03)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"competence_by_ks_index_{pct}pct.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  -> {out_path}")


def plot_ks_step_distance(ks_history: list, out_path: str) -> None:
    if not ks_history or len(ks_history) < 2:
        print("  [skip] not enough ks_history points for step-distance plot")
        return
    pcts = [h[0] for h in ks_history[1:]]
    X = np.stack([h[2] for h in ks_history])
    dists = np.linalg.norm(np.diff(X, axis=0), axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.plot(pcts, dists, color="#d95f02", linewidth=1.6, marker="o", markersize=4)
    ax.set_xlabel("% of training"); ax.set_ylabel("||ks[t] - ks[t-1]||")
    ax.set_title("Knowledge-state step distance between checkpoints")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_core_plots(run_dir: str, out_dir: str = None, smooth_window: int = 15) -> None:
    stats_path = os.path.join(run_dir, "stats.pt")
    if not os.path.exists(stats_path):
        print(f"  [warn] no stats.pt found in {run_dir}")
        return
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    out_dir = out_dir or os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_success_rate(stats.get("eval_history", []),
                      os.path.join(out_dir, "eval_success_curve.png"))
    plot_eval_success_by_group(stats.get("eval_history", []),
                               os.path.join(out_dir, "eval_success_by_group.png"))
    plot_loss(stats.get("loss_history", []),
             os.path.join(out_dir, "training_loss.png"), smooth_window=smooth_window)
    plot_ppo_loss(stats.get("ppo_loss_history", []),
                 os.path.join(out_dir, "ppo_loss.png"), smooth_window=smooth_window)

    if stats.get("curriculum") == "magellan_ks":
        ks_path = os.path.join(run_dir, "ks_history.pt")
        if os.path.exists(ks_path):
            ks_data = torch.load(ks_path, map_location="cpu", weights_only=False)
            plot_ks_trajectory_pca(ks_data["history"],
                                   os.path.join(out_dir, "ks_trajectory_pca.png"))
            plot_ks_step_distance(ks_data["history"],
                                  os.path.join(out_dir, "ks_step_distance.png"))
        else:
            print("  [warn] curriculum is magellan_ks but no ks_history.pt found")

        kbc_path = os.path.join(run_dir, "ks_buffer_competence_history.pt")
        if os.path.exists(kbc_path):
            kbc_data = torch.load(kbc_path, map_location="cpu", weights_only=False)
            plot_competence_by_ks_index(kbc_data["history"], kbc_data["goal_names"], out_dir)
        else:
            print("  [warn] curriculum is magellan_ks but no ks_buffer_competence_history.pt found")

    print(f"  Core plots -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--smooth_window", type=int, default=15,
                   help="Rolling median-then-mean window applied to the "
                        "loss plots (0/1 to disable). See smooth_curve.")
    args = p.parse_args()
    generate_core_plots(args.run_dir, args.out_dir, smooth_window=args.smooth_window)


if __name__ == "__main__":
    main()