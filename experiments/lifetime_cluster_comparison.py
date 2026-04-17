#!/usr/bin/env python
"""Visualize cluster membership across agent lifetimes for two T-SNE 3D sweep configs.

Compares:
  - takeru_200M config 5: PCA=30, perp=50, mcs=400, ms=5 -> 4 clusters, 33.5% noise
  - yaofeng_200M config 1: PCA=30, perp=50, mcs=50, ms=10 -> 3 clusters, 9.8% noise

Produces per-policy:
  1. Swim-lane lifetime plot: each agent's cluster membership over ticks
  2. Cluster transition heatmap: from-cluster (Y) -> to-cluster (X) frequencies

Usage:
    uv run python experiments/lifetime_cluster_comparison.py
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

# Add scripts/ to path so we can import analyze_activations
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from analyze_activations import load_data, compute_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLICIES = {
    "takeru_200M": {
        "data_dir": str(REPO_ROOT / "activation_data" / "takeru_200M"),
        "label_path": str(REPO_ROOT / "experiments" / "T-SNE-3d-sweep-takeru_200M" / "config_5" / "cluster_labels.npy"),
        "config_str": "PCA=30, perp=50, mcs=400, ms=5",
        "n_clusters": 4,
        "noise_pct": 33.5,
    },
    "yaofeng_200M": {
        "data_dir": str(REPO_ROOT / "activation_data" / "yaofeng_200M"),
        "label_path": str(REPO_ROOT / "experiments" / "T-SNE-3d-sweep-yaofeng_200M" / "config_1" / "cluster_labels.npy"),
        "config_str": "PCA=30, perp=50, mcs=50, ms=10",
        "n_clusters": 3,
        "noise_pct": 9.8,
    },
}

SUBSAMPLE_RATE = 5
MIN_STEPS = 20  # minimum subsampled steps for an agent to be shown
N_AGENTS = 6    # number of agents to show per policy
OUTPUT_DIR = REPO_ROOT / "experiments" / "lifetime_clusters"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_trajectories(metadata, labels):
    """Group labels by (env_id, agent_id) trajectory, sorted by step.

    Returns dict: (env_id, agent_id) -> list of (step, label).
    """
    trajs = defaultdict(list)
    for i in range(len(labels)):
        key = (int(metadata["env_id"][i]), int(metadata["agent_id"][i]))
        trajs[key].append((int(metadata["step"][i]), int(labels[i])))
    for key in trajs:
        trajs[key].sort(key=lambda x: x[0])
    return trajs


def count_transitions(traj):
    """Count how many times the cluster label changes between consecutive steps."""
    n = 0
    for i in range(1, len(traj)):
        if traj[i][1] != traj[i-1][1]:
            n += 1
    return n


def select_interesting_agents(trajs, n_agents=N_AGENTS, min_steps=MIN_STEPS):
    """Select agents that have many cluster transitions and long lifetimes.

    Ranks by number of transitions, breaking ties by trajectory length.
    """
    candidates = []
    for key, steps in trajs.items():
        if len(steps) < min_steps:
            continue
        trans = count_transitions(steps)
        candidates.append((key, len(steps), trans))

    # Sort by transitions (desc), then by length (desc)
    candidates.sort(key=lambda x: (-x[2], -x[1]))
    return [c[0] for c in candidates[:n_agents]]


def get_cluster_colors(n_clusters):
    """Return a color map: cluster_id -> RGBA, with -1 mapped to gray.

    Uses tab10 for up to 10 clusters, matching the 3D scatter plots.
    """
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {}
    for i in range(n_clusters):
        colors[i] = cmap(i)
    colors[-1] = (0.78, 0.78, 0.78, 1.0)  # light gray for noise
    return colors


# ---------------------------------------------------------------------------
# Plot 1: Swim-lane / timeline chart
# ---------------------------------------------------------------------------

def plot_lifetimes(trajs, agent_keys, cluster_colors, n_clusters, policy_name,
                   config_str, noise_pct, output_path):
    """Create a swim-lane chart of cluster membership across agent lifetimes."""
    n_agents = len(agent_keys)
    fig, ax = plt.subplots(figsize=(16, max(3, 0.7 * n_agents + 1.5)))

    bar_height = 0.7

    for row, key in enumerate(agent_keys):
        traj = trajs[key]
        steps = [t[0] for t in traj]
        labels = [t[1] for t in traj]

        for i, (step, label) in enumerate(zip(steps, labels)):
            # Determine bar width: extend to next step or use average spacing
            if i < len(steps) - 1:
                width = steps[i + 1] - step
            elif len(steps) > 1:
                width = steps[-1] - steps[-2]  # use last interval
            else:
                width = 1

            color = cluster_colors.get(label, (0.5, 0.5, 0.5, 1.0))
            ax.barh(row, width, left=step, height=bar_height, color=color,
                    edgecolor="none", linewidth=0)

    # Y-axis: agent labels
    ax.set_yticks(range(n_agents))
    ylabels = []
    for key in agent_keys:
        traj = trajs[key]
        n_trans = count_transitions(traj)
        cluster_counts = Counter(t[1] for t in traj)
        dominant = cluster_counts.most_common(1)[0]
        dom_str = f"C{dominant[0]}" if dominant[0] != -1 else "noise"
        ylabels.append(
            f"env={key[0]}, agent={key[1]}\n"
            f"{len(traj)} steps, {n_trans} trans, dom={dom_str}"
        )
    ax.set_yticklabels(ylabels, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Tick (simulation timestep)", fontsize=11)
    ax.set_title(
        f"{policy_name} — Cluster Membership Over Agent Lifetimes\n"
        f"Config: {config_str} | {n_clusters} clusters, {noise_pct}% noise | "
        f"Top {n_agents} agents by transitions (subsample={SUBSAMPLE_RATE})",
        fontsize=12, fontweight="bold",
    )

    # Legend
    handles = []
    for c in range(n_clusters):
        handles.append(mpatches.Patch(color=cluster_colors[c], label=f"Cluster {c}"))
    handles.append(mpatches.Patch(color=cluster_colors[-1], label="Noise (-1)"))
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_ylim(-0.5, n_agents - 0.5)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Transition heatmap
# ---------------------------------------------------------------------------

def plot_transitions(trajs, cluster_colors, n_clusters, policy_name,
                     config_str, noise_pct, output_path):
    """Create a heatmap of cluster-to-cluster transition frequencies.

    Counts transitions across ALL agents (not just the selected ones), since
    we want a population-level view.
    """
    # Label set: -1 (noise) + 0..n_clusters-1
    all_labels = [-1] + list(range(n_clusters))
    n_labels = len(all_labels)
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    # Count transitions
    trans_counts = np.zeros((n_labels, n_labels), dtype=np.int64)
    total_transitions = 0
    total_steps = 0

    for key, traj in trajs.items():
        for i in range(1, len(traj)):
            from_label = traj[i - 1][1]
            to_label = traj[i][1]
            fi = label_to_idx.get(from_label)
            ti = label_to_idx.get(to_label)
            if fi is not None and ti is not None:
                trans_counts[fi, ti] += 1
                total_steps += 1
                if from_label != to_label:
                    total_transitions += 1

    # Normalize to row percentages (from-cluster -> to-cluster probability)
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_probs = np.where(row_sums > 0, trans_counts / row_sums * 100, 0)

    transition_rate = total_transitions / max(total_steps, 1) * 100

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: raw counts
    ax = axes[0]
    im = ax.imshow(trans_counts, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    tick_labels = ["noise"] + [f"C{i}" for i in range(n_clusters)]
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("To cluster", fontsize=11)
    ax.set_ylabel("From cluster", fontsize=11)
    ax.set_title("Transition Counts (raw)", fontsize=11)

    # Annotate cells
    for i in range(n_labels):
        for j in range(n_labels):
            val = trans_counts[i, j]
            if val > 0:
                text_color = "white" if val > trans_counts.max() * 0.6 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center",
                        fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8)

    # Right: row-normalized probabilities
    ax = axes[1]
    im2 = ax.imshow(trans_probs, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("To cluster", fontsize=11)
    ax.set_ylabel("From cluster", fontsize=11)
    ax.set_title("Transition Probabilities (row %)", fontsize=11)

    for i in range(n_labels):
        for j in range(n_labels):
            val = trans_probs[i, j]
            if val > 0.5:
                text_color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    fig.colorbar(im2, ax=ax, shrink=0.8, label="%")

    fig.suptitle(
        f"{policy_name} — Cluster Transition Heatmap\n"
        f"Config: {config_str} | {n_clusters} clusters, {noise_pct}% noise | "
        f"Overall transition rate: {transition_rate:.1f}% "
        f"({total_transitions}/{total_steps} consecutive pairs)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    return trans_counts, trans_probs, transition_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for policy_name, cfg in POLICIES.items():
        print(f"\n{'='*60}")
        print(f"Processing {policy_name}")
        print(f"  Config: {cfg['config_str']}")
        print(f"{'='*60}")

        # Load data
        print("  Loading data...", flush=True)
        records, _ = load_data(cfg["data_dir"], subsample_rate=SUBSAMPLE_RATE)
        _, _, metadata = compute_features(records)

        # Load cluster labels
        labels = np.load(cfg["label_path"])
        assert len(labels) == len(metadata["step"]), (
            f"Label count mismatch: {len(labels)} labels vs {len(metadata['step'])} records"
        )
        print(f"  Labels loaded: {len(labels)} records, "
              f"clusters: {sorted(set(labels) - {-1})}, "
              f"noise: {(labels == -1).sum()} ({(labels == -1).mean()*100:.1f}%)")

        # Build trajectories
        trajs = build_trajectories(metadata, labels)
        print(f"  {len(trajs)} trajectories")

        # Select interesting agents
        agent_keys = select_interesting_agents(trajs)
        print(f"  Selected {len(agent_keys)} agents for visualization:")
        for key in agent_keys:
            traj = trajs[key]
            n_trans = count_transitions(traj)
            print(f"    env={key[0]:2d}, agent={key[1]:2d}: "
                  f"{len(traj)} steps, {n_trans} transitions")

        n_clusters = cfg["n_clusters"]
        cluster_colors = get_cluster_colors(n_clusters)

        # Plot 1: Lifetimes
        lifetime_path = OUTPUT_DIR / f"{policy_name}_lifetimes.png"
        plot_lifetimes(
            trajs, agent_keys, cluster_colors, n_clusters,
            policy_name, cfg["config_str"], cfg["noise_pct"],
            str(lifetime_path),
        )

        # Plot 2: Transitions
        transition_path = OUTPUT_DIR / f"{policy_name}_transitions.png"
        trans_counts, trans_probs, transition_rate = plot_transitions(
            trajs, cluster_colors, n_clusters,
            policy_name, cfg["config_str"], cfg["noise_pct"],
            str(transition_path),
        )

        # Print summary stats
        print(f"\n  Transition rate: {transition_rate:.1f}%")
        tick_labels = ["noise"] + [f"C{i}" for i in range(n_clusters)]
        print(f"  Transition probability matrix (row -> col %):")
        header = "  " + f"{'From':>8s}" + "".join(f"{l:>8s}" for l in tick_labels)
        print(header)
        for i, from_label in enumerate(tick_labels):
            row_str = "  " + f"{from_label:>8s}"
            for j in range(len(tick_labels)):
                row_str += f"{trans_probs[i, j]:7.1f}%"
            print(row_str)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
