#!/usr/bin/env python
"""Visualize cluster membership alongside smoothed behavioral features for agent lifetimes.

For each policy, creates a figure showing 4-6 long-lived agents. Each agent gets
two vertically stacked subplots:
  1. Top: Smoothed feature time series (health, food, combat level, equipped items, visible players)
  2. Bottom: Cluster membership strip (colored bar showing cluster assignment at each tick)

Both subplots share the same x-axis (tick) so cluster transitions can be visually
correlated with feature changes.

Configs:
  - takeru_200M config 4: PCA=30, perp=50, mcs=200, ms=5
  - yaofeng_200M config 1: PCA=30, perp=50, mcs=50, ms=10

Usage:
    uv run python experiments/lifetime_features_comparison.py
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Add scripts/ to path so we can import analyze_activations
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from analyze_activations import load_data, compute_features, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLICIES = {
    "takeru_200M": {
        "data_dir": str(REPO_ROOT / "activation_data" / "takeru_200M"),
        "label_path": str(
            REPO_ROOT / "experiments" / "T-SNE-3d-sweep-takeru_200M" / "config_4" / "cluster_labels.npy"
        ),
        "config_str": "PCA=30, perp=50, mcs=200, ms=5",
        "n_clusters": 4,
    },
    "yaofeng_200M": {
        "data_dir": str(REPO_ROOT / "activation_data" / "yaofeng_200M"),
        "label_path": str(
            REPO_ROOT / "experiments" / "T-SNE-3d-sweep-yaofeng_200M" / "config_1" / "cluster_labels.npy"
        ),
        "config_str": "PCA=30, perp=50, mcs=50, ms=10",
        "n_clusters": 3,
    },
}

SUBSAMPLE_RATE = 5
MIN_STEPS = 30  # minimum subsampled steps for an agent to be shown
N_AGENTS = 5    # number of agents to show per policy
SMOOTH_WINDOW = 7  # rolling mean window for feature smoothing
# Gap detection: if consecutive ticks differ by more than this many normal
# subsample intervals, we treat it as a death/respawn gap and break the line.
GAP_THRESHOLD_FACTOR = 3  # gap > subsample_rate * this factor => break

OUTPUT_DIR = REPO_ROOT / "experiments" / "lifetime_clusters"

# Features to plot: (feature_name, index_in_FEATURE_NAMES, color, display_name)
# Verified indices from FEATURE_NAMES:
#   0: n_visible_entities, 1: n_visible_npcs, 2: n_visible_players,
#   3: nearest_entity_dist, 4: nearest_player_dist,
#   5: self_health, 6: self_food, 7: self_water, 8: self_gold,
#   9: max_combat_level, 10: melee_level, ...
#   21: item_level, 22: n_inventory_items, 23: n_equipped_items, 24: n_listed_items
FEATURES_TO_PLOT = [
    (5, "#d62728", "health"),         # self_health - red
    (6, "#ff7f0e", "food"),           # self_food - orange
    (9, "#1f77b4", "combat_level"),   # max_combat_level - blue
    (23, "#2ca02c", "equipped"),      # n_equipped_items - green
    (2, "#9467bd", "visible_players"),  # n_visible_players - purple
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_trajectories(metadata, labels, features):
    """Group data by (env_id, agent_id) trajectory, sorted by tick.

    Returns dict: (env_id, agent_id) -> {
        'ticks': list of int,
        'labels': list of int,
        'features': np.ndarray of shape (n_steps, n_features),
    }
    """
    trajs = defaultdict(lambda: {"ticks": [], "labels": [], "feat_rows": []})
    for i in range(len(labels)):
        key = (int(metadata["env_id"][i]), int(metadata["agent_id"][i]))
        trajs[key]["ticks"].append(int(metadata["step"][i]))
        trajs[key]["labels"].append(int(labels[i]))
        trajs[key]["feat_rows"].append(i)

    # Sort by tick within each trajectory and build feature arrays
    result = {}
    for key, data in trajs.items():
        order = np.argsort(data["ticks"])
        ticks = np.array(data["ticks"])[order]
        labs = np.array(data["labels"])[order]
        feat_idx = np.array(data["feat_rows"])[order]
        feats = features[feat_idx]
        result[key] = {"ticks": ticks, "labels": labs, "features": feats}

    return result


def find_life_segments(ticks, gap_threshold=None):
    """Find contiguous life segments by detecting death/respawn gaps.

    Returns list of (start_idx, end_idx) slices for each life segment.
    A gap is detected when consecutive ticks differ by more than gap_threshold.
    """
    if gap_threshold is None:
        gap_threshold = SUBSAMPLE_RATE * GAP_THRESHOLD_FACTOR

    segments = []
    seg_start = 0
    for i in range(1, len(ticks)):
        if ticks[i] - ticks[i - 1] > gap_threshold:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, len(ticks)))
    return segments


def count_transitions(labels):
    """Count how many times the cluster label changes between consecutive steps."""
    n = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            n += 1
    return n


def select_interesting_agents(trajs, n_agents=N_AGENTS, min_steps=MIN_STEPS):
    """Select agents with many cluster transitions and long lifetimes."""
    candidates = []
    for key, data in trajs.items():
        n_steps = len(data["ticks"])
        if n_steps < min_steps:
            continue
        n_trans = count_transitions(data["labels"])
        # Require at least some transitions to be interesting
        if n_trans < 2:
            continue
        candidates.append((key, n_steps, n_trans))

    # Sort by transitions (desc), then by length (desc)
    candidates.sort(key=lambda x: (-x[2], -x[1]))
    return [c[0] for c in candidates[:n_agents]]


def get_cluster_colors(n_clusters):
    """Return a color map: cluster_id -> RGBA, with -1 mapped to gray."""
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {}
    for i in range(n_clusters):
        colors[i] = cmap(i)
    colors[-1] = (0.78, 0.78, 0.78, 1.0)  # light gray for noise
    return colors


def smooth_features(features_array, window=SMOOTH_WINDOW):
    """Apply rolling mean smoothing to each feature column.

    Uses pandas rolling with min_periods=1 so edges are not lost.
    """
    df = pd.DataFrame(features_array)
    smoothed = df.rolling(window=window, min_periods=1, center=True).mean()
    return smoothed.values


def normalize_01(arr):
    """Normalize array to [0, 1] range."""
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_features_and_clusters(
    trajs, agent_keys, cluster_colors, n_clusters,
    policy_name, config_str, output_path
):
    """Create a figure with paired feature time-series and cluster strips per agent."""
    n_agents = len(agent_keys)

    # Height ratios: for each agent, feature plot (3) + cluster strip (1)
    # Plus a small gap between agents
    height_ratios = []
    for i in range(n_agents):
        height_ratios.extend([3, 1])

    fig, axes = plt.subplots(
        nrows=n_agents * 2,
        ncols=1,
        figsize=(20, 3.2 * n_agents),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05},
        squeeze=False,
    )
    axes = axes.flatten()

    # Compute noise percentage
    all_labels = np.concatenate([trajs[k]["labels"] for k in trajs])
    noise_pct = (all_labels == -1).mean() * 100

    for agent_idx, key in enumerate(agent_keys):
        data = trajs[key]
        ticks = data["ticks"]
        labels = data["labels"]
        raw_features = data["features"]

        # Detect life segments (split at death/respawn gaps)
        segments = find_life_segments(ticks)

        ax_feat = axes[agent_idx * 2]       # feature time series
        ax_strip = axes[agent_idx * 2 + 1]  # cluster strip

        # --- Feature time series ---
        # Smooth per-life-segment (so smoothing doesn't bleed across deaths),
        # then normalize across the whole trajectory for consistent scale,
        # then plot each segment separately so matplotlib does not draw lines
        # across death gaps.
        smoothed_segments = []
        for seg_start, seg_end in segments:
            seg_feats = raw_features[seg_start:seg_end]
            smoothed_segments.append(smooth_features(seg_feats, window=SMOOTH_WINDOW))

        for feat_plot_idx, (feat_idx, color, display_name) in enumerate(FEATURES_TO_PLOT):
            # Gather all smoothed values for this feature to compute global normalization
            all_vals = np.concatenate([s[:, feat_idx] for s in smoothed_segments])
            vmin, vmax = all_vals.min(), all_vals.max()

            for seg_i, (seg_start, seg_end) in enumerate(segments):
                seg_ticks = ticks[seg_start:seg_end]
                seg_vals = smoothed_segments[seg_i][:, feat_idx]
                # Normalize using global min/max
                if vmax - vmin < 1e-10:
                    seg_normed = np.zeros_like(seg_vals)
                else:
                    seg_normed = (seg_vals - vmin) / (vmax - vmin)
                # Only add label on the first segment to avoid legend duplicates
                lbl = display_name if seg_i == 0 else None
                ax_feat.plot(seg_ticks, seg_normed, color=color, linewidth=1.2,
                             label=lbl, alpha=0.85)

        # Draw vertical markers at death/respawn boundaries
        for seg_i in range(len(segments) - 1):
            _, seg_end = segments[seg_i]
            next_start, _ = segments[seg_i + 1]
            gap_left = ticks[seg_end - 1]
            gap_right = ticks[next_start]
            # Shade the dead region
            ax_feat.axvspan(gap_left, gap_right, alpha=0.12, color="red",
                            zorder=0)
            ax_strip.axvspan(gap_left, gap_right, alpha=0.12, color="red",
                             zorder=0)

        # Agent info label
        n_trans = count_transitions(labels)
        cluster_counts = Counter(labels)
        dominant = cluster_counts.most_common(1)[0]
        dom_str = f"C{dominant[0]}" if dominant[0] != -1 else "noise"
        n_lives = len(segments)
        lives_str = f", {n_lives} lives" if n_lives > 1 else ""

        ax_feat.set_ylabel(
            f"env={key[0]}, agent={key[1]}\n"
            f"{len(ticks)} steps, {n_trans} trans\ndom={dom_str}{lives_str}",
            fontsize=8, fontfamily="monospace", rotation=0,
            labelpad=90, ha="left", va="center",
        )
        ax_feat.set_ylim(-0.05, 1.15)
        ax_feat.set_xlim(ticks[0], ticks[-1])
        ax_feat.grid(True, alpha=0.15)

        # Only show legend on first agent
        if agent_idx == 0:
            ax_feat.legend(
                loc="upper right", fontsize=8, ncol=len(FEATURES_TO_PLOT),
                framealpha=0.9, columnspacing=1.0,
            )

        # Hide x-axis labels on feature plot (shared with strip below)
        ax_feat.tick_params(axis="x", labelbottom=False)
        ax_feat.tick_params(axis="y", labelsize=7)

        # --- Cluster membership strip ---
        # Only draw bars within life segments, leaving gaps empty (dead region)
        for seg_start, seg_end in segments:
            for i in range(seg_start, seg_end):
                tick = ticks[i]
                label = labels[i]
                # Bar width: extend to next tick within the same segment
                if i < seg_end - 1:
                    width = ticks[i + 1] - tick
                elif seg_end - seg_start > 1:
                    width = ticks[i] - ticks[i - 1]
                else:
                    width = SUBSAMPLE_RATE
                color = cluster_colors.get(label, (0.5, 0.5, 0.5, 1.0))
                ax_strip.barh(0, width, left=tick, height=1.0, color=color,
                              edgecolor="none", linewidth=0)

        ax_strip.set_xlim(ticks[0], ticks[-1])
        ax_strip.set_ylim(-0.5, 0.5)
        ax_strip.set_yticks([])
        ax_strip.grid(True, axis="x", alpha=0.15)

        # Only show tick labels on the bottom-most strip
        if agent_idx < n_agents - 1:
            ax_strip.tick_params(axis="x", labelbottom=False)
        else:
            ax_strip.set_xlabel("Tick (simulation timestep)", fontsize=11)
            ax_strip.tick_params(axis="x", labelsize=8)

        # Add thin separator line between agent blocks (except after last)
        if agent_idx < n_agents - 1:
            ax_strip.spines["bottom"].set_linewidth(2)
            ax_strip.spines["bottom"].set_color("black")

    # Cluster legend at the bottom
    handles = []
    for c in range(n_clusters):
        handles.append(mpatches.Patch(color=cluster_colors[c], label=f"Cluster {c}"))
    handles.append(mpatches.Patch(color=cluster_colors[-1], label="Noise (-1)"))
    fig.legend(
        handles=handles, loc="lower center", ncol=n_clusters + 1,
        fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"{policy_name} — Feature Time Series & Cluster Membership\n"
        f"Config: {config_str} | {n_clusters} clusters, {noise_pct:.1f}% noise | "
        f"Features normalized to [0,1], smoothed (window={SMOOTH_WINDOW}) | "
        f"subsample={SUBSAMPLE_RATE}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for policy_name, cfg in POLICIES.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {policy_name}")
        print(f"  Config: {cfg['config_str']}")
        print(f"{'=' * 60}")

        # Load data
        print("  Loading data...", flush=True)
        records, _ = load_data(cfg["data_dir"], subsample_rate=SUBSAMPLE_RATE)
        _, features, metadata = compute_features(records)

        # Load cluster labels
        labels = np.load(cfg["label_path"])
        assert len(labels) == len(metadata["step"]), (
            f"Label count mismatch: {len(labels)} labels vs {len(metadata['step'])} records"
        )
        n_clusters = cfg["n_clusters"]
        noise_count = (labels == -1).sum()
        noise_pct = noise_count / len(labels) * 100
        print(f"  Labels loaded: {len(labels)} records, "
              f"clusters: {sorted(set(labels) - {-1})}, "
              f"noise: {noise_count} ({noise_pct:.1f}%)")

        # Build trajectories (includes features)
        trajs = build_trajectories(metadata, labels, features)
        print(f"  {len(trajs)} unique agent trajectories")

        # Select interesting agents
        agent_keys = select_interesting_agents(trajs)
        print(f"  Selected {len(agent_keys)} agents for visualization:")
        for key in agent_keys:
            data = trajs[key]
            n_trans = count_transitions(data["labels"])
            print(f"    env={key[0]:2d}, agent={key[1]:2d}: "
                  f"{len(data['ticks'])} steps, {n_trans} transitions")

        cluster_colors = get_cluster_colors(n_clusters)

        # Generate the combined features + clusters plot
        output_path = OUTPUT_DIR / f"{policy_name}_features_and_clusters.png"
        plot_features_and_clusters(
            trajs, agent_keys, cluster_colors, n_clusters,
            policy_name, cfg["config_str"], str(output_path),
        )

        # Print per-agent feature-cluster correlation summary
        print(f"\n  Per-agent summary (feature means by dominant cluster):")
        for key in agent_keys:
            data = trajs[key]
            labels_arr = data["labels"]
            feats = data["features"]
            unique_clusters = sorted(set(labels_arr) - {-1})
            print(f"    Agent env={key[0]}, id={key[1]}:")
            for c in unique_clusters:
                mask = labels_arr == c
                if mask.sum() < 3:
                    continue
                means = feats[mask].mean(axis=0)
                parts = []
                for feat_idx, _, display_name in FEATURES_TO_PLOT:
                    parts.append(f"{display_name}={means[feat_idx]:.1f}")
                print(f"      Cluster {c} ({mask.sum():3d} ticks): {', '.join(parts)}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
