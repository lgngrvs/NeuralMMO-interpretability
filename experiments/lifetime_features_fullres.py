#!/usr/bin/env python
"""Full-resolution lifetime cluster membership + feature plots using KNN-inferred labels.

T-SNE cluster labels exist only for subsampled data (subsample=5). This script:
1. Loads subsampled data (subsample=5) to match existing cluster labels.
2. Loads full-resolution data (subsample=1).
3. Fits PCA(n_components=30) on subsampled activations (matching the sweep).
4. Trains a KNeighborsClassifier on (PCA-reduced subsampled acts -> cluster label),
   excluding noise points (label=-1) from training.
5. Predicts cluster labels for ALL full-resolution ticks.
6. Plots per-agent: feature time series + cluster strip at full tick resolution,
   with lives concatenated (relative tick x-axis) and death separators.

Configs:
  - takeru_200M config 4: PCA=30, perp=50, mcs=200, ms=5
  - yaofeng_200M config 1: PCA=30, perp=50, mcs=50, ms=10

Usage:
    uv run python experiments/lifetime_features_fullres.py
"""

import gc
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

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
PCA_COMPONENTS = 30
KNN_NEIGHBORS = 15
KNN_CONFIDENCE_THRESHOLD = 0.5
MIN_STEPS = 150        # minimum full-res steps for an agent to be shown
N_AGENTS = 5           # number of agents to show per policy
SMOOTH_WINDOW = 7      # rolling mean window for feature smoothing
GAP_THRESHOLD = 3      # gap > this many ticks => death boundary
GAP_VISUAL_WIDTH = 10  # visual gap width between lives on the x-axis

OUTPUT_DIR = REPO_ROOT / "experiments" / "lifetime_clusters"

# Features to plot: (feature_index, color, display_name)
FEATURES_TO_PLOT = [
    (5, "#d62728", "health"),           # self_health - red
    (6, "#ff7f0e", "food"),             # self_food - orange
    (9, "#1f77b4", "combat_level"),     # max_combat_level - blue
    (23, "#2ca02c", "equipped"),        # n_equipped_items - green
    (2, "#9467bd", "visible_players"),  # n_visible_players - purple
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_trajectories(metadata, labels, features):
    """Group data by (env_id, agent_id) trajectory, sorted by tick."""
    trajs = defaultdict(lambda: {"ticks": [], "labels": [], "feat_rows": []})
    for i in range(len(labels)):
        key = (int(metadata["env_id"][i]), int(metadata["agent_id"][i]))
        trajs[key]["ticks"].append(int(metadata["step"][i]))
        trajs[key]["labels"].append(int(labels[i]))
        trajs[key]["feat_rows"].append(i)

    result = {}
    for key, data in trajs.items():
        order = np.argsort(data["ticks"])
        ticks = np.array(data["ticks"])[order]
        labs = np.array(data["labels"])[order]
        feat_idx = np.array(data["feat_rows"])[order]
        feats = features[feat_idx]
        result[key] = {"ticks": ticks, "labels": labs, "features": feats}

    return result


def find_life_segments(ticks, gap_threshold=GAP_THRESHOLD):
    """Find contiguous life segments by detecting death/respawn gaps."""
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
    return int(np.sum(np.diff(labels) != 0))


def select_interesting_agents(trajs, n_agents=N_AGENTS, min_steps=MIN_STEPS):
    """Select agents with many cluster transitions and long lifetimes."""
    candidates = []
    for key, data in trajs.items():
        n_steps = len(data["ticks"])
        if n_steps < min_steps:
            continue
        n_trans = count_transitions(data["labels"])
        if n_trans < 2:
            continue
        candidates.append((key, n_steps, n_trans))

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


def smooth_features_per_segment(features_array, segments, window=SMOOTH_WINDOW):
    """Apply rolling mean smoothing per life segment (not across death boundaries)."""
    smoothed = np.copy(features_array).astype(np.float64)
    for seg_start, seg_end in segments:
        seg = features_array[seg_start:seg_end]
        df = pd.DataFrame(seg)
        seg_smoothed = df.rolling(window=window, min_periods=1, center=True).mean()
        smoothed[seg_start:seg_end] = seg_smoothed.values
    return smoothed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fullres_agent(ax_feat, ax_strip, data, cluster_colors, is_last, is_first):
    """Plot feature time series and cluster strip for one agent using relative ticks.

    Lives are concatenated with visual gaps and death separators.
    Uses broken_barh for efficient cluster strip rendering.
    """
    ticks = data["ticks"]
    labels = data["labels"]
    raw_features = data["features"]

    segments = find_life_segments(ticks)

    # Smooth features per-segment
    smoothed = smooth_features_per_segment(raw_features, segments, window=SMOOTH_WINDOW)

    # Build concatenated x-positions with gaps between lives
    concat_x = np.zeros(len(ticks), dtype=np.float64)
    death_x_positions = []
    cumulative_offset = 0.0

    for seg_i, (seg_start, seg_end) in enumerate(segments):
        seg_ticks = ticks[seg_start:seg_end]
        rel_ticks = seg_ticks - seg_ticks[0]
        concat_x[seg_start:seg_end] = rel_ticks + cumulative_offset

        seg_len = rel_ticks[-1] + 1
        cumulative_offset += seg_len

        if seg_i < len(segments) - 1:
            death_x_positions.append(cumulative_offset + GAP_VISUAL_WIDTH / 2.0)
            cumulative_offset += GAP_VISUAL_WIDTH

    total_x_range = cumulative_offset

    # --- Feature time series ---
    for feat_plot_idx, (feat_idx, color, display_name) in enumerate(FEATURES_TO_PLOT):
        all_vals = smoothed[:, feat_idx]
        vmin, vmax = all_vals.min(), all_vals.max()

        for seg_i, (seg_start, seg_end) in enumerate(segments):
            seg_x = concat_x[seg_start:seg_end]
            seg_vals = smoothed[seg_start:seg_end, feat_idx]
            if vmax - vmin < 1e-10:
                seg_normed = np.zeros_like(seg_vals)
            else:
                seg_normed = (seg_vals - vmin) / (vmax - vmin)
            lbl = display_name if seg_i == 0 else None
            ax_feat.plot(seg_x, seg_normed, color=color, linewidth=0.9,
                         label=lbl, alpha=0.85)

    # Draw death separators
    for dx in death_x_positions:
        ax_feat.axvline(dx, color="black", linewidth=1.5, linestyle="-", alpha=0.7)
        ax_strip.axvline(dx, color="black", linewidth=1.5, linestyle="-", alpha=0.7)
        ax_feat.text(dx, 1.08, "died", ha="center", va="bottom", fontsize=6,
                     color="red", fontweight="bold", alpha=0.8)

    # Agent info label
    n_trans = count_transitions(labels)
    cluster_counts = Counter(labels.tolist())
    dominant = cluster_counts.most_common(1)[0]
    dom_str = f"C{dominant[0]}" if dominant[0] != -1 else "noise"
    n_lives = len(segments)
    lives_str = f", {n_lives} lives" if n_lives > 1 else ""
    noise_pct_agent = (labels == -1).mean() * 100

    key = data.get("key", ("?", "?"))
    ax_feat.set_ylabel(
        f"env={key[0]}, agent={key[1]}\n"
        f"{len(ticks)} ticks, {n_trans} trans\n"
        f"dom={dom_str}{lives_str}\n"
        f"noise={noise_pct_agent:.0f}%",
        fontsize=7, fontfamily="monospace", rotation=0,
        labelpad=100, ha="left", va="center",
    )
    ax_feat.set_ylim(-0.05, 1.20)
    ax_feat.set_xlim(0, total_x_range)
    ax_feat.grid(True, alpha=0.15)

    if is_first:
        ax_feat.legend(
            loc="upper right", fontsize=7, ncol=len(FEATURES_TO_PLOT),
            framealpha=0.9, columnspacing=1.0,
        )

    ax_feat.tick_params(axis="x", labelbottom=False)
    ax_feat.tick_params(axis="y", labelsize=6)

    # --- Cluster membership strip (using broken_barh for efficiency) ---
    # Group consecutive ticks with the same label into runs
    for seg_i, (seg_start, seg_end) in enumerate(segments):
        seg_labels = labels[seg_start:seg_end]
        seg_x = concat_x[seg_start:seg_end]
        n_seg = len(seg_labels)

        if n_seg == 0:
            continue

        # Find runs of same label
        run_starts = [0]
        for j in range(1, n_seg):
            if seg_labels[j] != seg_labels[j - 1]:
                run_starts.append(j)

        for ri, rs in enumerate(run_starts):
            re = run_starts[ri + 1] if ri + 1 < len(run_starts) else n_seg
            x_start = seg_x[rs]
            if re < n_seg:
                x_end = seg_x[re]
            else:
                x_end = seg_x[re - 1] + 1.0
            width = x_end - x_start
            label_val = seg_labels[rs]
            color = cluster_colors.get(int(label_val), (0.5, 0.5, 0.5, 1.0))
            ax_strip.broken_barh([(x_start, width)], (-0.5, 1.0),
                                 facecolors=[color], edgecolors="none")

    ax_strip.set_xlim(0, total_x_range)
    ax_strip.set_ylim(-0.5, 0.5)
    ax_strip.set_yticks([])
    ax_strip.grid(True, axis="x", alpha=0.15)

    if is_last:
        ax_strip.set_xlabel("Relative tick (steps alive, reset per life)", fontsize=10)
        ax_strip.tick_params(axis="x", labelsize=7)
    else:
        ax_strip.tick_params(axis="x", labelbottom=False)


def plot_policy(
    trajs, agent_keys, cluster_colors, n_clusters,
    policy_name, config_str, output_path, noise_pct_global,
):
    """Create a full figure for one policy."""
    n_agents = len(agent_keys)

    # Height ratios: feature plot (5) + cluster strip (1) per agent
    height_ratios = []
    for _ in range(n_agents):
        height_ratios.extend([5, 1])

    fig, axes = plt.subplots(
        nrows=n_agents * 2,
        ncols=1,
        figsize=(24, 3.0 * n_agents),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
        squeeze=False,
    )
    axes = axes.flatten()

    for agent_idx, key in enumerate(agent_keys):
        data = trajs[key]
        data["key"] = key
        ax_feat = axes[agent_idx * 2]
        ax_strip = axes[agent_idx * 2 + 1]

        plot_fullres_agent(
            ax_feat, ax_strip, data, cluster_colors,
            is_last=(agent_idx == n_agents - 1),
            is_first=(agent_idx == 0),
        )

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
        f"{policy_name} -- Full resolution (no subsampling), KNN-inferred labels\n"
        f"Config: {config_str} | {n_clusters} clusters, {noise_pct_global:.1f}% noise (KNN) | "
        f"Features normalized to [0,1], smoothed (window={SMOOTH_WINDOW}) | "
        f"KNN k={KNN_NEIGHBORS}, confidence threshold={KNN_CONFIDENCE_THRESHOLD}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_policy(policy_name, cfg):
    """Process a single policy: load data, KNN inference, plot."""
    print(f"\n{'=' * 60}")
    print(f"Processing {policy_name}")
    print(f"  Config: {cfg['config_str']}")
    print(f"{'=' * 60}")

    # --- Step 1: Load subsampled data (matching cluster labels) ---
    print("  Loading subsampled data (subsample=5)...", flush=True)
    sub_records, _ = load_data(cfg["data_dir"], subsample_rate=SUBSAMPLE_RATE)
    sub_activations_raw, sub_features, sub_metadata = compute_features(sub_records)
    del sub_records, sub_features  # don't need features from subsampled data
    gc.collect()

    # Load cluster labels
    sub_labels = np.load(cfg["label_path"])
    assert len(sub_labels) == len(sub_metadata["step"]), (
        f"Label count mismatch: {len(sub_labels)} labels vs {len(sub_metadata['step'])} records"
    )
    n_clusters = cfg["n_clusters"]
    print(f"  Subsampled: {len(sub_labels)} records, "
          f"clusters: {sorted(set(sub_labels) - {-1})}, "
          f"noise: {(sub_labels == -1).sum()} ({(sub_labels == -1).mean() * 100:.1f}%)")

    # --- Step 2: Fit PCA on subsampled, then free subsampled activations ---
    print(f"  Fitting PCA(n_components={PCA_COMPONENTS}) on subsampled activations...", flush=True)
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    sub_pca = pca.fit_transform(sub_activations_raw)
    del sub_activations_raw
    gc.collect()
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # --- Step 3: Train KNN ---
    train_mask = sub_labels != -1
    n_train = train_mask.sum()
    print(f"  Training KNN (k={KNN_NEIGHBORS}) on {n_train} non-noise subsampled points...", flush=True)
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, n_jobs=1)
    knn.fit(sub_pca[train_mask], sub_labels[train_mask])
    del sub_pca, sub_labels, sub_metadata, train_mask
    gc.collect()

    # --- Step 4: Load full-resolution data ---
    print("  Loading full-resolution data (subsample=1)...", flush=True)
    full_records, _ = load_data(cfg["data_dir"], subsample_rate=1)
    full_activations_raw, full_features, full_metadata = compute_features(full_records)
    del full_records
    gc.collect()
    print(f"  Full-res: {len(full_metadata['step'])} records")

    # --- Step 5: PCA transform + KNN predict in batches ---
    print(f"  PCA-transforming {len(full_activations_raw)} full-res points...", flush=True)
    full_pca = pca.transform(full_activations_raw)
    del full_activations_raw, pca
    gc.collect()

    print(f"  Predicting labels in batches...", flush=True)
    BATCH_SIZE = 5000
    n_total = len(full_pca)
    full_labels = np.zeros(n_total, dtype=np.int64)
    max_probs = np.zeros(n_total, dtype=np.float64)

    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch_pca = full_pca[start:end]
        batch_labels = knn.predict(batch_pca)
        batch_probs = knn.predict_proba(batch_pca)
        batch_max_prob = batch_probs.max(axis=1)
        full_labels[start:end] = batch_labels
        max_probs[start:end] = batch_max_prob
        if (start // BATCH_SIZE) % 2 == 0:
            print(f"    ... {end}/{n_total} done", flush=True)

    del full_pca, knn
    gc.collect()

    # Low confidence -> noise
    low_conf_mask = max_probs < KNN_CONFIDENCE_THRESHOLD
    full_labels[low_conf_mask] = -1
    noise_count = (full_labels == -1).sum()
    noise_pct = noise_count / len(full_labels) * 100
    print(f"  KNN prediction complete: {len(full_labels)} points, "
          f"noise (low confidence): {noise_count} ({noise_pct:.1f}%), "
          f"mean max_prob: {max_probs.mean():.3f}, "
          f"median max_prob: {np.median(max_probs):.3f}")
    del max_probs, low_conf_mask
    gc.collect()

    # --- Step 6: Build trajectories ---
    trajs = build_trajectories(full_metadata, full_labels, full_features)
    del full_metadata, full_labels, full_features
    gc.collect()
    print(f"  {len(trajs)} unique agent trajectories")

    # Select interesting agents
    agent_keys = select_interesting_agents(trajs, n_agents=N_AGENTS, min_steps=MIN_STEPS)
    if len(agent_keys) < N_AGENTS:
        print(f"  WARNING: Only found {len(agent_keys)} agents meeting criteria "
              f"(min_steps={MIN_STEPS}, min_transitions=2). Relaxing min_steps...")
        agent_keys = select_interesting_agents(trajs, n_agents=N_AGENTS, min_steps=50)

    print(f"  Selected {len(agent_keys)} agents for visualization:")
    for key in agent_keys:
        data = trajs[key]
        n_trans = count_transitions(data["labels"])
        segments = find_life_segments(data["ticks"])
        print(f"    env={key[0]:2d}, agent={key[1]:2d}: "
              f"{len(data['ticks'])} ticks, {n_trans} transitions, "
              f"{len(segments)} lives")

    cluster_colors = get_cluster_colors(n_clusters)

    # Generate the plot
    output_path = OUTPUT_DIR / f"{policy_name}_fullres.png"
    plot_policy(
        trajs, agent_keys, cluster_colors, n_clusters,
        policy_name, cfg["config_str"], str(output_path), noise_pct,
    )

    # Print per-agent cluster summary
    print(f"\n  Per-agent summary (feature means by cluster):")
    for key in agent_keys:
        data = trajs[key]
        labels_arr = data["labels"]
        feats = data["features"]
        unique_clusters = sorted(set(labels_arr.tolist()) - {-1})
        print(f"    Agent env={key[0]}, id={key[1]}:")
        for c in unique_clusters:
            mask = labels_arr == c
            if mask.sum() < 3:
                continue
            means = feats[mask].mean(axis=0)
            parts = []
            for feat_idx, _, display_name in FEATURES_TO_PLOT:
                parts.append(f"{display_name}={means[feat_idx]:.1f}")
            print(f"      Cluster {c} ({mask.sum():4d} ticks): {', '.join(parts)}")

    # Print transition smoothness analysis
    print(f"\n  Transition smoothness analysis:")
    for key in agent_keys:
        data = trajs[key]
        labels_arr = data["labels"]
        n_total = len(labels_arr)
        n_trans = count_transitions(labels_arr)
        trans_rate = n_trans / max(n_total - 1, 1)

        # Compute run lengths
        diffs = np.where(np.diff(labels_arr) != 0)[0]
        if len(diffs) == 0:
            runs = np.array([n_total])
        else:
            run_ends = np.concatenate([[0], diffs + 1, [n_total]])
            runs = np.diff(run_ends)

        print(f"    Agent env={key[0]:2d}, id={key[1]:2d}: "
              f"transition_rate={trans_rate:.3f}, "
              f"mean_run={runs.mean():.1f}, "
              f"median_run={np.median(runs):.0f}, "
              f"max_run={runs.max()}, "
              f"n_runs={len(runs)}")

    # Clean up
    del trajs
    gc.collect()

    return noise_pct


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for policy_name, cfg in POLICIES.items():
        process_policy(policy_name, cfg)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
