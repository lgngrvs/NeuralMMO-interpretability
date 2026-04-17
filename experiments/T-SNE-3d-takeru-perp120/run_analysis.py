#!/usr/bin/env python
"""T-SNE 3D (perplexity=120) + HDBSCAN on combined Takeru 200M data.

Produces:
  - tsne_3d_scatter.png  (2x2 multi-angle 3D scatter)
  - cluster_features.png (Cohen's d heatmap)
  - summary.txt          (config, cluster descriptions, transition rates)
  - cluster_labels.npy, env_id.npy, agent_id.npy, step.npy
  - MCS sensitivity table (mcs=50,100,200,400 with fixed perp=120)

Usage:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python experiments/T-SNE-3d-takeru-perp120/run_analysis.py
"""

import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import hdbscan as hdbscan_lib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add scripts/ to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from analyze_activations import (
    FEATURE_NAMES,
    compute_cluster_stats,
    compute_features,
    load_data,
)
from sweep_cluster_params import score_combo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "T-SNE-3d-takeru-perp120")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCA_DIM = 20
TSNE_PERPLEXITY = 120
PRIMARY_MCS = 400
PRIMARY_MS_VALUES = [5, 10, 25]
MCS_SENSITIVITY = [50, 100, 200, 400]
SEED = 42

VIEWING_ANGLES = [
    (30, 45),
    (30, 135),
    (30, 225),
    (60, 0),
]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_3d_clusters(embedding, labels, config_str, output_path):
    """Save a 2x2 grid of 3D scatter plots from four viewing angles."""
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    cmap_name = "tab10" if n_clusters <= 10 else "tab20"
    cmap = plt.colormaps.get_cmap(cmap_name).resampled(max(n_clusters, 1))

    fig = plt.figure(figsize=(16, 14))

    for panel_idx, (elev, azim) in enumerate(VIEWING_ANGLES, start=1):
        ax = fig.add_subplot(2, 2, panel_idx, projection="3d")

        # Noise first (behind everything)
        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(
                embedding[noise_mask, 0],
                embedding[noise_mask, 1],
                embedding[noise_mask, 2],
                c="lightgray", s=1, alpha=0.3, label="noise", rasterized=True,
            )

        for i, cl in enumerate(cluster_ids):
            mask = labels == cl
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                c=[cmap(i)], s=3, alpha=0.5,
                label=f"C{cl} (n={mask.sum()})", rasterized=True,
            )

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("T-SNE 1", fontsize=8)
        ax.set_ylabel("T-SNE 2", fontsize=8)
        ax.set_zlabel("T-SNE 3", fontsize=8)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{config_str}\n{n_clusters} clusters, {noise_pct:.1f}% noise",
        fontsize=13, y=0.98,
    )

    handles, leg_labels = fig.axes[0].get_legend_handles_labels()
    if n_clusters <= 15:
        fig.legend(handles, leg_labels, loc="lower center",
                   ncol=min(n_clusters + 1, 6),
                   markerscale=4, fontsize=8, framealpha=0.9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def plot_feature_heatmap(stats, features, output_path,
                         title="Cohen's d per Cluster per Feature"):
    """Save a Cohen's d heatmap (RdBu_r, vmin=-3, vmax=3).

    Computes Cohen's d as (cluster_mean - global_mean) / global_std for
    display purposes. Annotates cells with |d| > 0.5.
    """
    cluster_ids = sorted(stats.keys())
    if not cluster_ids:
        return
    n_feats = len(FEATURE_NAMES)

    # Compute global stats
    global_mean = features.mean(axis=0)
    global_std = features.std(axis=0)
    global_std[global_std < 1e-8] = 1.0  # avoid div by zero

    # Build matrix: (cluster_mean - global_mean) / global_std
    d_matrix = np.zeros((len(cluster_ids), n_feats))
    for i, c in enumerate(cluster_ids):
        cluster_mean = stats[c]["means"]
        d_matrix[i] = (cluster_mean - global_mean) / global_std

    fig, ax = plt.subplots(figsize=(max(12, n_feats * 0.8),
                                    max(4, len(cluster_ids) * 0.6)))
    im = ax.imshow(d_matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"C{c} (n={stats[c]['size']})" for c in cluster_ids],
                       fontsize=8)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="Cohen's d  (cluster - global) / global_std")

    for i in range(len(cluster_ids)):
        for j in range(n_feats):
            val = d_matrix[i, j]
            if abs(val) > 0.5:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 1.5 else "black")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def compute_transition_rates(labels, metadata):
    """Compute cluster transition rates for temporal coherence analysis."""
    steps = metadata["step"]
    env_ids = metadata["env_id"]
    agent_ids = metadata["agent_id"]

    trajectories = defaultdict(list)
    for i in range(len(labels)):
        key = (int(env_ids[i]), int(agent_ids[i]))
        trajectories[key].append((int(steps[i]), int(labels[i])))

    total_transitions = 0
    total_same = 0
    total_different = 0

    for traj_key, points in trajectories.items():
        points.sort(key=lambda x: x[0])
        for j in range(1, len(points)):
            prev_label = points[j - 1][1]
            curr_label = points[j][1]
            if prev_label == -1 or curr_label == -1:
                continue
            total_transitions += 1
            if prev_label == curr_label:
                total_same += 1
            else:
                total_different += 1

    if total_transitions == 0:
        return {"transition_rate": 0.0, "total_transitions": 0,
                "same_cluster": 0, "different_cluster": 0,
                "n_trajectories": len(trajectories)}

    return {
        "transition_rate": total_different / total_transitions,
        "total_transitions": total_transitions,
        "same_cluster": total_same,
        "different_cluster": total_different,
        "n_trajectories": len(trajectories),
    }


def write_summary_txt(detail, mcs, ms, tr, output_path, N, features,
                      mcs_table, best_ms_results):
    """Write a human-readable summary."""
    stats = detail["stats"]

    # Compute global stats for our own d calculation
    global_mean = features.mean(axis=0)
    global_std = features.std(axis=0)
    global_std[global_std < 1e-8] = 1.0

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("T-SNE 3D + HDBSCAN Analysis: Combined Takeru 200M\n")
        f.write("=" * 80 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  PCA dimensions:    {PCA_DIM}\n")
        f.write(f"  T-SNE perplexity:  {TSNE_PERPLEXITY}\n")
        f.write(f"  T-SNE components:  3\n")
        f.write(f"  HDBSCAN mcs:       {mcs}\n")
        f.write(f"  HDBSCAN ms:        {ms}\n")
        f.write(f"  Total datapoints:  {N}\n")
        f.write(f"  Subsample rate:    10\n")
        f.write(f"  Random seed:       {SEED}\n\n")

        f.write("Results:\n")
        f.write(f"  Clusters:          {detail['n_clusters']} "
                f"(raw: {detail['n_clusters_raw']})\n")
        f.write(f"  Noise:             {detail['noise_pct']:.1f}%\n")
        f.write(f"  Composite score:   {detail['composite_score']:.4f}\n")
        f.write(f"  Passes verify:     {detail['passes_verification']}\n")
        f.write(f"  Mean weighted |d|: {detail['mean_weighted_d']:.3f}\n")
        f.write(f"  Entropy factor:    {detail['entropy_factor']:.3f}\n")
        f.write(f"  Min step entropy:  {detail['min_step_entropy']:.3f}\n")
        f.write(f"  Min agent entropy: {detail['min_agent_entropy']:.3f}\n")

        f.write(f"\n{'='*80}\n")
        f.write("Per-Cluster Details:\n")
        f.write(f"{'='*80}\n")

        for label in sorted(stats.keys()):
            s = stats[label]
            cluster_mean = s["means"]
            d_vals = (cluster_mean - global_mean) / global_std
            top_idx = np.argsort(np.abs(d_vals))[::-1]

            f.write(f"\nCluster {label} (n={s['size']}, "
                    f"{s['size']/N*100:.1f}%):\n")
            f.write(f"  Step entropy:  {s['step_entropy']:.3f}\n")
            f.write(f"  Agent entropy: {s['agent_entropy']:.3f}\n")
            f.write(f"  Top 5 features by |d| "
                    f"(cluster-global)/global_std:\n")
            for rank in range(min(5, len(FEATURE_NAMES))):
                idx = top_idx[rank]
                d = d_vals[idx]
                f.write(f"    {rank+1}. {FEATURE_NAMES[idx]:25s} "
                        f"d={d:+.2f}\n")

        # Transition rate info
        f.write(f"\n{'='*80}\n")
        f.write("Temporal Coherence (Transition Rate Analysis):\n")
        f.write(f"{'='*80}\n")
        f.write(f"  Transition rate:   {tr['transition_rate']:.4f}\n")
        f.write(f"  Total transitions: {tr['total_transitions']}\n")
        f.write(f"  Same cluster:      {tr['same_cluster']}\n")
        f.write(f"  Different cluster: {tr['different_cluster']}\n")
        f.write(f"  Trajectories:      {tr['n_trajectories']}\n")

        # min_samples comparison for mcs=400
        f.write(f"\n{'='*80}\n")
        f.write(f"min_samples Comparison (mcs={PRIMARY_MCS}):\n")
        f.write(f"{'='*80}\n")
        f.write(f"  {'ms':>4} | {'#Cl':>4} | {'Noise%':>7} | {'Score':>7} | "
                f"{'WtdD':>6} | {'TransRate':>9} | {'Pass':>5}\n")
        f.write(f"  {'-'*60}\n")
        for ms_val, info in sorted(best_ms_results.items()):
            f.write(f"  {ms_val:4d} | {info['n_clusters']:4d} | "
                    f"{info['noise_pct']:6.1f}% | "
                    f"{info['score']:7.3f} | "
                    f"{info['wtd_d']:6.2f} | "
                    f"{info['transition_rate']:9.4f} | "
                    f"{'YES' if info['passes'] else 'no':>5}\n")

        # MCS sensitivity table
        f.write(f"\n{'='*80}\n")
        f.write("MCS Sensitivity (best ms per mcs, same T-SNE embedding):\n")
        f.write(f"{'='*80}\n")
        f.write(f"  {'MCS':>6} | {'ms':>4} | {'#Cl':>4} | {'Noise%':>7} | "
                f"{'Score':>7} | {'WtdD':>6} | {'TransRate':>9} | {'Pass':>5}\n")
        f.write(f"  {'-'*70}\n")
        for mcs_val, info in sorted(mcs_table.items()):
            f.write(f"  {mcs_val:6d} | {info['ms']:4d} | "
                    f"{info['n_clusters']:4d} | "
                    f"{info['noise_pct']:6.1f}% | "
                    f"{info['score']:7.3f} | "
                    f"{info['wtd_d']:6.2f} | "
                    f"{info['transition_rate']:9.4f} | "
                    f"{'YES' if info['passes'] else 'no':>5}\n")

    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # ==================================================================
    # 1. Load and combine both datasets
    # ==================================================================
    print("=" * 80)
    print("Loading Dataset 1: activation_data/takeru_200M/")
    print("=" * 80, flush=True)
    data_dir1 = os.path.join(PROJECT_ROOT, "activation_data", "takeru_200M")
    records1, total1 = load_data(data_dir1, subsample_rate=10)
    act1, feat1, meta1 = compute_features(records1)
    print(f"  Dataset 1: {act1.shape[0]} records (from {total1} total)")
    del records1
    gc.collect()

    print()
    print("=" * 80)
    print("Loading Dataset 2: activation_data/takeru_200M_extra/takeru_200M/")
    print("=" * 80, flush=True)
    data_dir2 = os.path.join(PROJECT_ROOT, "activation_data",
                              "takeru_200M_extra", "takeru_200M")
    records2, total2 = load_data(data_dir2, subsample_rate=10)
    act2, feat2, meta2 = compute_features(records2)
    print(f"  Dataset 2: {act2.shape[0]} records (from {total2} total)")
    del records2
    gc.collect()

    # Combine
    activations = np.vstack([act1, act2])
    features = np.vstack([feat1, feat2])
    metadata = {k: np.concatenate([meta1[k], meta2[k]]) for k in meta1}

    del act1, act2, feat1, feat2, meta1, meta2
    gc.collect()

    N = len(activations)
    print(f"\nCombined data: {activations.shape}")
    print(f"Features: {features.shape}")
    print(f"Total points: {N}")

    min_trustworthy = max(20, int(0.02 * N))
    print(f"Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # ==================================================================
    # 2. PCA reduction
    # ==================================================================
    print(f"\nPCA to {PCA_DIM} dimensions...", flush=True)
    d = min(PCA_DIM, activations.shape[1], activations.shape[0])
    pca = PCA(n_components=d, random_state=SEED)
    act_pca = pca.fit_transform(activations)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {d}D: {explained:.1%} variance explained")

    del activations
    gc.collect()

    # ==================================================================
    # 3. T-SNE 3D (perplexity=120)
    # ==================================================================
    print(f"\nFitting T-SNE 3D (perplexity={TSNE_PERPLEXITY})...", flush=True)
    tsne = TSNE(
        n_components=3,
        perplexity=TSNE_PERPLEXITY,
        learning_rate="auto",
        max_iter=1000,
        random_state=SEED,
        verbose=1,
    )
    embedding = tsne.fit_transform(act_pca)
    t_tsne = time.time() - t_start
    print(f"  T-SNE done in {t_tsne:.0f}s")

    # ==================================================================
    # 4. Primary analysis: mcs=400 with ms=[5, 10, 25]
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"HDBSCAN with mcs={PRIMARY_MCS}, ms={PRIMARY_MS_VALUES}")
    print(f"{'='*80}")

    best_ms_results = {}
    primary_results = []

    for ms in PRIMARY_MS_VALUES:
        print(f"\n  --- mcs={PRIMARY_MCS}, ms={ms} ---")
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=PRIMARY_MCS, min_samples=ms)
        labels = clusterer.fit_predict(embedding)
        score, detail = score_combo(
            labels, features, metadata, min_trustworthy, SEED)

        tr = compute_transition_rates(labels, metadata)

        n_cl = detail["n_clusters"]
        noise_pct = detail["noise_pct"]
        print(f"    Clusters: {n_cl}, Noise: {noise_pct:.1f}%, "
              f"Score: {score:.3f}, "
              f"TransRate: {tr['transition_rate']:.4f}")

        best_ms_results[ms] = {
            "n_clusters": n_cl,
            "noise_pct": noise_pct,
            "score": score,
            "wtd_d": detail["mean_weighted_d"],
            "transition_rate": tr["transition_rate"],
            "passes": detail["passes_verification"],
        }
        primary_results.append((score, ms, labels, detail, tr))

    # Pick best ms for primary config
    primary_results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_ms, best_labels, best_detail, best_tr = primary_results[0]

    print(f"\n  Best: ms={best_ms} with score={best_score:.4f}")

    # ==================================================================
    # 5. MCS sensitivity: mcs=[50, 100, 200, 400]
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"MCS Sensitivity Table")
    print(f"{'='*80}")

    mcs_table = {}
    for mcs_val in MCS_SENSITIVITY:
        best_for_mcs = None
        for ms_val in [5, 10, 25]:
            if ms_val > mcs_val:
                continue
            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=mcs_val, min_samples=ms_val)
            labels_tmp = clusterer.fit_predict(embedding)
            score_tmp, detail_tmp = score_combo(
                labels_tmp, features, metadata, min_trustworthy, SEED)
            tr_tmp = compute_transition_rates(labels_tmp, metadata)

            if best_for_mcs is None or score_tmp > best_for_mcs[0]:
                best_for_mcs = (score_tmp, ms_val, labels_tmp,
                                detail_tmp, tr_tmp)

        if best_for_mcs is not None:
            sc, ms_v, _, det, tr_v = best_for_mcs
            mcs_table[mcs_val] = {
                "ms": ms_v,
                "n_clusters": det["n_clusters"],
                "noise_pct": det["noise_pct"],
                "score": sc,
                "wtd_d": det["mean_weighted_d"],
                "transition_rate": tr_v["transition_rate"],
                "passes": det["passes_verification"],
            }
            print(f"  mcs={mcs_val:4d}, best ms={ms_v:2d}: "
                  f"{det['n_clusters']} cl, {det['noise_pct']:.1f}% noise, "
                  f"score={sc:.3f}, trans={tr_v['transition_rate']:.4f}")

    # ==================================================================
    # 6. Generate outputs for the primary config (mcs=400, best ms)
    # ==================================================================
    print(f"\n{'='*80}")
    print(f"Generating visualizations for mcs={PRIMARY_MCS}, ms={best_ms}")
    print(f"{'='*80}")

    config_str = (f"Takeru 200M Combined | PCA={PCA_DIM}, "
                  f"perp={TSNE_PERPLEXITY}, "
                  f"mcs={PRIMARY_MCS}, ms={best_ms}")

    # 1. 3D scatter
    plot_3d_clusters(
        embedding, best_labels, config_str,
        os.path.join(OUTPUT_DIR, "tsne_3d_scatter.png"),
    )

    # 2. Feature heatmap
    stats = best_detail["stats"]
    plot_feature_heatmap(
        stats, features,
        os.path.join(OUTPUT_DIR, "cluster_features.png"),
        title=f"Cohen's d: (cluster - global) / global_std\n{config_str}",
    )

    # 3. Summary text
    write_summary_txt(
        best_detail, PRIMARY_MCS, best_ms, best_tr,
        os.path.join(OUTPUT_DIR, "summary.txt"),
        N, features, mcs_table, best_ms_results,
    )

    # 4. Save arrays
    np.save(os.path.join(OUTPUT_DIR, "cluster_labels.npy"), best_labels)
    np.save(os.path.join(OUTPUT_DIR, "env_id.npy"), metadata["env_id"])
    np.save(os.path.join(OUTPUT_DIR, "agent_id.npy"), metadata["agent_id"])
    np.save(os.path.join(OUTPUT_DIR, "step.npy"), metadata["step"])
    np.save(os.path.join(OUTPUT_DIR, "tsne_embedding.npy"), embedding)
    print(f"  Saved .npy arrays to {OUTPUT_DIR}/")

    # ==================================================================
    # 7. Final console report
    # ==================================================================
    elapsed = time.time() - t_start

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"  Total time:      {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Config:          PCA={PCA_DIM}, perp={TSNE_PERPLEXITY}, "
          f"mcs={PRIMARY_MCS}, ms={best_ms}")
    print(f"  Clusters:        {best_detail['n_clusters']}")
    print(f"  Noise:           {best_detail['noise_pct']:.1f}%")
    print(f"  Composite score: {best_score:.4f}")
    print(f"  Transition rate: {best_tr['transition_rate']:.4f}")
    print(f"  Passes verify:   {best_detail['passes_verification']}")
    print()

    # Print per-cluster summary
    global_mean = features.mean(axis=0)
    global_std = features.std(axis=0)
    global_std[global_std < 1e-8] = 1.0

    for label in sorted(stats.keys()):
        s = stats[label]
        cluster_mean = s["means"]
        d_vals = (cluster_mean - global_mean) / global_std
        top_idx = np.argsort(np.abs(d_vals))[::-1]

        print(f"  Cluster {label} (n={s['size']}, "
              f"{s['size']/N*100:.1f}%):")
        for rank in range(min(5, len(FEATURE_NAMES))):
            idx = top_idx[rank]
            d = d_vals[idx]
            print(f"    {rank+1}. {FEATURE_NAMES[idx]:25s} d={d:+.2f}")
        print()

    # MCS sensitivity table
    print(f"\nMCS Sensitivity:")
    print(f"  {'MCS':>6} | {'ms':>4} | {'#Cl':>4} | {'Noise%':>7} | "
          f"{'Score':>7} | {'TransRate':>9} | {'Pass':>5}")
    print(f"  {'-'*65}")
    for mcs_val in sorted(mcs_table.keys()):
        info = mcs_table[mcs_val]
        print(f"  {mcs_val:6d} | {info['ms']:4d} | {info['n_clusters']:4d} | "
              f"{info['noise_pct']:6.1f}% | {info['score']:7.3f} | "
              f"{info['transition_rate']:9.4f} | "
              f"{'YES' if info['passes'] else 'no':>5}")

    print(f"\n  All outputs in: {OUTPUT_DIR}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
