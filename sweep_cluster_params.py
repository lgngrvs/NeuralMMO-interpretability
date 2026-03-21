#!/usr/bin/env python
"""Sweep clustering hyperparameters to find behavioral mode clusters.

Three strategies:
  1. PCA → HDBSCAN (high-D density clustering)
  2. PCA → UMAP 2D → HDBSCAN (cluster on UMAP embedding)
  3. PCA → K-means (forced k clusters, good for continuous manifolds)

Usage:
    uv run python sweep_cluster_params.py activation_data/baseline_10M/ --subsample 1
"""
import argparse
import csv
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import hdbscan as hdbscan_lib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from analyze_activations import (
    FEATURE_NAMES,
    compute_cluster_stats,
    compute_features,
    load_data,
    run_umap,
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def cluster_count_factor(n):
    """Peaked factor: 1.0 for 2-8 clusters, drops outside."""
    if n <= 1:
        return 0.0
    if 2 <= n <= 8:
        return 1.0
    if n == 9:
        return 0.7
    if n == 10:
        return 0.4
    return 0.1


def score_combo(labels, features, metadata, min_trustworthy, seed=42):
    """Score a clustering result. Returns (score, detail_dict)."""
    unique_labels = sorted(set(labels))
    has_noise = -1 in unique_labels
    if has_noise:
        unique_labels.remove(-1)

    N = len(labels)
    noise_ratio = (labels == -1).sum() / N if has_noise else 0.0

    # Filter out clusters smaller than min_trustworthy
    surviving = [l for l in unique_labels if (labels == l).sum() >= min_trustworthy]
    n_clusters = len(surviving)

    if n_clusters == 0:
        return 0.0, {
            "n_clusters_raw": len(unique_labels),
            "n_clusters": 0,
            "noise_pct": noise_ratio * 100,
            "mean_weighted_d": 0.0,
            "cluster_count_factor": 0.0,
            "entropy_factor": 0.0,
            "composite_score": 0.0,
            "passes_verification": False,
        }

    # Remap: non-surviving → noise, surviving → 0,1,2,...
    filtered_labels = labels.copy()
    for l in unique_labels:
        if l not in surviving:
            filtered_labels[filtered_labels == l] = -1
    # Renumber surviving clusters sequentially
    remap = {old: new for new, old in enumerate(sorted(surviving))}
    for old, new in remap.items():
        filtered_labels[labels == old] = new
    surviving = sorted(remap.values())

    stats = compute_cluster_stats(features, filtered_labels, metadata,
                                  n_baseline_samples=100, rng_seed=seed)

    total_assigned = sum(stats[c]["size"] for c in surviving)

    # A: Size-weighted mean of top-3 |Cohen's d|
    weighted_d = 0.0
    for c in surviving:
        s = stats[c]
        top3_idx = s["top_features"][:3]
        mean_top3_d = np.abs(s["cohens_d"][top3_idx]).mean()
        weighted_d += (s["size"] / total_assigned) * mean_top3_d

    # B: Cluster count factor
    ccf = cluster_count_factor(n_clusters)

    # C: Noise penalty
    noise_factor = 1.0 - noise_ratio

    # D: Entropy factor (size-weighted)
    weighted_step_ent = 0.0
    weighted_agent_ent = 0.0
    min_step_ent = 1.0
    min_agent_ent = 1.0
    for c in surviving:
        s = stats[c]
        w = s["size"] / total_assigned
        weighted_step_ent += w * s["step_entropy"]
        weighted_agent_ent += w * s["agent_entropy"]
        min_step_ent = min(min_step_ent, s["step_entropy"])
        min_agent_ent = min(min_agent_ent, s["agent_entropy"])

    entropy_factor = (weighted_step_ent + weighted_agent_ent) / 2.0
    composite = weighted_d * ccf * noise_factor * entropy_factor

    passes = (
        2 <= n_clusters <= 8
        and noise_ratio < 0.30
        and weighted_d > 0.8
        and min_step_ent > 0.4
        and min_agent_ent > 0.4
    )

    detail = {
        "n_clusters_raw": len(unique_labels) + (1 if has_noise else 0),
        "n_clusters": n_clusters,
        "noise_pct": noise_ratio * 100,
        "mean_weighted_d": weighted_d,
        "cluster_count_factor": ccf,
        "entropy_factor": entropy_factor,
        "composite_score": composite,
        "min_step_entropy": min_step_ent,
        "min_agent_entropy": min_agent_ent,
        "passes_verification": passes,
        "stats": stats,
    }
    return composite, detail


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_best(embedding, features, labels, stats, output_dir):
    """Generate UMAP scatter and feature heatmap for the best combo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                   c="lightgray", s=1, alpha=0.3, label="noise")
    cluster_labels = sorted(set(labels) - {-1})
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(cluster_labels), 1))
    for i, label in enumerate(cluster_labels):
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], s=3, alpha=0.5, label=f"C{label} (n={(labels==label).sum()})")
    ax.set_title("Best Sweep Combo — UMAP Colored by Cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if len(cluster_labels) <= 20:
        ax.legend(markerscale=4, fontsize=8)
    scatter_path = os.path.join(output_dir, "best_umap.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {scatter_path}")

    # Feature heatmap
    cluster_ids = sorted(stats.keys())
    if not cluster_ids:
        return
    n_feats = len(FEATURE_NAMES)
    d_matrix = np.array([stats[c]["cohens_d"] for c in cluster_ids])

    fig, ax = plt.subplots(figsize=(max(12, n_feats * 0.8), max(4, len(cluster_ids) * 0.6)))
    im = ax.imshow(d_matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"C{c} (n={stats[c]['size']})" for c in cluster_ids], fontsize=8)
    ax.set_title("Best Sweep Combo — Cohen's d per Feature per Cluster")
    fig.colorbar(im, ax=ax, label="Cohen's d")

    for i in range(len(cluster_ids)):
        for j in range(n_feats):
            val = d_matrix[i, j]
            if abs(val) > 0.5:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 1.5 else "black")

    feat_path = os.path.join(output_dir, "best_features.png")
    fig.savefig(feat_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {feat_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sweep clustering hyperparameters for behavioral mode clustering")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation data")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Keep every Nth record per trajectory (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: auto-named under sweep_results/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    records, total_datapoints = load_data(args.data_dir, subsample_rate=args.subsample)
    activations, features, metadata = compute_features(records)
    N = len(activations)
    print(f"Activations: {activations.shape}, Features: {features.shape}")

    min_trustworthy = max(20, int(0.02 * N))
    print(f"Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # --- Pre-compute PCA reductions ---
    PCA_DIMS = [10, 20, 30, 50]
    print("Pre-computing PCA reductions...", flush=True)
    pca_cache = {}
    for dims in PCA_DIMS:
        d = min(dims, activations.shape[1], activations.shape[0])
        pca = PCA(n_components=d, random_state=args.seed)
        pca_cache[dims] = pca.fit_transform(activations)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA {d}D: {explained:.1%} variance explained")

    # --- Pre-compute UMAP embeddings for a few PCA dims ---
    UMAP_PCA_DIMS = [20, 30]
    UMAP_NEIGHBORS = [15, 30]
    UMAP_MIN_DISTS = [0.0, 0.1]
    print("Pre-computing UMAP embeddings...", flush=True)
    umap_cache = {}
    for pca_d in UMAP_PCA_DIMS:
        for nn in UMAP_NEIGHBORS:
            for md in UMAP_MIN_DISTS:
                key = (pca_d, nn, md)
                print(f"  UMAP: PCA={pca_d}D, n_neighbors={nn}, min_dist={md}...", flush=True)
                emb, _ = run_umap(pca_cache[pca_d], n_neighbors=nn,
                                  min_dist=md, random_state=args.seed)
                umap_cache[key] = emb
    print()

    results = []

    # ===================================================================
    # Strategy 1: PCA → HDBSCAN (high-D)
    # ===================================================================
    hdbscan_mcs = [50, 100, 200, 400]
    hdbscan_ms = [5, 10, 25]
    grid1 = list(product(PCA_DIMS, hdbscan_mcs, hdbscan_ms))
    print(f"Strategy 1: PCA → HDBSCAN ({len(grid1)} combos)")
    for pca_dims, mcs, ms in tqdm(grid1, desc="S1: PCA+HDBSCAN"):
        if ms > mcs:
            continue
        act_reduced = pca_cache[pca_dims]
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        labels = clusterer.fit_predict(act_reduced)
        score, detail = score_combo(labels, features, metadata, min_trustworthy, args.seed)
        row = {
            "strategy": "pca_hdbscan",
            "pca_dims": pca_dims, "umap_neighbors": "", "umap_min_dist": "",
            "algo": "hdbscan", "k_or_mcs": mcs, "min_samples": ms,
            **{k: v for k, v in detail.items() if k != "stats"},
        }
        results.append((score, row, detail, labels))

    # ===================================================================
    # Strategy 2: PCA → UMAP 2D → HDBSCAN
    # ===================================================================
    umap_mcs = [50, 100, 200, 400]
    umap_ms = [5, 10, 25]
    grid2 = list(product(UMAP_PCA_DIMS, UMAP_NEIGHBORS, UMAP_MIN_DISTS,
                         umap_mcs, umap_ms))
    print(f"Strategy 2: PCA → UMAP → HDBSCAN ({len(grid2)} combos)")
    for pca_d, nn, md, mcs, ms in tqdm(grid2, desc="S2: UMAP+HDBSCAN"):
        if ms > mcs:
            continue
        emb = umap_cache[(pca_d, nn, md)]
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        labels = clusterer.fit_predict(emb)
        score, detail = score_combo(labels, features, metadata, min_trustworthy, args.seed)
        row = {
            "strategy": "umap_hdbscan",
            "pca_dims": pca_d, "umap_neighbors": nn, "umap_min_dist": md,
            "algo": "hdbscan", "k_or_mcs": mcs, "min_samples": ms,
            **{k: v for k, v in detail.items() if k != "stats"},
        }
        results.append((score, row, detail, labels))

    # ===================================================================
    # Strategy 3: PCA → K-means
    # ===================================================================
    K_VALUES = [2, 3, 4, 5, 6, 7, 8]
    grid3 = list(product(PCA_DIMS, K_VALUES))
    print(f"Strategy 3: PCA → K-means ({len(grid3)} combos)")
    for pca_dims, k in tqdm(grid3, desc="S3: PCA+KMeans"):
        act_reduced = pca_cache[pca_dims]
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        labels = km.fit_predict(act_reduced)
        score, detail = score_combo(labels, features, metadata, min_trustworthy, args.seed)
        row = {
            "strategy": "pca_kmeans",
            "pca_dims": pca_dims, "umap_neighbors": "", "umap_min_dist": "",
            "algo": "kmeans", "k_or_mcs": k, "min_samples": "",
            **{k_: v for k_, v in detail.items() if k_ != "stats"},
        }
        results.append((score, row, detail, labels))

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)

    # Output — per-policy subdirectory so runs don't clobber each other
    policy_name = Path(args.data_dir).name
    output_dir = os.path.join("sweep_results", policy_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"sweep_{policy_name}_{ts}.csv")
    else:
        csv_path = args.output

    fieldnames = [
        "strategy", "pca_dims", "umap_neighbors", "umap_min_dist",
        "algo", "k_or_mcs", "min_samples",
        "n_clusters_raw", "n_clusters", "noise_pct",
        "mean_weighted_d", "cluster_count_factor", "entropy_factor",
        "min_step_entropy", "min_agent_entropy",
        "composite_score", "passes_verification",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row, _, _ in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nWrote {csv_path} ({len(results)} combos)")

    # Console: top 20
    print(f"\n{'='*120}")
    print(f"{'Strategy':>15} {'PCA':>4} {'UNN':>4} {'UMD':>5} {'Algo':>8} "
          f"{'K/MCS':>6} {'MS':>4} {'#Cl':>4} {'Noise%':>7} "
          f"{'WtdD':>6} {'Ent':>5} {'Score':>7} {'Pass':>5}")
    print(f"{'-'*120}")
    for i, (score, row, detail, _) in enumerate(results[:20]):
        un = row['umap_neighbors'] if row['umap_neighbors'] != '' else '-'
        ud = row['umap_min_dist'] if row['umap_min_dist'] != '' else '-'
        ms = row['min_samples'] if row['min_samples'] != '' else '-'
        print(f"{row['strategy']:>15} {row['pca_dims']:4} {str(un):>4} {str(ud):>5} "
              f"{row['algo']:>8} {row['k_or_mcs']:>6} {str(ms):>4} "
              f"{row['n_clusters']:4} {row['noise_pct']:6.1f}% "
              f"{row['mean_weighted_d']:6.2f} {row['entropy_factor']:5.2f} "
              f"{row['composite_score']:7.3f} {'YES' if row['passes_verification'] else 'no':>5}")

    # Detailed output for best combo
    best_score, best_row, best_detail, best_labels = results[0]
    best_stats = best_detail["stats"]

    print(f"\n{'='*120}")
    print(f"BEST: {best_row['strategy']} — "
          f"PCA={best_row['pca_dims']}, algo={best_row['algo']}, "
          f"k_or_mcs={best_row['k_or_mcs']}, min_samples={best_row['min_samples']}")
    if best_row['umap_neighbors'] != '':
        print(f"  UMAP: n_neighbors={best_row['umap_neighbors']}, "
              f"min_dist={best_row['umap_min_dist']}")
    print(f"Score: {best_score:.3f}  "
          f"({best_row['n_clusters']} clusters, {best_row['noise_pct']:.1f}% noise)")
    print(f"{'='*120}")

    for label in sorted(best_stats.keys()):
        s = best_stats[label]
        print(f"\n  Cluster {label} (n={s['size']}, "
              f"{s['size']/N*100:.1f}%):")
        print(f"    Step entropy: {s['step_entropy']:.3f}  "
              f"Agent entropy: {s['agent_entropy']:.3f}")
        print(f"    Top features:")
        for rank, idx in enumerate(s["top_features"][:5]):
            d = s["cohens_d"][idx]
            print(f"      {rank+1}. {FEATURE_NAMES[idx]:25s} d={d:+.2f}")

    # Visualize best combo
    # Use a UMAP embedding for visualization
    # If best used UMAP, reuse that embedding; otherwise compute one
    if best_row['strategy'] == 'umap_hdbscan' and best_row['umap_neighbors'] != '':
        key = (best_row['pca_dims'], best_row['umap_neighbors'],
               best_row['umap_min_dist'])
        best_embedding = umap_cache[key]
    else:
        # Use default UMAP on PCA 30D for viz
        if (30, 15, 0.1) in umap_cache:
            best_embedding = umap_cache[(30, 15, 0.1)]
        else:
            print("\nRunning UMAP for visualization...", flush=True)
            best_embedding, _ = run_umap(pca_cache[30], n_neighbors=15,
                                         random_state=args.seed)

    plot_best(best_embedding, features, best_labels, best_stats, output_dir)


if __name__ == "__main__":
    main()
