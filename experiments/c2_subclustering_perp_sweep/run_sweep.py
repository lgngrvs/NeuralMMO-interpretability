#!/usr/bin/env python
"""Perplexity sweep for C2 sub-clustering (yaofeng_200M).

Runs T-SNE 2D at perplexities [40, 50, 60, 70] on the C2 (steady-state)
cluster, then applies HDBSCAN grid search to find sub-clusters at each
perplexity.  Generates per-perplexity outputs and a comparison overview.

Usage:
    uv run python experiments/c2_subclustering_perp_sweep/run_sweep.py
"""

import gc
import os
import sys

import numpy as np

# Ensure scripts/ is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from analyze_activations import (
    FEATURE_NAMES,
    compute_features,
    load_data,
    run_hdbscan,
    run_tsne,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "c2_subclustering_perp_sweep")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PERPLEXITIES = [40, 50, 60, 70]
MCS_VALUES = [50, 100, 200]
MS_VALUES = [5, 10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_best_hdbscan(embedding, mcs_values, ms_values):
    """Run HDBSCAN grid search and return the best (labels, config_dict).

    Best = most clusters with <40% noise; ties broken by most clusters,
    then lowest noise.
    """
    results = []
    for mcs in mcs_values:
        for ms in ms_values:
            if ms > mcs:
                continue
            sub_labels, _ = run_hdbscan(embedding, min_cluster_size=mcs, min_samples=ms)
            n_clusters = len(set(sub_labels) - {-1})
            noise_frac = (sub_labels == -1).sum() / len(sub_labels)
            results.append({
                "mcs": mcs,
                "ms": ms,
                "n_clusters": n_clusters,
                "noise_frac": noise_frac,
                "labels": sub_labels,
            })
            print(f"      mcs={mcs}, ms={ms}: {n_clusters} clusters, {noise_frac:.1%} noise")

    # Filter: >=2 clusters, <40% noise
    passing = [r for r in results if r["n_clusters"] >= 2 and r["noise_frac"] < 0.40]
    if not passing:
        passing = [r for r in results if r["n_clusters"] >= 2 and r["noise_frac"] < 0.50]
    if not passing:
        passing = [r for r in results if r["n_clusters"] >= 2]
    if not passing:
        # Fall back to config with most clusters
        passing = sorted(results, key=lambda r: r["n_clusters"], reverse=True)

    # Sort: most clusters first, then lowest noise
    passing.sort(key=lambda r: (-r["n_clusters"], r["noise_frac"]))
    best = passing[0]
    return best["labels"], best


def plot_tsne_scatter(embedding, labels, title, output_path):
    """2D scatter colored by sub-cluster."""
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    noise_pct = (labels == -1).sum() / len(labels) * 100

    if n_clusters <= 10:
        cmap = plt.colormaps.get_cmap("tab10")
    else:
        cmap = plt.colormaps.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Noise first (underneath)
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            embedding[noise_mask, 0], embedding[noise_mask, 1],
            c="lightgray", s=2, alpha=0.3, label=f"noise (n={noise_mask.sum()})",
            rasterized=True,
        )

    for i, cl in enumerate(cluster_ids):
        mask = labels == cl
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[cmap(i % 20)], s=4, alpha=0.5,
            label=f"C2-{cl} (n={mask.sum()})", rasterized=True,
        )

    ax.set_xlabel("T-SNE 1", fontsize=10)
    ax.set_ylabel("T-SNE 2", fontsize=10)
    ax.set_title(f"{title}\n{n_clusters} sub-clusters, {noise_pct:.1f}% noise", fontsize=12)
    ax.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.9, ncol=2)
    ax.grid(alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def compute_cohens_d(c2_feat, sub_labels):
    """Compute Cohen's d matrix: (n_clusters, n_features)."""
    cluster_ids = sorted(set(sub_labels) - {-1})
    global_mean = c2_feat.mean(0)
    global_std = c2_feat.std(0) + 1e-8
    d_matrix = np.zeros((len(cluster_ids), c2_feat.shape[1]))
    cluster_sizes = {}
    for i, cid in enumerate(cluster_ids):
        cmask = sub_labels == cid
        cluster_sizes[cid] = cmask.sum()
        d_matrix[i] = (c2_feat[cmask].mean(0) - global_mean) / global_std
    return d_matrix, cluster_ids, cluster_sizes


def plot_feature_heatmap(d_matrix, cluster_ids, cluster_sizes, output_path, title):
    """Cohen's d heatmap (RdBu_r, annotate |d|>0.5)."""
    n_feats = len(FEATURE_NAMES)
    n_cls = len(cluster_ids)

    fig, ax = plt.subplots(
        figsize=(max(14, n_feats * 0.7), max(4, n_cls * 0.5))
    )
    im = ax.imshow(d_matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(
        [f"C2-{c} (n={cluster_sizes[c]})" for c in cluster_ids], fontsize=8
    )
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="Cohen's d")

    for i in range(n_cls):
        for j in range(n_feats):
            val = d_matrix[i, j]
            if abs(val) > 0.5:
                ax.text(
                    j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(val) > 1.5 else "black",
                )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def write_summary(sub_labels, d_matrix, cluster_ids, cluster_sizes,
                   perp, best_config, output_path):
    """Write summary.txt with per-sub-cluster details."""
    n_clusters = len(cluster_ids)
    noise_pct = (sub_labels == -1).sum() / len(sub_labels) * 100
    total = len(sub_labels)

    with open(output_path, "w") as f:
        f.write(f"C2 Sub-clustering — Perplexity {perp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total C2 points: {total}\n")
        f.write(f"Sub-clusters found: {n_clusters}\n")
        f.write(f"Noise points: {(sub_labels == -1).sum()} ({noise_pct:.1f}%)\n")
        f.write(f"HDBSCAN params: min_cluster_size={best_config['mcs']}, "
                f"min_samples={best_config['ms']}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("Per-sub-cluster breakdown:\n\n")

        for i, cid in enumerate(cluster_ids):
            size = cluster_sizes[cid]
            pct = size / total * 100
            f.write(f"  Sub-cluster C2-{cid}  (n={size}, {pct:.1f}%)\n")

            # Top 3 features by |d|
            abs_d = np.abs(d_matrix[i])
            top_idx = np.argsort(abs_d)[::-1][:3]
            f.write(f"    Top features by |Cohen's d|:\n")
            for j in top_idx:
                f.write(f"      {FEATURE_NAMES[j]:30s}  d = {d_matrix[i, j]:+.2f}\n")
            f.write("\n")

    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("C2 Sub-clustering Perplexity Sweep — yaofeng_200M")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data and extract C2 subset
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading data (subsample=5)...")
    records, _ = load_data(
        os.path.join(PROJECT_ROOT, "activation_data", "yaofeng_200M"),
        subsample_rate=5,
    )
    activations, features, metadata = compute_features(records)
    del records
    gc.collect()

    N_total = len(activations)
    print(f"Total points: {N_total}")

    # Load original cluster labels
    labels_path = os.path.join(
        PROJECT_ROOT,
        "experiments", "T-SNE-3d-sweep-yaofeng_200M", "config_1",
        "cluster_labels.npy",
    )
    labels = np.load(labels_path)
    assert len(labels) == N_total, (
        f"Label count mismatch: {len(labels)} vs {N_total} activations"
    )

    c2_mask = labels == 2
    c2_act = activations[c2_mask]
    c2_feat = features[c2_mask]
    N_c2 = len(c2_act)
    print(f"C2 points: {N_c2} ({N_c2 / N_total * 100:.1f}% of total)")

    # Free full arrays
    del activations, features, metadata, labels
    gc.collect()

    # ------------------------------------------------------------------
    # 2. PCA reduction on C2
    # ------------------------------------------------------------------
    print("\n[Step 2] PCA on C2 activations...")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=30, random_state=42)
    c2_pca = pca.fit_transform(c2_act)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA 30D: {explained:.1%} variance explained")

    del c2_act
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Sweep perplexities
    # ------------------------------------------------------------------
    # Store results for comparison figure
    sweep_results = {}  # perp -> {embedding, labels, n_clusters, noise_pct, best_config}

    for perp in PERPLEXITIES:
        print(f"\n{'=' * 60}")
        print(f"  Perplexity = {perp}")
        print(f"{'=' * 60}")

        perp_dir = os.path.join(OUTPUT_DIR, f"perp_{perp}")
        os.makedirs(perp_dir, exist_ok=True)

        # T-SNE 2D
        print(f"  [a] Running T-SNE 2D (perplexity={perp})...")
        embedding, _ = run_tsne(c2_pca, perplexity=perp, n_components=2, random_state=42)

        # HDBSCAN grid search
        print(f"  [b] HDBSCAN grid search...")
        sub_labels, best_config = pick_best_hdbscan(embedding, MCS_VALUES, MS_VALUES)

        n_clusters = best_config["n_clusters"]
        noise_pct = best_config["noise_frac"] * 100

        print(f"  Best: mcs={best_config['mcs']}, ms={best_config['ms']} "
              f"-> {n_clusters} clusters, {noise_pct:.1f}% noise")

        # Cohen's d
        d_matrix, cluster_ids, cluster_sizes = compute_cohens_d(c2_feat, sub_labels)

        # Save plots + summary
        plot_tsne_scatter(
            embedding, sub_labels,
            f"C2 Sub-clusters (perp={perp})",
            os.path.join(perp_dir, "tsne_scatter.png"),
        )
        plot_feature_heatmap(
            d_matrix, cluster_ids, cluster_sizes,
            os.path.join(perp_dir, "cluster_features.png"),
            f"Cohen's d: C2 sub-clusters vs C2 baseline (perp={perp})",
        )
        write_summary(
            sub_labels, d_matrix, cluster_ids, cluster_sizes,
            perp, best_config,
            os.path.join(perp_dir, "summary.txt"),
        )

        sweep_results[perp] = {
            "embedding": embedding,
            "labels": sub_labels,
            "n_clusters": n_clusters,
            "noise_pct": noise_pct,
            "best_config": best_config,
        }

    # ------------------------------------------------------------------
    # 4. Comparison figure (1x4 grid)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Generating comparison figure...")
    print(f"{'=' * 60}")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    for ax_idx, perp in enumerate(PERPLEXITIES):
        ax = axes[ax_idx]
        res = sweep_results[perp]
        emb = res["embedding"]
        lbls = res["labels"]
        cluster_ids = sorted(set(lbls) - {-1})
        n_cls = len(cluster_ids)
        noise_pct = res["noise_pct"]

        if n_cls <= 10:
            cmap = plt.colormaps.get_cmap("tab10")
        else:
            cmap = plt.colormaps.get_cmap("tab20")

        # Noise
        noise_mask = lbls == -1
        if noise_mask.any():
            ax.scatter(
                emb[noise_mask, 0], emb[noise_mask, 1],
                c="lightgray", s=1, alpha=0.2, rasterized=True,
            )

        for i, cl in enumerate(cluster_ids):
            mask = lbls == cl
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=[cmap(i % 20)], s=2, alpha=0.4, rasterized=True,
            )

        ax.set_title(
            f"perp={perp}\n{n_cls} clusters, {noise_pct:.1f}% noise",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("T-SNE 1", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("T-SNE 2", fontsize=9)
        ax.grid(alpha=0.15)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "C2 Sub-clustering: Perplexity Sweep [40, 50, 60, 70]",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, "comparison.png")
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {comparison_path}")

    # ------------------------------------------------------------------
    # 5. Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Perp':>6}  {'Clusters':>8}  {'Noise%':>7}  {'HDBSCAN config':>20}")
    print("-" * 50)
    for perp in PERPLEXITIES:
        res = sweep_results[perp]
        cfg = res["best_config"]
        print(f"{perp:>6}  {res['n_clusters']:>8}  {res['noise_pct']:>6.1f}%  "
              f"mcs={cfg['mcs']}, ms={cfg['ms']}")

    # Cleanup
    del sweep_results, c2_feat, c2_pca
    gc.collect()

    print("\nDone!")


if __name__ == "__main__":
    main()
