#!/usr/bin/env python
"""Sub-cluster analysis of the dominant C2 cluster from yaofeng_200M T-SNE 3D.

Extracts C2 points (steady-state cluster), runs PCA -> T-SNE -> HDBSCAN to find
hidden sub-structure, and generates visualizations + summary.

Usage:
    uv run python experiments/c2_subclustering/analyze_c2_subclusters.py
"""

import gc
import os
import sys

import numpy as np

# Ensure project root is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

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

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "c2_subclustering")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIEWING_ANGLES = [
    (30, 45),
    (30, 135),
    (30, 225),
    (60, 0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_2d_subclusters(embedding, labels, title, output_path):
    """2D scatter colored by sub-cluster."""
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Noise first
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
            c=[cmap(i)], s=4, alpha=0.5,
            label=f"C2-{cl} (n={mask.sum()})", rasterized=True,
        )

    ax.set_xlabel("T-SNE 1", fontsize=10)
    ax.set_ylabel("T-SNE 2", fontsize=10)
    ax.set_title(f"{title}\n{n_clusters} sub-clusters, {noise_pct:.1f}% noise", fontsize=12)
    ax.legend(markerscale=3, fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def plot_3d_subclusters(embedding, labels, title, output_path):
    """2x2 grid of 3D scatter plots from four viewing angles."""
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

    fig = plt.figure(figsize=(16, 14))

    for panel_idx, (elev, azim) in enumerate(VIEWING_ANGLES, start=1):
        ax = fig.add_subplot(2, 2, panel_idx, projection="3d")

        # Noise first
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
                label=f"C2-{cl} (n={mask.sum()})", rasterized=True,
            )

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("T-SNE 1", fontsize=8)
        ax.set_ylabel("T-SNE 2", fontsize=8)
        ax.set_zlabel("T-SNE 3", fontsize=8)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{title}\n{n_clusters} sub-clusters, {noise_pct:.1f}% noise",
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


def plot_feature_heatmap(cohens_d_matrix, cluster_ids, cluster_sizes, output_path,
                         title="Cohen's d: C2 sub-clusters vs C2 baseline"):
    """Cohen's d heatmap (RdBu_r, annotate |d|>0.5)."""
    n_feats = len(FEATURE_NAMES)

    fig, ax = plt.subplots(
        figsize=(max(12, n_feats * 0.8), max(4, len(cluster_ids) * 0.6))
    )
    im = ax.imshow(cohens_d_matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels(
        [f"C2-{c} (n={cluster_sizes[c]})" for c in cluster_ids], fontsize=8
    )
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="Cohen's d")

    for i in range(len(cluster_ids)):
        for j in range(n_feats):
            val = cohens_d_matrix[i, j]
            if abs(val) > 0.5:
                ax.text(
                    j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(val) > 1.5 else "black",
                )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("C2 Sub-clustering Analysis — yaofeng_200M")
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
    c2_meta = {k: v[c2_mask] for k, v in metadata.items()}
    N_c2 = len(c2_act)
    print(f"C2 points: {N_c2} ({N_c2 / N_total * 100:.1f}% of total)")

    # Free the full arrays
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
    # 3. Grid search: T-SNE 2D + HDBSCAN
    # ------------------------------------------------------------------
    print("\n[Step 3] T-SNE 2D + HDBSCAN grid search on C2...")

    perplexities = [30, 50]
    mcs_values = [50, 100, 200]
    ms_values = [5, 10]

    # Pre-compute T-SNE embeddings
    tsne_2d_cache = {}
    for perp in perplexities:
        print(f"  T-SNE 2D: perplexity={perp}...", flush=True)
        emb, _ = run_tsne(c2_pca, perplexity=perp, n_components=2, random_state=42)
        tsne_2d_cache[perp] = emb

    # Search over HDBSCAN params
    results = []
    for perp in perplexities:
        emb = tsne_2d_cache[perp]
        for mcs in mcs_values:
            for ms in ms_values:
                if ms > mcs:
                    continue
                sub_labels, _ = run_hdbscan(emb, min_cluster_size=mcs, min_samples=ms)
                n_clusters = len(set(sub_labels) - {-1})
                noise_frac = (sub_labels == -1).sum() / len(sub_labels)

                results.append({
                    "perp": perp,
                    "mcs": mcs,
                    "ms": ms,
                    "n_clusters": n_clusters,
                    "noise_frac": noise_frac,
                    "labels": sub_labels,
                    "embedding": emb,
                })
                print(
                    f"    perp={perp}, mcs={mcs}, ms={ms}: "
                    f"{n_clusters} clusters, {noise_frac:.1%} noise"
                )

    # Filter: 2+ clusters, <40% noise
    passing = [r for r in results if r["n_clusters"] >= 2 and r["noise_frac"] < 0.40]

    print(f"\n  Passing configs (>=2 clusters, <40% noise): {len(passing)}")

    if not passing:
        # If nothing passes strict criteria, relax noise to 50%
        passing = [r for r in results if r["n_clusters"] >= 2 and r["noise_frac"] < 0.50]
        print(f"  Relaxed to <50% noise: {len(passing)}")

    if not passing:
        # Fall back to best available with multiple clusters
        passing = [r for r in results if r["n_clusters"] >= 2]
        print(f"  Any with 2+ clusters: {len(passing)}")

    if not passing:
        print("\n  ERROR: No configuration produced 2+ clusters. C2 appears homogeneous.")
        # Still write summary
        with open(os.path.join(OUTPUT_DIR, "c2_subcluster_summary.txt"), "w") as f:
            f.write("C2 Sub-clustering: NO SUB-CLUSTERS FOUND\n\n")
            f.write("All HDBSCAN configurations produced 0 or 1 cluster.\n")
            f.write("C2 appears to be a homogeneous blob with no hidden structure.\n\n")
            for r in results:
                f.write(
                    f"  perp={r['perp']}, mcs={r['mcs']}, ms={r['ms']}: "
                    f"{r['n_clusters']} clusters, {r['noise_frac']:.1%} noise\n"
                )
        return

    # Pick best: most clusters first, then least noise
    passing.sort(key=lambda r: (-r["n_clusters"], r["noise_frac"]))
    best = passing[0]
    print(
        f"\n  Best config: perp={best['perp']}, mcs={best['mcs']}, ms={best['ms']}"
        f" -> {best['n_clusters']} clusters, {best['noise_frac']:.1%} noise"
    )

    # ------------------------------------------------------------------
    # 4. Compute Cohen's d for sub-clusters vs C2 baseline
    # ------------------------------------------------------------------
    print("\n[Step 4] Computing Cohen's d for sub-clusters...")

    sub_labels = best["labels"]
    sub_emb_2d = best["embedding"]
    cluster_ids = sorted(set(sub_labels) - {-1})
    n_feats = len(FEATURE_NAMES)

    # C2 baseline stats
    c2_mean = c2_feat.mean(axis=0)
    c2_std = c2_feat.std(axis=0)

    cohens_d_matrix = np.zeros((len(cluster_ids), n_feats))
    cluster_sizes = {}
    cluster_stats = {}

    for i, cl in enumerate(cluster_ids):
        cl_mask = sub_labels == cl
        cl_feats = c2_feat[cl_mask]
        cluster_sizes[cl] = cl_mask.sum()
        cohens_d = (cl_feats.mean(axis=0) - c2_mean) / (c2_std + 1e-8)
        cohens_d_matrix[i] = cohens_d

        # Top features by |d|
        top_idx = np.argsort(np.abs(cohens_d))[::-1][:5]
        cluster_stats[cl] = {
            "size": int(cl_mask.sum()),
            "cohens_d": cohens_d,
            "top_features": top_idx,
        }

    # ------------------------------------------------------------------
    # 5. Visualizations
    # ------------------------------------------------------------------
    print("\n[Step 5] Generating visualizations...")

    # 5a. 2D scatter
    plot_2d_subclusters(
        sub_emb_2d, sub_labels,
        f"C2 Sub-clusters (perp={best['perp']}, mcs={best['mcs']}, ms={best['ms']})",
        os.path.join(OUTPUT_DIR, "c2_tsne_subclusters.png"),
    )

    # 5b. Feature heatmap
    plot_feature_heatmap(
        cohens_d_matrix, cluster_ids, cluster_sizes,
        os.path.join(OUTPUT_DIR, "c2_subcluster_features.png"),
        title=(
            f"Cohen's d: C2 sub-clusters vs C2 baseline "
            f"(perp={best['perp']}, mcs={best['mcs']}, ms={best['ms']})"
        ),
    )

    # 5c. 3D T-SNE
    print("\n  Running T-SNE 3D for C2 sub-cluster visualization...")
    emb_3d, _ = run_tsne(
        c2_pca, perplexity=best["perp"], n_components=3, random_state=42
    )

    # Re-run HDBSCAN on 3D embedding with same params for consistent labeling
    sub_labels_3d, _ = run_hdbscan(
        emb_3d, min_cluster_size=best["mcs"], min_samples=best["ms"]
    )

    plot_3d_subclusters(
        emb_3d, sub_labels_3d,
        f"C2 Sub-clusters 3D (perp={best['perp']}, mcs={best['mcs']}, ms={best['ms']})",
        os.path.join(OUTPUT_DIR, "c2_tsne3d_subclusters.png"),
    )

    # ------------------------------------------------------------------
    # 6. Summary text
    # ------------------------------------------------------------------
    print("\n[Step 6] Writing summary...")

    summary_path = os.path.join(OUTPUT_DIR, "c2_subcluster_summary.txt")
    with open(summary_path, "w") as f:
        f.write("C2 Sub-clustering Analysis: yaofeng_200M\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: C2 from T-SNE-3d-sweep-yaofeng_200M/config_1\n")
        f.write(f"C2 size: {N_c2} points ({N_c2/N_total*100:.1f}% of total {N_total})\n\n")

        f.write("Grid search results (2D T-SNE + HDBSCAN):\n")
        f.write("-" * 60 + "\n")
        for r in sorted(results, key=lambda x: (-x["n_clusters"], x["noise_frac"])):
            flag = " <-- BEST" if (
                r["perp"] == best["perp"]
                and r["mcs"] == best["mcs"]
                and r["ms"] == best["ms"]
            ) else ""
            f.write(
                f"  perp={r['perp']:3d}, mcs={r['mcs']:3d}, ms={r['ms']:2d}: "
                f"{r['n_clusters']:2d} clusters, {r['noise_frac']:.1%} noise{flag}\n"
            )

        f.write(f"\nBest config: perp={best['perp']}, mcs={best['mcs']}, ms={best['ms']}\n")
        f.write(f"Sub-clusters found: {best['n_clusters']}\n")
        f.write(f"Noise: {best['noise_frac']:.1%}\n")

        f.write(f"\n{'=' * 60}\n")
        f.write("Sub-cluster profiles (Cohen's d vs C2 population):\n")
        f.write("=" * 60 + "\n")

        for cl in cluster_ids:
            s = cluster_stats[cl]
            f.write(f"\nSub-cluster C2-{cl} (n={s['size']}, "
                    f"{s['size']/N_c2*100:.1f}% of C2):\n")
            f.write("  Top distinguishing features:\n")
            for rank, idx in enumerate(s["top_features"]):
                d = s["cohens_d"][idx]
                f.write(f"    {rank+1}. {FEATURE_NAMES[idx]:25s} d={d:+.2f}\n")

        # Also note noise cluster
        n_noise = (sub_labels == -1).sum()
        f.write(f"\nNoise points: {n_noise} ({n_noise/N_c2*100:.1f}% of C2)\n")

        # 3D results
        n_clusters_3d = len(set(sub_labels_3d) - {-1})
        noise_frac_3d = (sub_labels_3d == -1).sum() / len(sub_labels_3d)
        f.write(f"\n{'=' * 60}\n")
        f.write("3D T-SNE results (same perplexity, HDBSCAN params):\n")
        f.write(f"  Sub-clusters: {n_clusters_3d}\n")
        f.write(f"  Noise: {noise_frac_3d:.1%}\n")

        # Overall assessment
        f.write(f"\n{'=' * 60}\n")
        f.write("Assessment:\n")
        f.write("=" * 60 + "\n")
        if best["n_clusters"] >= 3:
            f.write(
                "C2 contains meaningful sub-structure with multiple distinct\n"
                "sub-clusters that differ in interpretable behavioral features.\n"
            )
        elif best["n_clusters"] == 2:
            f.write(
                "C2 shows some internal structure (2 sub-clusters), suggesting\n"
                "a possible bifurcation within the steady-state regime.\n"
            )
        else:
            f.write(
                "C2 appears largely homogeneous, with only weak/marginal\n"
                "sub-structure detected.\n"
            )

    print(f"  Wrote {summary_path}")
    print(f"\nDone! All outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
