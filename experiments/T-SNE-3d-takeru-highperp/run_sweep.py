#!/usr/bin/env python
"""T-SNE 3D clustering sweep on takeru_200M with MORE data and higher perplexity.

Uses subsample=2 (instead of 5) and perplexities [40, 50, 60, 70, 80].
For each perplexity, sweeps HDBSCAN params and picks the best config.
Generates per-perplexity visualizations and a comparison figure.

Usage:
    uv run python experiments/T-SNE-3d-takeru-highperp/run_sweep.py
"""

import os
import sys
from itertools import product
from pathlib import Path

import hdbscan as hdbscan_lib
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from analyze_activations import (
    FEATURE_NAMES,
    compute_cluster_stats,
    compute_features,
    load_data,
    run_tsne,
)
from sweep_cluster_params import score_combo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- Configuration ----
DATA_DIR = os.path.join(REPO_ROOT, "activation_data", "takeru_200M")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUBSAMPLE_RATE = 2
PCA_DIMS = 30
SEED = 42
PERPLEXITIES = [40, 50, 60, 70, 80]
MCS_VALUES = [50, 100, 200, 400]
MS_VALUES = [5, 10, 25]

VIEWING_ANGLES = [
    (30, 45),
    (30, 135),
    (30, 225),
    (60, 0),
]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_3d_scatter(embedding, labels, config_str, output_path):
    """Save a 2x2 grid of 3D scatter plots from four viewing angles."""
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    cmap_name = "tab10" if n_clusters <= 10 else "tab20"
    cmap = plt.colormaps.get_cmap(cmap_name).resampled(max(n_clusters, 1))

    fig = plt.figure(figsize=(16, 14))

    for panel_idx, (elev, azim) in enumerate(VIEWING_ANGLES, start=1):
        ax = fig.add_subplot(2, 2, panel_idx, projection="3d")

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


def plot_feature_heatmap(stats, output_path, title="Cohen's d per Cluster per Feature"):
    """Save a Cohen's d heatmap (RdBu_r)."""
    cluster_ids = sorted(stats.keys())
    if not cluster_ids:
        return
    n_feats = len(FEATURE_NAMES)
    d_matrix = np.array([stats[c]["cohens_d"] for c in cluster_ids])

    fig, ax = plt.subplots(
        figsize=(max(12, n_feats * 0.8), max(4, len(cluster_ids) * 0.6))
    )
    im = ax.imshow(d_matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels(
        [f"C{c} (n={stats[c]['size']})" for c in cluster_ids], fontsize=8
    )
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="Cohen's d")

    for i in range(len(cluster_ids)):
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


def write_summary_txt(detail, perp, mcs, ms, output_path, N):
    """Write a human-readable summary for a single perplexity's best config."""
    stats = detail["stats"]
    with open(output_path, "w") as f:
        f.write(f"Strategy: T-SNE 3D + HDBSCAN (high perplexity sweep)\n")
        f.write(f"Policy: takeru_200M\n")
        f.write(f"Subsample rate: {SUBSAMPLE_RATE}\n")
        f.write(f"PCA dims: {PCA_DIMS}\n")
        f.write(f"T-SNE perplexity: {perp}\n")
        f.write(f"HDBSCAN min_cluster_size: {mcs}\n")
        f.write(f"HDBSCAN min_samples: {ms}\n")
        f.write(f"\nComposite score: {detail['composite_score']:.4f}\n")
        f.write(f"Passes verification: {detail['passes_verification']}\n")
        f.write(f"Clusters: {detail['n_clusters']} (raw: {detail['n_clusters_raw']})\n")
        f.write(f"Noise: {detail['noise_pct']:.1f}%\n")
        f.write(f"Mean weighted |d|: {detail['mean_weighted_d']:.3f}\n")
        f.write(f"Entropy factor: {detail['entropy_factor']:.3f}\n")
        f.write(f"Min step entropy: {detail.get('min_step_entropy', 'N/A')}\n")
        f.write(f"Min agent entropy: {detail.get('min_agent_entropy', 'N/A')}\n")
        f.write(f"\n{'='*80}\n")
        for label in sorted(stats.keys()):
            s = stats[label]
            f.write(f"\nCluster {label} (n={s['size']}, {s['size']/N*100:.1f}%):\n")
            f.write(f"  Step entropy: {s['step_entropy']:.3f}\n")
            f.write(f"  Agent entropy: {s['agent_entropy']:.3f}\n")
            f.write(f"  Top features:\n")
            for rank, idx in enumerate(s["top_features"][:5]):
                d = s["cohens_d"][idx]
                f.write(f"    {rank+1}. {FEATURE_NAMES[idx]:25s} d={d:+.2f}\n")
    print(f"  Wrote {output_path}")


def plot_comparison(perp_results, output_path):
    """1x5 grid showing one 3D scatter per perplexity (elev=30, azim=45)."""
    n_perps = len(perp_results)
    fig = plt.figure(figsize=(5 * n_perps, 6))

    for idx, (perp, emb, labels, score, detail) in enumerate(perp_results):
        ax = fig.add_subplot(1, n_perps, idx + 1, projection="3d")

        cluster_ids = sorted(set(labels) - {-1})
        n_clusters = len(cluster_ids)
        noise_pct = (labels == -1).sum() / len(labels) * 100
        cmap_name = "tab10" if n_clusters <= 10 else "tab20"
        cmap = plt.colormaps.get_cmap(cmap_name).resampled(max(n_clusters, 1))

        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(
                emb[noise_mask, 0], emb[noise_mask, 1], emb[noise_mask, 2],
                c="lightgray", s=1, alpha=0.3, rasterized=True,
            )

        for i, cl in enumerate(cluster_ids):
            mask = labels == cl
            ax.scatter(
                emb[mask, 0], emb[mask, 1], emb[mask, 2],
                c=[cmap(i)], s=2, alpha=0.5,
                label=f"C{cl}", rasterized=True,
            )

        ax.view_init(elev=30, azim=45)
        ax.set_xlabel("T-SNE 1", fontsize=7)
        ax.set_ylabel("T-SNE 2", fontsize=7)
        ax.set_zlabel("T-SNE 3", fontsize=7)
        ax.set_title(
            f"perp={perp}\n{n_clusters} cl, {noise_pct:.0f}% noise\nscore={score:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=6)

    fig.suptitle(
        "takeru_200M: T-SNE 3D High-Perplexity Comparison (subsample=2)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("T-SNE 3D High-Perplexity Sweep: takeru_200M")
    print(f"Subsample rate: {SUBSAMPLE_RATE}, PCA dims: {PCA_DIMS}")
    print(f"Perplexities: {PERPLEXITIES}")
    print(f"HDBSCAN grid: mcs={MCS_VALUES}, ms={MS_VALUES}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...", flush=True)
    records, total_datapoints = load_data(DATA_DIR, subsample_rate=SUBSAMPLE_RATE)
    activations, features, metadata = compute_features(records)
    N = len(activations)
    print(f"Data: {activations.shape} ({N} points)")

    min_trustworthy = max(20, int(0.02 * N))
    print(f"Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # ------------------------------------------------------------------
    # 2. PCA reduction
    # ------------------------------------------------------------------
    print(f"\n[2/5] PCA to {PCA_DIMS}D...", flush=True)
    pca = PCA(n_components=PCA_DIMS, random_state=SEED)
    act_pca = pca.fit_transform(activations)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {PCA_DIMS}D: {explained:.1%} variance explained")

    # ------------------------------------------------------------------
    # 3. Run T-SNE 3D at each perplexity + HDBSCAN sweep
    # ------------------------------------------------------------------
    print(f"\n[3/5] T-SNE 3D + HDBSCAN sweep...", flush=True)

    # Collect results for comparison plot
    comparison_data = []  # (perp, embedding, labels, score, detail)
    all_sweep_results = []

    for perp in PERPLEXITIES:
        print(f"\n--- Perplexity = {perp} ---")
        perp_dir = os.path.join(OUTPUT_DIR, f"perp_{perp}")
        os.makedirs(perp_dir, exist_ok=True)

        # Run T-SNE 3D
        print(f"  Running T-SNE 3D (perp={perp})...", flush=True)
        emb, _ = run_tsne(act_pca, perplexity=perp, n_components=3, random_state=SEED)
        print(f"  T-SNE embedding shape: {emb.shape}")

        # HDBSCAN grid search
        best_score = -1
        best_labels = None
        best_detail = None
        best_mcs = None
        best_ms = None

        grid = list(product(MCS_VALUES, MS_VALUES))
        for mcs, ms in tqdm(grid, desc=f"  HDBSCAN (perp={perp})", leave=False):
            if ms > mcs:
                continue
            clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            labels = clusterer.fit_predict(emb)
            score, detail = score_combo(
                labels, features, metadata, min_trustworthy, SEED
            )

            all_sweep_results.append({
                "perplexity": perp,
                "mcs": mcs,
                "ms": ms,
                "n_clusters": detail["n_clusters"],
                "noise_pct": detail["noise_pct"],
                "score": score,
                "passes": detail["passes_verification"],
            })

            if score > best_score:
                best_score = score
                best_labels = labels.copy()
                best_detail = detail
                best_mcs = mcs
                best_ms = ms

        print(f"  Best: mcs={best_mcs}, ms={best_ms}, "
              f"score={best_score:.4f}, "
              f"clusters={best_detail['n_clusters']}, "
              f"noise={best_detail['noise_pct']:.1f}%")

        # Save cluster labels and metadata
        np.save(os.path.join(perp_dir, "cluster_labels.npy"), best_labels)
        np.save(os.path.join(perp_dir, "env_id.npy"), metadata["env_id"])
        np.save(os.path.join(perp_dir, "agent_id.npy"), metadata["agent_id"])
        np.save(os.path.join(perp_dir, "step.npy"), metadata["step"])
        print(f"  Saved cluster_labels.npy + metadata arrays to {perp_dir}/")

        # Collect for comparison
        comparison_data.append((perp, emb, best_labels, best_score, best_detail))

        # ------------------------------------------------------------------
        # 4. Per-perplexity visualizations
        # ------------------------------------------------------------------
        config_str = (f"takeru_200M: perp={perp}, mcs={best_mcs}, ms={best_ms}")

        # 3D scatter (2x2 grid, 4 angles)
        plot_3d_scatter(
            emb, best_labels, config_str,
            os.path.join(perp_dir, "tsne_3d_scatter.png"),
        )

        # Cohen's d heatmap
        stats = best_detail["stats"]
        plot_feature_heatmap(
            stats,
            os.path.join(perp_dir, "cluster_features.png"),
            title=f"takeru_200M perp={perp}: Cohen's d vs Global Baseline",
        )

        # Summary text
        write_summary_txt(
            best_detail, perp, best_mcs, best_ms,
            os.path.join(perp_dir, "summary.txt"), N,
        )

    # ------------------------------------------------------------------
    # 5. Comparison figure
    # ------------------------------------------------------------------
    print(f"\n[4/5] Generating comparison figure...", flush=True)
    plot_comparison(comparison_data, os.path.join(OUTPUT_DIR, "comparison.png"))

    # ------------------------------------------------------------------
    # 6. Print full sweep summary
    # ------------------------------------------------------------------
    print(f"\n[5/5] Summary")
    print("=" * 90)
    print(f"{'Perp':>5} {'Best MCS':>9} {'Best MS':>8} {'#Cl':>4} "
          f"{'Noise%':>7} {'Score':>8} {'Pass':>5}")
    print("-" * 90)
    for perp, emb, labels, score, detail in comparison_data:
        n_cl = detail["n_clusters"]
        noise = detail["noise_pct"]
        passes = detail["passes_verification"]
        print(f"{perp:5} {detail.get('best_mcs', '?'):>9} {detail.get('best_ms', '?'):>8} "
              f"{n_cl:4} {noise:6.1f}% {score:8.4f} {'YES' if passes else 'no':>5}")

    # Also print a cleaner version from comparison_data directly
    print("\n" + "=" * 90)
    print("Per-perplexity best configs:")
    for perp, emb, labels, score, detail in comparison_data:
        n_cl = detail["n_clusters"]
        noise = detail["noise_pct"]
        passes = detail["passes_verification"]
        wd = detail["mean_weighted_d"]
        ef = detail["entropy_factor"]
        print(f"  perp={perp:2d}: {n_cl} clusters, {noise:.1f}% noise, "
              f"wtd_d={wd:.2f}, ent={ef:.2f}, score={score:.4f}"
              f"{' [PASSES]' if passes else ''}")

    # Print full grid for reference
    print(f"\nFull HDBSCAN grid results ({len(all_sweep_results)} configs):")
    print(f"{'Perp':>5} {'MCS':>5} {'MS':>4} {'#Cl':>4} {'Noise%':>7} {'Score':>8} {'Pass':>5}")
    print("-" * 50)
    for r in sorted(all_sweep_results, key=lambda x: x["score"], reverse=True)[:30]:
        print(f"{r['perplexity']:5} {r['mcs']:5} {r['ms']:4} "
              f"{r['n_clusters']:4} {r['noise_pct']:6.1f}% "
              f"{r['score']:8.4f} {'YES' if r['passes'] else 'no':>5}")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
