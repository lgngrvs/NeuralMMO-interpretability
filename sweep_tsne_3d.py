#!/usr/bin/env python
"""Sweep T-SNE 3D + HDBSCAN clustering hyperparameters.

Projects activations to 3D via T-SNE (instead of 2D), then clusters with
HDBSCAN.  Produces a CSV of scored results, per-config 3D scatter plots from
multiple camera angles, Cohen's-d feature heatmaps, and summary visualisations.

Usage:
    uv run python sweep_tsne_3d.py activation_data/yaofeng_200M/ --subsample 5
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
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
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


# ---------------------------------------------------------------------------
# 3-D Visualisation helpers
# ---------------------------------------------------------------------------

VIEWING_ANGLES = [
    (30, 45),
    (30, 135),
    (30, 225),
    (60, 0),
]


def plot_3d_clusters(embedding, labels, config_str, output_path):
    """Save a 2x2 grid of 3-D scatter plots from four viewing angles.

    Parameters
    ----------
    embedding : ndarray, shape (N, 3)
    labels : ndarray, shape (N,)  — cluster labels (-1 = noise)
    config_str : str — human-readable config description for the title
    output_path : str — PNG path
    """
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

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

    # Add legend for the first subplot only
    handles, leg_labels = fig.axes[0].get_legend_handles_labels()
    if n_clusters <= 15:
        fig.legend(handles, leg_labels, loc="lower center", ncol=min(n_clusters + 1, 6),
                   markerscale=4, fontsize=8, framealpha=0.9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def plot_feature_heatmap(stats, output_path, title="Cohen's d per Cluster per Feature"):
    """Save a Cohen's-d heatmap (RdBu_r, same style as the existing code)."""
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
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="Cohen's d")

    for i in range(len(cluster_ids)):
        for j in range(n_feats):
            val = d_matrix[i, j]
            if abs(val) > 0.5:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 1.5 else "black")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def write_summary_txt(detail, config_row, output_path, N):
    """Write a human-readable summary for a single config."""
    stats = detail["stats"]
    with open(output_path, "w") as f:
        f.write(f"Strategy: T-SNE 3D + HDBSCAN\n")
        f.write(f"PCA dims: {config_row['pca_dims']}\n")
        f.write(f"T-SNE perplexity: {config_row['tsne_perplexity']}\n")
        f.write(f"HDBSCAN min_cluster_size: {config_row['k_or_mcs']}\n")
        f.write(f"HDBSCAN min_samples: {config_row['min_samples']}\n")
        f.write(f"\nComposite score: {detail['composite_score']:.4f}\n")
        f.write(f"Passes verification: {detail['passes_verification']}\n")
        f.write(f"Clusters: {detail['n_clusters']} (raw: {detail['n_clusters_raw']})\n")
        f.write(f"Noise: {detail['noise_pct']:.1f}%\n")
        f.write(f"Mean weighted |d|: {detail['mean_weighted_d']:.3f}\n")
        f.write(f"Entropy factor: {detail['entropy_factor']:.3f}\n")
        f.write(f"Min step entropy: {detail['min_step_entropy']:.3f}\n")
        f.write(f"Min agent entropy: {detail['min_agent_entropy']:.3f}\n")
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


# ---------------------------------------------------------------------------
# Summary visualisations (across all configs)
# ---------------------------------------------------------------------------

def plot_score_vs_perplexity(df, viz_dir, policy_name="yaofeng_200M"):
    """Box plot of composite score grouped by T-SNE perplexity."""
    df_nonzero = df[df["composite_score"] > 0].copy()
    if df_nonzero.empty:
        print("  No nonzero scores — skipping score_vs_perplexity.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    perplexities = sorted(df_nonzero["tsne_perplexity"].unique())
    data_by_perp = [
        df_nonzero[df_nonzero["tsne_perplexity"] == p]["composite_score"].values
        for p in perplexities
    ]

    bp = ax.boxplot(
        data_by_perp, positions=range(len(perplexities)),
        widths=0.5, patch_artist=True,
        boxprops=dict(facecolor="lightblue", edgecolor="navy"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="navy"),
        capprops=dict(color="navy"),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5),
    )

    for i, (p, vals) in enumerate(zip(perplexities, data_by_perp)):
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
        passing_mask = df_nonzero[df_nonzero["tsne_perplexity"] == p]["passes_verification"].values
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            c=["green" if v else "gray" for v in passing_mask],
            s=20, alpha=0.7, zorder=3, edgecolors="none",
        )

    ax.set_xticks(range(len(perplexities)))
    ax.set_xticklabels([str(int(p)) for p in perplexities])
    ax.set_xlabel("T-SNE Perplexity")
    ax.set_ylabel("Composite Score")
    ax.set_title(f"{policy_name} 3D: Composite Score vs. T-SNE Perplexity\n(green = passes verification)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    out = os.path.join(viz_dir, "score_vs_perplexity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_passing_heatmap(df, viz_dir, policy_name="yaofeng_200M"):
    """Heatmap: perplexity x min_cluster_size (mean composite score)."""
    df_nonzero = df[df["composite_score"] > 0].copy()
    if df_nonzero.empty:
        print("  No nonzero scores — skipping passing_configs_heatmap.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = df_nonzero.groupby(["tsne_perplexity", "k_or_mcs"])["composite_score"].mean().unstack()
    pivot = pivot.sort_index(ascending=True)
    pivot = pivot[sorted(pivot.columns)]

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(i)) for i in pivot.index])
    ax.set_xlabel("HDBSCAN min_cluster_size")
    ax.set_ylabel("T-SNE Perplexity")
    ax.set_title(f"{policy_name} 3D: Mean Composite Score\n(averaged over PCA dims & min_samples)")

    max_val = np.nanmax(pivot.values)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > max_val * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, label="Composite Score")
    out = os.path.join(viz_dir, "passing_configs_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_best_3d_comparison(top3_embeddings, top3_labels, top3_configs, viz_dir, policy_name="yaofeng_200M"):
    """Side-by-side 3D scatter for the top 3 configs (single angle each)."""
    fig = plt.figure(figsize=(18, 6))

    for idx, (emb, labels, cfg_str) in enumerate(
        zip(top3_embeddings, top3_labels, top3_configs)
    ):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")

        cluster_ids = sorted(set(labels) - {-1})
        n_clusters = len(cluster_ids)
        noise_pct = (labels == -1).sum() / len(labels) * 100
        cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

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
                c=[cmap(i)], s=3, alpha=0.5,
                label=f"C{cl} (n={mask.sum()})", rasterized=True,
            )

        ax.view_init(elev=30, azim=45)
        ax.set_xlabel("T-SNE 1", fontsize=8)
        ax.set_ylabel("T-SNE 2", fontsize=8)
        ax.set_zlabel("T-SNE 3", fontsize=8)
        ax.set_title(f"{cfg_str}\n{n_clusters} cl, {noise_pct:.0f}% noise", fontsize=10)
        ax.tick_params(labelsize=7)

        if n_clusters <= 10:
            ax.legend(markerscale=3, fontsize=6, loc="best", framealpha=0.8)

    fig.suptitle(f"{policy_name}: Top 3 T-SNE 3D + HDBSCAN Configurations",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    out = os.path.join(viz_dir, "best_clusters_3d_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep T-SNE 3D + HDBSCAN clustering hyperparameters")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation data")
    parser.add_argument("--subsample", type=int, default=5,
                        help="Keep every Nth record per trajectory (default: 5)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: experiments/T-SNE-3d-sweep-<policy>/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    policy_name = Path(args.data_dir).name

    if args.output_dir is None:
        output_dir = os.path.join("experiments", f"T-SNE-3d-sweep-{policy_name}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...", flush=True)
    records, total_datapoints = load_data(args.data_dir, subsample_rate=args.subsample)
    activations, features, metadata = compute_features(records)
    N = len(activations)
    print(f"Activations: {activations.shape}, Features: {features.shape}")

    min_trustworthy = max(20, int(0.02 * N))
    print(f"Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # ------------------------------------------------------------------
    # 2. Pre-compute PCA reductions
    # ------------------------------------------------------------------
    PCA_DIMS = [20, 30]
    print("Pre-computing PCA reductions...", flush=True)
    pca_cache = {}
    for dims in PCA_DIMS:
        d = min(dims, activations.shape[1], activations.shape[0])
        pca = PCA(n_components=d, random_state=args.seed)
        pca_cache[dims] = pca.fit_transform(activations)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA {d}D: {explained:.1%} variance explained")

    # ------------------------------------------------------------------
    # 3. Pre-compute T-SNE 3D embeddings (PARALLEL)
    # ------------------------------------------------------------------
    TSNE_PERPLEXITIES = [5, 15, 30, 50]
    tsne_cache = {}

    # Build list of jobs
    tsne_jobs = []
    for pca_d in PCA_DIMS:
        for perp in TSNE_PERPLEXITIES:
            if perp >= pca_cache[pca_d].shape[0]:
                print(f"  Skipping perp={perp} (>= n_samples={pca_cache[pca_d].shape[0]})")
                continue
            tsne_jobs.append((pca_d, perp))

    n_workers = min(len(tsne_jobs), 8)
    print(f"Pre-computing {len(tsne_jobs)} T-SNE 3D embeddings "
          f"({n_workers} workers)...", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fit_one(pca_d, perp):
        from sklearn.manifold import TSNE as _TSNE
        t = _TSNE(n_components=3, perplexity=perp, learning_rate="auto",
                  max_iter=1000, random_state=args.seed, verbose=0)
        return t.fit_transform(pca_cache[pca_d])

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fit_one, pca_d, perp): (pca_d, perp)
                   for pca_d, perp in tsne_jobs}
        for fut in as_completed(futures):
            key = futures[fut]
            tsne_cache[key] = fut.result()
            print(f"  Done: PCA={key[0]}D, perplexity={key[1]}", flush=True)
    print()

    # ------------------------------------------------------------------
    # 4. Run HDBSCAN on 3D embeddings
    # ------------------------------------------------------------------
    MCS_VALUES = [50, 100, 200, 400]
    MS_VALUES = [5, 10, 25]

    grid = list(product(PCA_DIMS, TSNE_PERPLEXITIES, MCS_VALUES, MS_VALUES))
    print(f"T-SNE 3D + HDBSCAN sweep: {len(grid)} combos")

    results = []
    for pca_d, perp, mcs, ms in tqdm(grid, desc="T-SNE 3D + HDBSCAN"):
        if ms > mcs:
            continue
        key = (pca_d, perp)
        if key not in tsne_cache:
            continue
        emb = tsne_cache[key]
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        labels = clusterer.fit_predict(emb)
        score, detail = score_combo(labels, features, metadata, min_trustworthy, args.seed)
        row = {
            "strategy": "tsne3d_hdbscan",
            "pca_dims": pca_d,
            "tsne_perplexity": perp,
            "algo": "hdbscan",
            "k_or_mcs": mcs,
            "min_samples": ms,
            "n_clusters_raw": detail["n_clusters_raw"],
            "n_clusters": detail["n_clusters"],
            "noise_pct": detail["noise_pct"],
            "mean_weighted_d": detail["mean_weighted_d"],
            "cluster_count_factor": detail["cluster_count_factor"],
            "entropy_factor": detail["entropy_factor"],
            "min_step_entropy": detail.get("min_step_entropy", ""),
            "min_agent_entropy": detail.get("min_agent_entropy", ""),
            "composite_score": detail["composite_score"],
            "passes_verification": detail["passes_verification"],
        }
        results.append((score, row, detail, labels, key))

    results.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # 5. Write CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    fieldnames = [
        "strategy", "pca_dims", "tsne_perplexity",
        "algo", "k_or_mcs", "min_samples",
        "n_clusters_raw", "n_clusters", "noise_pct",
        "mean_weighted_d", "cluster_count_factor", "entropy_factor",
        "min_step_entropy", "min_agent_entropy",
        "composite_score", "passes_verification",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row, _, _, _ in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nWrote {csv_path} ({len(results)} combos)")

    # ------------------------------------------------------------------
    # 6. Console: top 20
    # ------------------------------------------------------------------
    print(f"\n{'='*110}")
    print(f"{'Strategy':>16} {'PCA':>4} {'Perp':>5} {'Algo':>8} "
          f"{'MCS':>6} {'MS':>4} {'#Cl':>4} {'Noise%':>7} "
          f"{'WtdD':>6} {'Ent':>5} {'Score':>7} {'Pass':>5}")
    print(f"{'-'*110}")
    for i, (score, row, detail, _, _) in enumerate(results[:20]):
        print(f"{row['strategy']:>16} {row['pca_dims']:4} {row['tsne_perplexity']:5} "
              f"{row['algo']:>8} {row['k_or_mcs']:>6} {row['min_samples']:>4} "
              f"{row['n_clusters']:4} {row['noise_pct']:6.1f}% "
              f"{row['mean_weighted_d']:6.2f} {row['entropy_factor']:5.2f} "
              f"{row['composite_score']:7.3f} {'YES' if row['passes_verification'] else 'no':>5}")

    # ------------------------------------------------------------------
    # 7. Top 5 config outputs
    # ------------------------------------------------------------------
    top5 = results[:5]
    top3_embeddings = []
    top3_labels = []
    top3_configs = []

    for rank, (score, row, detail, labels, cache_key) in enumerate(top5, start=1):
        cfg_dir = os.path.join(output_dir, f"config_{rank}")
        os.makedirs(cfg_dir, exist_ok=True)

        pca_d, perp = cache_key
        emb = tsne_cache[cache_key]

        config_str = (f"PCA={pca_d}, perp={perp}, "
                      f"mcs={row['k_or_mcs']}, ms={row['min_samples']}")

        print(f"\n--- Config {rank}: {config_str} (score={score:.3f}) ---")

        # 3D scatter from four angles
        plot_3d_clusters(
            emb, labels, config_str,
            os.path.join(cfg_dir, "tsne_3d_clusters.png"),
        )

        # Feature heatmap
        stats = detail["stats"]
        plot_feature_heatmap(
            stats,
            os.path.join(cfg_dir, "cluster_features.png"),
            title=f"Config {rank}: Cohen's d  ({config_str})",
        )

        # Summary text
        write_summary_txt(detail, row, os.path.join(cfg_dir, "summary.txt"), N)

        # Save labels
        np.save(os.path.join(cfg_dir, "cluster_labels.npy"), labels)

        # Collect top 3 for comparison plot
        if rank <= 3:
            top3_embeddings.append(emb)
            top3_labels.append(labels)
            top3_configs.append(f"#{rank}: {config_str}")

    # ------------------------------------------------------------------
    # 8. Summary visualisations
    # ------------------------------------------------------------------
    import pandas as pd

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    print("\nGenerating summary visualisations...")
    plot_score_vs_perplexity(df, viz_dir, policy_name=policy_name)
    plot_passing_heatmap(df, viz_dir, policy_name=policy_name)
    plot_best_3d_comparison(top3_embeddings, top3_labels, top3_configs, viz_dir, policy_name=policy_name)

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    n_passing = sum(1 for _, row, _, _, _ in results if row["passes_verification"])
    print(f"\n{'='*80}")
    print(f"DONE — {len(results)} combos evaluated, {n_passing} passing verification")
    print(f"Best score: {results[0][0]:.4f}")
    print(f"All outputs in: {output_dir}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
