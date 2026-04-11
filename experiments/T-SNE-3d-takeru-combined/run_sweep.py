#!/usr/bin/env python
"""T-SNE 3D + HDBSCAN sweep on COMBINED Takeru 200M datasets.

Loads both Takeru 200M activation datasets (original ~57k + extra ~193k),
subsamples by 10 to reduce autocorrelation, combines into ~25k points,
then runs a parallel T-SNE 3D sweep with HDBSCAN clustering.

Usage:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python experiments/T-SNE-3d-takeru-combined/run_sweep.py
"""

import csv
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import hdbscan as hdbscan_lib
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

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
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "T-SNE-3d-takeru-combined")
os.makedirs(OUTPUT_DIR, exist_ok=True)

POLICY_NAME = "takeru_200M_combined"

# ---------------------------------------------------------------------------
# 3D Visualisation helpers
# ---------------------------------------------------------------------------
VIEWING_ANGLES = [
    (30, 45),
    (30, 135),
    (30, 225),
    (60, 0),
]


def plot_3d_clusters(embedding, labels, config_str, output_path):
    """Save a 2x2 grid of 3D scatter plots from four viewing angles."""
    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

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
        f.write(f"Strategy: T-SNE 3D + HDBSCAN (Combined Takeru 200M)\n")
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
# Summary visualisations
# ---------------------------------------------------------------------------

def plot_score_vs_perplexity(df, viz_dir):
    """Box plot of composite score grouped by T-SNE perplexity."""
    df_nonzero = df[df["composite_score"] > 0].copy()
    if df_nonzero.empty:
        print("  No nonzero scores -- skipping score_vs_perplexity.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.set_title(f"{POLICY_NAME} 3D: Composite Score vs. T-SNE Perplexity\n(green = passes verification)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    out = os.path.join(viz_dir, "score_vs_perplexity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")


def plot_passing_heatmap(df, viz_dir):
    """Heatmap: perplexity x min_cluster_size (mean composite score)."""
    df_nonzero = df[df["composite_score"] > 0].copy()
    if df_nonzero.empty:
        print("  No nonzero scores -- skipping passing_configs_heatmap.")
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
    ax.set_title(f"{POLICY_NAME} 3D: Mean Composite Score\n(averaged over PCA dims & min_samples)")

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


def plot_comparison(perp_best, tsne_cache, viz_dir):
    """1x6 grid showing the best config per perplexity value."""
    n_perps = len(perp_best)
    if n_perps == 0:
        print("  No configs to compare -- skipping comparison.png")
        return

    fig = plt.figure(figsize=(5 * n_perps, 5))

    for idx, (perp, info) in enumerate(sorted(perp_best.items())):
        ax = fig.add_subplot(1, n_perps, idx + 1, projection="3d")

        emb = tsne_cache[info["cache_key"]]
        labels = info["labels"]
        score = info["score"]

        cluster_ids = sorted(set(labels) - {-1})
        n_clusters = len(cluster_ids)
        noise_pct = (labels == -1).sum() / len(labels) * 100
        cmap_local = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

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
                c=[cmap_local(i)], s=2, alpha=0.5,
                label=f"C{cl}", rasterized=True,
            )

        ax.view_init(elev=30, azim=45)
        ax.set_xlabel("T1", fontsize=7)
        ax.set_ylabel("T2", fontsize=7)
        ax.set_zlabel("T3", fontsize=7)
        ax.set_title(f"perp={perp}\n{n_clusters}cl, {noise_pct:.0f}%n\nscore={score:.3f}",
                      fontsize=9)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"{POLICY_NAME}: Best Config per Perplexity (T-SNE 3D + HDBSCAN)",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out = os.path.join(viz_dir, "comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out}")

    # Also copy to top-level
    import shutil
    shutil.copy2(out, os.path.join(OUTPUT_DIR, "comparison.png"))
    print(f"  Copied to {os.path.join(OUTPUT_DIR, 'comparison.png')}")


# ---------------------------------------------------------------------------
# Transition rate computation (temporal coherence)
# ---------------------------------------------------------------------------

def compute_transition_rates(labels, metadata):
    """Compute cluster transition rates for temporal coherence analysis.

    Returns a dict with overall transition rate and per-cluster stats.
    """
    steps = metadata["step"]
    env_ids = metadata["env_id"]
    agent_ids = metadata["agent_id"]

    # Group by (env_id, agent_id) trajectory
    from collections import defaultdict
    trajectories = defaultdict(list)
    for i in range(len(labels)):
        key = (int(env_ids[i]), int(agent_ids[i]))
        trajectories[key].append((int(steps[i]), int(labels[i])))

    total_transitions = 0
    total_same = 0
    total_different = 0

    for traj_key, points in trajectories.items():
        points.sort(key=lambda x: x[0])  # sort by step
        for j in range(1, len(points)):
            prev_label = points[j-1][1]
            curr_label = points[j][1]
            if prev_label == -1 or curr_label == -1:
                continue
            total_transitions += 1
            if prev_label == curr_label:
                total_same += 1
            else:
                total_different += 1

    if total_transitions == 0:
        return {"transition_rate": 0.0, "total_transitions": 0}

    transition_rate = total_different / total_transitions
    return {
        "transition_rate": transition_rate,
        "total_transitions": total_transitions,
        "same_cluster": total_same,
        "different_cluster": total_different,
        "n_trajectories": len(trajectories),
    }


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

    print()
    print("=" * 80)
    print("Loading Dataset 2: activation_data/takeru_200M_extra/takeru_200M/")
    print("=" * 80, flush=True)
    data_dir2 = os.path.join(PROJECT_ROOT, "activation_data", "takeru_200M_extra", "takeru_200M")
    records2, total2 = load_data(data_dir2, subsample_rate=10)
    act2, feat2, meta2 = compute_features(records2)
    print(f"  Dataset 2: {act2.shape[0]} records (from {total2} total)")

    # Combine
    activations = np.vstack([act1, act2])
    features = np.vstack([feat1, feat2])
    metadata = {k: np.concatenate([meta1[k], meta2[k]]) for k in meta1}

    # Free memory
    del records1, records2, act1, act2, feat1, feat2, meta1, meta2
    gc.collect()

    N = len(activations)
    print(f"\nCombined data: {activations.shape}")
    print(f"Features: {features.shape}")
    print(f"Total points: {N}")

    min_trustworthy = max(20, int(0.02 * N))
    print(f"Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # ==================================================================
    # 2. PCA reductions
    # ==================================================================
    PCA_DIMS = [20, 30]
    print("\nPre-computing PCA reductions...", flush=True)
    pca_cache = {}
    for dims in PCA_DIMS:
        d = min(dims, activations.shape[1], activations.shape[0])
        pca = PCA(n_components=d, random_state=42)
        pca_cache[dims] = pca.fit_transform(activations)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA {d}D: {explained:.1%} variance explained")

    # ==================================================================
    # 3. Parallel T-SNE 3D embeddings
    # ==================================================================
    TSNE_PERPLEXITIES = [30, 40, 50, 60, 70, 80]
    tsne_jobs = [(pca_d, perp) for pca_d in PCA_DIMS for perp in TSNE_PERPLEXITIES]

    def fit_tsne(pca_d, perp):
        from sklearn.manifold import TSNE
        t = TSNE(
            n_components=3,
            perplexity=perp,
            learning_rate="auto",
            max_iter=1000,
            random_state=42,
            verbose=0,
        )
        return t.fit_transform(pca_cache[pca_d])

    print(f"\nFitting {len(tsne_jobs)} T-SNE 3D embeddings in parallel (max_workers=8)...",
          flush=True)
    tsne_cache = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fit_tsne, pd, p): (pd, p) for pd, p in tsne_jobs}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                tsne_cache[key] = fut.result()
                elapsed = time.time() - t_start
                print(f"  Done: PCA={key[0]}, perp={key[1]}  [{elapsed:.0f}s elapsed]")
            except Exception as e:
                print(f"  FAILED: PCA={key[0]}, perp={key[1]}: {e}")

    print(f"  {len(tsne_cache)}/{len(tsne_jobs)} embeddings completed.")

    # ==================================================================
    # 4. HDBSCAN sweep
    # ==================================================================
    MCS_VALUES = [50, 100, 200, 400]
    MS_VALUES = [5, 10, 25]

    grid = list(product(PCA_DIMS, TSNE_PERPLEXITIES, MCS_VALUES, MS_VALUES))
    print(f"\nHDBSCAN sweep: {len(grid)} combos")

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
        score, detail = score_combo(labels, features, metadata, min_trustworthy, 42)
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

    # ==================================================================
    # 5. Write CSV
    # ==================================================================
    csv_path = os.path.join(OUTPUT_DIR, "sweep_results.csv")
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

    # ==================================================================
    # 6. Console: top 20
    # ==================================================================
    print(f"\n{'='*120}")
    print(f"{'Strategy':>16} {'PCA':>4} {'Perp':>5} {'Algo':>8} "
          f"{'MCS':>6} {'MS':>4} {'#Cl':>4} {'Noise%':>7} "
          f"{'WtdD':>6} {'Ent':>5} {'Score':>7} {'Pass':>5}")
    print(f"{'-'*120}")
    for i, (score, row, detail, _, _) in enumerate(results[:20]):
        print(f"{row['strategy']:>16} {row['pca_dims']:4} {row['tsne_perplexity']:5} "
              f"{row['algo']:>8} {row['k_or_mcs']:>6} {row['min_samples']:>4} "
              f"{row['n_clusters']:4} {row['noise_pct']:6.1f}% "
              f"{row['mean_weighted_d']:6.2f} {row['entropy_factor']:5.2f} "
              f"{row['composite_score']:7.3f} {'YES' if row['passes_verification'] else 'no':>5}")

    # ==================================================================
    # 7. Top 5 config outputs
    # ==================================================================
    top5 = results[:5]
    perp_best = {}  # best config per perplexity for comparison plot

    for rank, (score, row, detail, labels, cache_key) in enumerate(top5, start=1):
        cfg_dir = os.path.join(OUTPUT_DIR, f"config_{rank}")
        os.makedirs(cfg_dir, exist_ok=True)

        pca_d, perp = cache_key
        emb = tsne_cache[cache_key]

        config_str = (f"PCA={pca_d}, perp={perp}, "
                      f"mcs={row['k_or_mcs']}, ms={row['min_samples']}")

        print(f"\n--- Config {rank}: {config_str} (score={score:.3f}) ---")

        # 3D scatter from four angles
        plot_3d_clusters(
            emb, labels, config_str,
            os.path.join(cfg_dir, "tsne_3d_scatter.png"),
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

        # Save labels and metadata arrays
        np.save(os.path.join(cfg_dir, "cluster_labels.npy"), labels)
        np.save(os.path.join(cfg_dir, "env_id.npy"), metadata["env_id"])
        np.save(os.path.join(cfg_dir, "agent_id.npy"), metadata["agent_id"])
        np.save(os.path.join(cfg_dir, "step.npy"), metadata["step"])

        # Compute transition rates for this config
        tr = compute_transition_rates(labels, metadata)
        print(f"  Transition rate: {tr['transition_rate']:.3f} "
              f"({tr['total_transitions']} transitions, "
              f"{tr['n_trajectories']} trajectories)")

        # Append transition info to summary
        with open(os.path.join(cfg_dir, "summary.txt"), "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Temporal Coherence:\n")
            f.write(f"  Transition rate: {tr['transition_rate']:.4f}\n")
            f.write(f"  Total transitions: {tr['total_transitions']}\n")
            f.write(f"  Same cluster: {tr['same_cluster']}\n")
            f.write(f"  Different cluster: {tr['different_cluster']}\n")
            f.write(f"  Trajectories: {tr['n_trajectories']}\n")

    # Track best per perplexity across ALL results (not just top 5)
    for score, row, detail, labels, cache_key in results:
        perp = row["tsne_perplexity"]
        if perp not in perp_best or score > perp_best[perp]["score"]:
            perp_best[perp] = {
                "score": score,
                "labels": labels,
                "cache_key": cache_key,
                "row": row,
            }

    # ==================================================================
    # 8. Summary visualisations
    # ==================================================================
    import pandas as pd

    viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    print("\nGenerating summary visualisations...")
    plot_score_vs_perplexity(df, viz_dir)
    plot_passing_heatmap(df, viz_dir)
    plot_comparison(perp_best, tsne_cache, viz_dir)

    # ==================================================================
    # 9. Final summary
    # ==================================================================
    n_passing = sum(1 for _, row, _, _, _ in results if row["passes_verification"])
    elapsed = time.time() - t_start

    print(f"\n{'='*80}")
    print(f"DONE -- {len(results)} combos evaluated, {n_passing} passing verification")
    if results:
        print(f"Best score: {results[0][0]:.4f}")
        best_row = results[0][1]
        print(f"Best config: PCA={best_row['pca_dims']}, perp={best_row['tsne_perplexity']}, "
              f"mcs={best_row['k_or_mcs']}, ms={best_row['min_samples']}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"All outputs in: {OUTPUT_DIR}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
