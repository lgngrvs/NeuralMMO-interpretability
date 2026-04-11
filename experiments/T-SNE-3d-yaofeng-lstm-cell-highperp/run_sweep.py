#!/usr/bin/env python
"""T-SNE 3D clustering sweep on yaofeng_200M LSTM cell state with HIGH perplexity.

Previous sweep used perplexity 5-50 on 34,202 points and got poor results
(best score 0.78, 0 passing verification). The T-SNE literature recommends
perplexity ~ n/100 for n datapoints. With ~34k points, we try perplexity
100-400.

PCA dims: [20, 30]
Perplexities: [100, 200, 300, 400]
HDBSCAN grid: mcs=[50, 100, 200, 400], ms=[5, 10, 25]

T-SNE fits are parallelised with ThreadPoolExecutor (max_workers=8).

Usage:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      uv run python experiments/T-SNE-3d-yaofeng-lstm-cell-highperp/run_sweep.py
"""

import csv
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

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

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


# ---- Configuration ----
DATA_DIR = os.path.join(REPO_ROOT, "activation_data", "yaofeng_200M_lstm_cell")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
SUBSAMPLE_RATE = 5
SEED = 42
PCA_DIMS_LIST = [20, 30]
PERPLEXITIES = [100, 200, 300, 400]
MCS_VALUES = [50, 100, 200, 400]
MS_VALUES = [5, 10, 25]
TOP_K = 5  # number of top configs to save

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


def write_summary_txt(detail, pca_d, perp, mcs, ms, output_path, N):
    """Write a human-readable summary for a config."""
    stats = detail["stats"]
    with open(output_path, "w") as f:
        f.write(f"Strategy: T-SNE 3D + HDBSCAN (high perplexity sweep)\n")
        f.write(f"Policy: yaofeng_200M (LSTM cell state)\n")
        f.write(f"Subsample rate: {SUBSAMPLE_RATE}\n")
        f.write(f"PCA dims: {pca_d}\n")
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
    """1x4 grid showing best config per perplexity (elev=30, azim=45)."""
    n_perps = len(perp_results)
    fig = plt.figure(figsize=(5 * n_perps, 6))

    for idx, (perp, pca_d, emb, labels, score, detail) in enumerate(perp_results):
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
            f"perp={perp}, PCA={pca_d}\n{n_clusters} cl, {noise_pct:.0f}% noise\n"
            f"score={score:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=6)

    fig.suptitle(
        "yaofeng_200M LSTM cell: T-SNE 3D High-Perplexity Comparison",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def plot_score_vs_perplexity(all_results, output_path):
    """Box/strip plot of score vs perplexity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    perps_sorted = sorted(set(r["perplexity"] for r in all_results))
    for perp in perps_sorted:
        scores = [r["score"] for r in all_results if r["perplexity"] == perp]
        x = [perp] * len(scores)
        ax.scatter(x, scores, alpha=0.5, s=30, zorder=3)
        ax.scatter([perp], [max(scores)], marker="*", s=200, c="red",
                   edgecolors="black", zorder=4)

    ax.set_xlabel("Perplexity", fontsize=12)
    ax.set_ylabel("Composite Score", fontsize=12)
    ax.set_title("yaofeng_200M LSTM cell: Score vs Perplexity (high-perp sweep)", fontsize=13)
    ax.set_xticks(perps_sorted)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def plot_passing_heatmap(all_results, output_path):
    """Heatmap of passing configs: PCA x perplexity axes, colored by best score."""
    pca_dims = sorted(set(r["pca_dims"] for r in all_results))
    perps = sorted(set(r["perplexity"] for r in all_results))

    # Build grid: best score for each (pca, perp) combo
    grid = np.full((len(pca_dims), len(perps)), np.nan)
    pass_grid = np.full((len(pca_dims), len(perps)), False)
    for r in all_results:
        pi = pca_dims.index(r["pca_dims"])
        pj = perps.index(r["perplexity"])
        if np.isnan(grid[pi, pj]) or r["score"] > grid[pi, pj]:
            grid[pi, pj] = r["score"]
        if r["passes"]:
            pass_grid[pi, pj] = True

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(len(perps)))
    ax.set_xticklabels([str(p) for p in perps])
    ax.set_yticks(range(len(pca_dims)))
    ax.set_yticklabels([str(d) for d in pca_dims])
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("PCA dims")
    ax.set_title("yaofeng_200M LSTM cell: Best Score per (PCA, Perplexity)")
    fig.colorbar(im, ax=ax, label="Best Composite Score")

    for i in range(len(pca_dims)):
        for j in range(len(perps)):
            val = grid[i, j]
            if not np.isnan(val):
                txt = f"{val:.2f}"
                if pass_grid[i, j]:
                    txt += "\nPASS"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        fontweight="bold" if pass_grid[i, j] else "normal")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 80)
    print("T-SNE 3D High-Perplexity Sweep: yaofeng_200M LSTM cell state")
    print(f"Subsample rate: {SUBSAMPLE_RATE}")
    print(f"PCA dims: {PCA_DIMS_LIST}")
    print(f"Perplexities: {PERPLEXITIES}")
    print(f"HDBSCAN grid: mcs={MCS_VALUES}, ms={MS_VALUES}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...", flush=True)
    records, total_datapoints = load_data(DATA_DIR, subsample_rate=SUBSAMPLE_RATE)
    activations, features, metadata = compute_features(records)
    N = len(activations)
    print(f"Data: {activations.shape} ({N} points, {activations.shape[1]}-dim)")

    min_trustworthy = max(20, int(0.02 * N))
    print(f"Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # ------------------------------------------------------------------
    # 2. PCA reduction (cache both dims)
    # ------------------------------------------------------------------
    print(f"\n[2/6] PCA reduction...", flush=True)
    pca_cache = {}
    for dims in PCA_DIMS_LIST:
        pca = PCA(n_components=dims, random_state=SEED)
        pca_cache[dims] = pca.fit_transform(activations)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA {dims}D: {explained:.1%} variance explained")

    # ------------------------------------------------------------------
    # 3. Parallel T-SNE 3D at each (pca_dims, perplexity) combo
    # ------------------------------------------------------------------
    print(f"\n[3/6] T-SNE 3D (parallelized, {len(PCA_DIMS_LIST)*len(PERPLEXITIES)} fits)...",
          flush=True)

    def fit_tsne(pca_d, perp):
        from sklearn.manifold import TSNE
        t = TSNE(n_components=3, perplexity=perp, learning_rate="auto",
                 max_iter=1000, random_state=SEED, verbose=0)
        return t.fit_transform(pca_cache[pca_d])

    tsne_jobs = [(pd, p) for pd in PCA_DIMS_LIST for p in PERPLEXITIES]
    tsne_cache = {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fit_tsne, pd, p): (pd, p) for pd, p in tsne_jobs}
        for fut in as_completed(futures):
            key = futures[fut]
            tsne_cache[key] = fut.result()
            elapsed = time.time() - t_start
            print(f"  Done: PCA={key[0]}, perp={key[1]}  ({elapsed:.0f}s elapsed)")

    # ------------------------------------------------------------------
    # 4. HDBSCAN sweep over all combos
    # ------------------------------------------------------------------
    print(f"\n[4/6] HDBSCAN sweep...", flush=True)

    all_sweep_results = []  # list of dicts for CSV
    all_scored = []  # (score, detail, labels, pca_d, perp, mcs, ms, embedding)

    # Best per perplexity for comparison plot (keyed by perplexity)
    best_per_perp = {}  # perp -> (score, pca_d, emb, labels, detail)

    hdbscan_grid = list(product(MCS_VALUES, MS_VALUES))

    for (pca_d, perp), emb in sorted(tsne_cache.items()):
        print(f"\n  --- PCA={pca_d}, perp={perp} ---")

        best_score_here = -1
        best_labels_here = None
        best_detail_here = None
        best_mcs_here = None
        best_ms_here = None

        for mcs, ms in tqdm(hdbscan_grid, desc=f"    HDBSCAN", leave=False):
            if ms > mcs:
                continue
            clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            labels = clusterer.fit_predict(emb)
            score, detail = score_combo(
                labels, features, metadata, min_trustworthy, SEED
            )

            row = {
                "pca_dims": pca_d,
                "perplexity": perp,
                "mcs": mcs,
                "ms": ms,
                "n_clusters": detail["n_clusters"],
                "noise_pct": round(detail["noise_pct"], 1),
                "mean_weighted_d": round(detail["mean_weighted_d"], 4),
                "entropy_factor": round(detail["entropy_factor"], 4),
                "score": round(score, 4),
                "passes": detail["passes_verification"],
            }
            all_sweep_results.append(row)
            all_scored.append((score, detail, labels.copy(), pca_d, perp, mcs, ms, emb))

            if score > best_score_here:
                best_score_here = score
                best_labels_here = labels.copy()
                best_detail_here = detail
                best_mcs_here = mcs
                best_ms_here = ms

        print(f"    Best: mcs={best_mcs_here}, ms={best_ms_here}, "
              f"score={best_score_here:.4f}, "
              f"clusters={best_detail_here['n_clusters']}, "
              f"noise={best_detail_here['noise_pct']:.1f}%")

        # Track best per perplexity
        if perp not in best_per_perp or best_score_here > best_per_perp[perp][0]:
            best_per_perp[perp] = (best_score_here, pca_d, emb, best_labels_here,
                                   best_detail_here, best_mcs_here, best_ms_here)

    # ------------------------------------------------------------------
    # 5. Save sweep CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    fieldnames = ["pca_dims", "perplexity", "mcs", "ms", "n_clusters",
                  "noise_pct", "mean_weighted_d", "entropy_factor", "score", "passes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_sweep_results, key=lambda x: x["score"], reverse=True):
            writer.writerow(row)
    print(f"\nWrote {csv_path} ({len(all_sweep_results)} rows)")

    # ------------------------------------------------------------------
    # 6. Top-K config outputs
    # ------------------------------------------------------------------
    print(f"\n[5/6] Generating top-{TOP_K} config outputs...", flush=True)

    all_scored.sort(key=lambda x: x[0], reverse=True)

    for rank, (score, detail, labels, pca_d, perp, mcs, ms, emb) in enumerate(
        all_scored[:TOP_K], start=1
    ):
        config_dir = os.path.join(OUTPUT_DIR, f"config_{rank}")
        os.makedirs(config_dir, exist_ok=True)

        config_str = (
            f"yaofeng_200M LSTM cell: PCA={pca_d}, perp={perp}, "
            f"mcs={mcs}, ms={ms} (score={score:.3f})"
        )
        print(f"\n  Config #{rank}: {config_str}")
        print(f"    Clusters: {detail['n_clusters']}, Noise: {detail['noise_pct']:.1f}%, "
              f"Passes: {detail['passes_verification']}")

        # 3D scatter
        plot_3d_scatter(
            emb, labels, config_str,
            os.path.join(config_dir, "tsne_3d_scatter.png"),
        )

        # Cohen's d heatmap
        plot_feature_heatmap(
            detail["stats"],
            os.path.join(config_dir, "cluster_features.png"),
            title=f"Config #{rank}: Cohen's d (PCA={pca_d}, perp={perp}, mcs={mcs}, ms={ms})",
        )

        # Summary
        write_summary_txt(
            detail, pca_d, perp, mcs, ms,
            os.path.join(config_dir, "summary.txt"), N,
        )

        # Cluster labels
        np.save(os.path.join(config_dir, "cluster_labels.npy"), labels)

    # ------------------------------------------------------------------
    # 7. Comparison + analytics visualizations
    # ------------------------------------------------------------------
    print(f"\n[6/6] Generating comparison and analytics plots...", flush=True)

    # comparison.png: 1x4 grid, best config per perplexity
    comparison_data = []
    for perp in sorted(best_per_perp.keys()):
        sc, pd, emb, lbl, det, _, _ = best_per_perp[perp]
        comparison_data.append((perp, pd, emb, lbl, sc, det))

    plot_comparison(comparison_data, os.path.join(OUTPUT_DIR, "comparison.png"))

    # score_vs_perplexity.png
    plot_score_vs_perplexity(all_sweep_results, os.path.join(VIS_DIR, "score_vs_perplexity.png"))

    # passing_configs_heatmap.png
    plot_passing_heatmap(all_sweep_results, os.path.join(VIS_DIR, "passing_configs_heatmap.png"))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"SWEEP COMPLETE in {elapsed_total:.0f}s")
    print(f"{'='*90}")

    n_passing = sum(1 for r in all_sweep_results if r["passes"])
    n_total = len(all_sweep_results)
    print(f"\nTotal configs tested: {n_total}")
    print(f"Passing verification: {n_passing}/{n_total}")

    print(f"\nTop-{TOP_K} configs:")
    print(f"{'Rank':>5} {'PCA':>4} {'Perp':>5} {'MCS':>5} {'MS':>4} "
          f"{'#Cl':>4} {'Noise%':>7} {'Wtd d':>7} {'Ent':>6} {'Score':>8} {'Pass':>5}")
    print("-" * 75)
    for rank, (score, detail, labels, pca_d, perp, mcs, ms, emb) in enumerate(
        all_scored[:TOP_K], start=1
    ):
        print(f"{rank:5} {pca_d:4} {perp:5} {mcs:5} {ms:4} "
              f"{detail['n_clusters']:4} {detail['noise_pct']:6.1f}% "
              f"{detail['mean_weighted_d']:7.3f} {detail['entropy_factor']:6.3f} "
              f"{score:8.4f} {'YES' if detail['passes_verification'] else 'no':>5}")

    print(f"\nPer-perplexity best:")
    for perp in sorted(best_per_perp.keys()):
        sc, pd, _, lbl, det, mc, ms_ = best_per_perp[perp]
        n_cl = det["n_clusters"]
        noise = det["noise_pct"]
        passes = det["passes_verification"]
        wd = det["mean_weighted_d"]
        ef = det["entropy_factor"]
        print(f"  perp={perp:3d} (PCA={pd}): {n_cl} clusters, {noise:.1f}% noise, "
              f"wtd_d={wd:.2f}, ent={ef:.2f}, score={sc:.4f}"
              f"{' [PASSES]' if passes else ''}")

    # Full grid top 30
    print(f"\nFull grid top 30:")
    print(f"{'PCA':>4} {'Perp':>5} {'MCS':>5} {'MS':>4} {'#Cl':>4} "
          f"{'Noise%':>7} {'Score':>8} {'Pass':>5}")
    print("-" * 55)
    for r in sorted(all_sweep_results, key=lambda x: x["score"], reverse=True)[:30]:
        print(f"{r['pca_dims']:4} {r['perplexity']:5} {r['mcs']:5} {r['ms']:4} "
              f"{r['n_clusters']:4} {r['noise_pct']:6.1f}% "
              f"{r['score']:8.4f} {'YES' if r['passes'] else 'no':>5}")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print(f"Total time: {elapsed_total:.0f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
