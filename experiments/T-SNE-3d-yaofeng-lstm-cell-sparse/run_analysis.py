#!/usr/bin/env python
"""T-SNE 3D clustering with SPARSE subsampling on yaofeng_200M LSTM cell state.

Previous best: perp=100 on subsample=5 (34k points): score 1.024, 3 clusters,
43% noise. The hypothesis is that sparser subsampling (subsample=50) will reduce
temporal autocorrelation, yielding better-separated clusters.

PCA dims: 20
Perplexities: [50, 75, 100]  (sweeping around 100, adjusted for smaller N)
HDBSCAN grid: mcs=[15, 25, 50, 75, 100, 150, 200, 300], ms=[3, 5, 10, 15, 25]

Usage:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      uv run python experiments/T-SNE-3d-yaofeng-lstm-cell-sparse/run_analysis.py
"""

import csv
import gc
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore")

import hdbscan as hdbscan_lib
import numpy as np
from sklearn.decomposition import PCA

# Add scripts/ to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

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
SUBSAMPLE_RATE = 50
SEED = 42
PCA_DIMS = 20
PERPLEXITIES = [50, 75, 100]
MCS_VALUES = [15, 25, 50, 75, 100, 150, 200, 300]
MS_VALUES = [3, 5, 10, 15, 25]
TOP_K = 5

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
                c=[cmap(i)], s=5, alpha=0.6,
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


def write_summary_txt(detail, perp, mcs, ms, output_path, N, transition_rate=None):
    """Write a human-readable summary for a config."""
    stats = detail["stats"]
    with open(output_path, "w") as f:
        f.write(f"Strategy: T-SNE 3D + HDBSCAN (sparse subsampling)\n")
        f.write(f"Policy: yaofeng_200M (LSTM cell state)\n")
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
        if transition_rate is not None:
            f.write(f"Transition rate: {transition_rate:.4f}\n")
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
    """1x3 grid showing best config per perplexity (elev=30, azim=45)."""
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
                c=[cmap(i)], s=3, alpha=0.5,
                label=f"C{cl}", rasterized=True,
            )

        ax.view_init(elev=30, azim=45)
        ax.set_xlabel("T-SNE 1", fontsize=7)
        ax.set_ylabel("T-SNE 2", fontsize=7)
        ax.set_zlabel("T-SNE 3", fontsize=7)
        ax.set_title(
            f"perp={perp}\n{n_clusters} cl, {noise_pct:.0f}% noise\n"
            f"score={score:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=6)

    fig.suptitle(
        f"yaofeng_200M LSTM cell: T-SNE 3D Sparse Subsampling (rate={SUBSAMPLE_RATE})",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Transition rate computation
# ---------------------------------------------------------------------------

def compute_transition_rate(labels, metadata):
    """Compute the fraction of consecutive timesteps where the cluster label changes.

    Groups points by (env_id, agent_id) trajectory, sorts by step, and counts
    transitions. Only counts transitions between non-noise labels. Returns the
    overall transition rate and a per-trajectory dict.
    """
    # Group by trajectory
    traj_indices = defaultdict(list)
    for i in range(len(labels)):
        key = (int(metadata["env_id"][i]), int(metadata["agent_id"][i]))
        traj_indices[key].append(i)

    total_transitions = 0
    total_pairs = 0

    for key, indices in traj_indices.items():
        if len(indices) < 2:
            continue
        # Sort by step
        indices = sorted(indices, key=lambda i: metadata["step"][i])
        traj_labels = labels[indices]

        for j in range(len(traj_labels) - 1):
            l_cur = traj_labels[j]
            l_next = traj_labels[j + 1]
            # Skip if either is noise
            if l_cur == -1 or l_next == -1:
                continue
            total_pairs += 1
            if l_cur != l_next:
                total_transitions += 1

    if total_pairs == 0:
        return 0.0
    return total_transitions / total_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 90)
    print("T-SNE 3D SPARSE SUBSAMPLING: yaofeng_200M LSTM cell state")
    print(f"Subsample rate: {SUBSAMPLE_RATE} (10x sparser than previous)")
    print(f"PCA dims: {PCA_DIMS}")
    print(f"Perplexities: {PERPLEXITIES}")
    print(f"HDBSCAN grid: mcs={MCS_VALUES}, ms={MS_VALUES}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # 1. Load data with subsample=50
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...", flush=True)
    records, total_datapoints = load_data(DATA_DIR, subsample_rate=SUBSAMPLE_RATE)
    activations, features, metadata = compute_features(records)
    N = len(activations)
    print(f"Data: {activations.shape} ({N} points, {activations.shape[1]}-dim)")
    print(f"  (Previous subsample=5 gave ~34k points; subsample=50 gives ~{N} points)")

    min_trustworthy = max(20, int(0.02 * N))
    print(f"Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # Adjust perplexity if too high for the data
    safe_perplexities = [min(p, N // 5) for p in PERPLEXITIES]
    if safe_perplexities != PERPLEXITIES:
        print(f"  Adjusted perplexities for N={N}: {PERPLEXITIES} -> {safe_perplexities}")
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for p in safe_perplexities:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        safe_perplexities = deduped
    perplexities = safe_perplexities

    # ------------------------------------------------------------------
    # 2. PCA reduction
    # ------------------------------------------------------------------
    print(f"\n[2/6] PCA {PCA_DIMS}D reduction...", flush=True)
    pca = PCA(n_components=PCA_DIMS, random_state=SEED)
    pca_data = pca.fit_transform(activations)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {PCA_DIMS}D: {explained:.1%} variance explained")

    del activations
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Parallel T-SNE 3D at each perplexity
    # ------------------------------------------------------------------
    print(f"\n[3/6] T-SNE 3D ({len(perplexities)} perplexities, parallelized)...",
          flush=True)

    def fit_tsne(perp):
        from sklearn.manifold import TSNE
        t = TSNE(n_components=3, perplexity=perp, learning_rate="auto",
                 max_iter=1000, random_state=SEED, verbose=0)
        return t.fit_transform(pca_data)

    tsne_cache = {}

    with ThreadPoolExecutor(max_workers=len(perplexities)) as pool:
        futures = {pool.submit(fit_tsne, p): p for p in perplexities}
        for fut in as_completed(futures):
            perp = futures[fut]
            tsne_cache[perp] = fut.result()
            elapsed = time.time() - t_start
            print(f"  Done: perp={perp}  ({elapsed:.0f}s elapsed)")

    # ------------------------------------------------------------------
    # 4. Thorough HDBSCAN sweep
    # ------------------------------------------------------------------
    print(f"\n[4/6] HDBSCAN sweep...", flush=True)

    all_sweep_results = []  # list of dicts for CSV
    all_scored = []  # (score, detail, labels, perp, mcs, ms, embedding)

    # Best per perplexity for comparison plot
    best_per_perp = {}  # perp -> (score, emb, labels, detail, mcs, ms)

    hdbscan_grid = list(product(MCS_VALUES, MS_VALUES))
    # Filter: ms should not exceed mcs
    hdbscan_grid = [(mcs, ms) for mcs, ms in hdbscan_grid if ms <= mcs]
    print(f"  HDBSCAN combos per perplexity: {len(hdbscan_grid)}")
    print(f"  Total combos: {len(hdbscan_grid) * len(perplexities)}")

    from tqdm import tqdm

    for perp in sorted(tsne_cache.keys()):
        emb = tsne_cache[perp]
        print(f"\n  --- perp={perp} ---")

        best_score_here = -1
        best_labels_here = None
        best_detail_here = None
        best_mcs_here = None
        best_ms_here = None

        for mcs, ms in tqdm(hdbscan_grid, desc=f"    HDBSCAN (perp={perp})", leave=False):
            # Skip combos where min_cluster_size is too big for data
            if mcs > N // 2:
                continue

            clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            labels = clusterer.fit_predict(emb)
            score, detail = score_combo(
                labels, features, metadata, min_trustworthy, SEED
            )

            row = {
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
            all_scored.append((score, detail, labels.copy(), perp, mcs, ms, emb))

            if score > best_score_here:
                best_score_here = score
                best_labels_here = labels.copy()
                best_detail_here = detail
                best_mcs_here = mcs
                best_ms_here = ms

        if best_detail_here is not None:
            print(f"    Best: mcs={best_mcs_here}, ms={best_ms_here}, "
                  f"score={best_score_here:.4f}, "
                  f"clusters={best_detail_here['n_clusters']}, "
                  f"noise={best_detail_here['noise_pct']:.1f}%")

            if perp not in best_per_perp or best_score_here > best_per_perp[perp][0]:
                best_per_perp[perp] = (best_score_here, emb, best_labels_here,
                                       best_detail_here, best_mcs_here, best_ms_here)

    # ------------------------------------------------------------------
    # 5. Save sweep CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "sweep_results.csv")
    fieldnames = ["perplexity", "mcs", "ms", "n_clusters",
                  "noise_pct", "mean_weighted_d", "entropy_factor", "score", "passes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_sweep_results, key=lambda x: x["score"], reverse=True):
            writer.writerow(row)
    print(f"\nWrote {csv_path} ({len(all_sweep_results)} rows)")

    # Sort all results by score
    all_scored.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # 5b. Print full table of ALL configs sorted by score (top 30)
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"ALL CONFIGS SORTED BY SCORE (top 30 of {len(all_sweep_results)}):")
    print(f"{'='*80}")
    print(f"{'Rank':>5} {'Perp':>5} {'MCS':>5} {'MS':>4} "
          f"{'#Cl':>4} {'Noise%':>7} {'Wtd d':>7} {'Ent':>6} {'Score':>8} {'Pass':>5}")
    print("-" * 70)
    for rank, r in enumerate(
        sorted(all_sweep_results, key=lambda x: x["score"], reverse=True)[:30],
        start=1,
    ):
        print(f"{rank:5} {r['perplexity']:5} {r['mcs']:5} {r['ms']:4} "
              f"{r['n_clusters']:4} {r['noise_pct']:6.1f}% "
              f"{r['mean_weighted_d']:7.3f} {r['entropy_factor']:6.3f} "
              f"{r['score']:8.4f} {'YES' if r['passes'] else 'no':>5}")

    # ------------------------------------------------------------------
    # 6. Top-K config outputs with transition rate
    # ------------------------------------------------------------------
    print(f"\n[5/6] Generating top-{TOP_K} config outputs...", flush=True)

    transition_rates = {}

    for rank, (score, detail, labels, perp, mcs, ms, emb) in enumerate(
        all_scored[:TOP_K], start=1
    ):
        config_dir = os.path.join(OUTPUT_DIR, f"config_{rank}")
        os.makedirs(config_dir, exist_ok=True)

        config_str = (
            f"yaofeng_200M LSTM cell (sparse, sub={SUBSAMPLE_RATE}): "
            f"perp={perp}, mcs={mcs}, ms={ms} (score={score:.3f})"
        )
        print(f"\n  Config #{rank}: {config_str}")
        print(f"    Clusters: {detail['n_clusters']}, Noise: {detail['noise_pct']:.1f}%, "
              f"Passes: {detail['passes_verification']}")

        # Transition rate
        tr = compute_transition_rate(labels, metadata)
        transition_rates[rank] = tr
        print(f"    Transition rate: {tr:.4f}")

        # 3D scatter
        plot_3d_scatter(
            emb, labels, config_str,
            os.path.join(config_dir, "tsne_3d_scatter.png"),
        )

        # Cohen's d heatmap
        plot_feature_heatmap(
            detail["stats"],
            os.path.join(config_dir, "cluster_features.png"),
            title=f"Config #{rank}: Cohen's d (perp={perp}, mcs={mcs}, ms={ms})",
        )

        # Summary
        write_summary_txt(
            detail, perp, mcs, ms,
            os.path.join(config_dir, "summary.txt"), N,
            transition_rate=tr,
        )

        # Cluster labels
        np.save(os.path.join(config_dir, "cluster_labels.npy"), labels)

    # ------------------------------------------------------------------
    # 7. Comparison plot + analytics
    # ------------------------------------------------------------------
    print(f"\n[6/6] Generating comparison plot...", flush=True)

    comparison_data = []
    for perp in sorted(best_per_perp.keys()):
        sc, emb, lbl, det, _, _ = best_per_perp[perp]
        comparison_data.append((perp, emb, lbl, sc, det))

    if comparison_data:
        plot_comparison(comparison_data, os.path.join(OUTPUT_DIR, "comparison.png"))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"SWEEP COMPLETE in {elapsed_total:.0f}s")
    print(f"{'='*90}")

    n_passing = sum(1 for r in all_sweep_results if r["passes"])
    n_total = len(all_sweep_results)
    print(f"\nDatapoints: {N} (subsample={SUBSAMPLE_RATE})")
    print(f"PCA {PCA_DIMS}D: {explained:.1%} variance")
    print(f"Total configs tested: {n_total}")
    print(f"Passing verification: {n_passing}/{n_total}")

    print(f"\nTop-{TOP_K} configs:")
    print(f"{'Rank':>5} {'Perp':>5} {'MCS':>5} {'MS':>4} "
          f"{'#Cl':>4} {'Noise%':>7} {'Wtd d':>7} {'Ent':>6} {'Score':>8} {'Pass':>5} {'Trans':>6}")
    print("-" * 80)
    for rank, (score, detail, labels, perp, mcs, ms, emb) in enumerate(
        all_scored[:TOP_K], start=1
    ):
        tr = transition_rates.get(rank, -1)
        print(f"{rank:5} {perp:5} {mcs:5} {ms:4} "
              f"{detail['n_clusters']:4} {detail['noise_pct']:6.1f}% "
              f"{detail['mean_weighted_d']:7.3f} {detail['entropy_factor']:6.3f} "
              f"{score:8.4f} {'YES' if detail['passes_verification'] else 'no':>5} "
              f"{tr:6.4f}")

    print(f"\nPer-perplexity best:")
    for perp in sorted(best_per_perp.keys()):
        sc, emb, lbl, det, mc, ms_ = best_per_perp[perp]
        n_cl = det["n_clusters"]
        noise = det["noise_pct"]
        passes = det["passes_verification"]
        wd = det["mean_weighted_d"]
        ef = det["entropy_factor"]
        tr = compute_transition_rate(lbl, metadata)
        print(f"  perp={perp:3d}: {n_cl} clusters, {noise:.1f}% noise, "
              f"wtd_d={wd:.2f}, ent={ef:.2f}, score={sc:.4f}, trans={tr:.4f}"
              f"{' [PASSES]' if passes else ''}")

    # Comparison with previous results
    print(f"\n{'='*90}")
    print(f"COMPARISON WITH PREVIOUS (subsample=5, ~34k points):")
    print(f"  Previous best: perp=100, score=1.024, 3 clusters, 43.1% noise")
    if all_scored:
        best = all_scored[0]
        sc, det, lbl, perp, mcs, ms, emb = best
        tr = transition_rates.get(1, -1)
        print(f"  Current best:  perp={perp}, score={sc:.4f}, "
              f"{det['n_clusters']} clusters, {det['noise_pct']:.1f}% noise, "
              f"trans={tr:.4f}")
        print(f"  Score delta: {sc - 1.024:+.4f}")
    print(f"{'='*90}")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print(f"Total time: {elapsed_total:.0f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
