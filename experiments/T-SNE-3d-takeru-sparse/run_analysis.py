#!/usr/bin/env python
"""T-SNE 3D clustering on COMBINED Takeru 200M data with sparse subsampling.

Hypothesis: very sparse subsampling (25, 50, 100) will reduce temporal
autocorrelation and improve cluster quality, as seen with yaofeng LSTM cell
state (noise dropped 43% -> 29%, transition rate 0.42 -> 0.28 with sub=50).

Combines both datasets:
  - activation_data/takeru_200M/  (original)
  - activation_data/takeru_200M_extra/takeru_200M/  (new, 11GB JSONL)

For each subsample rate, runs:
  PCA 20D -> T-SNE 3D (perplexities 50, 75, 100, 120) -> HDBSCAN sweep

Usage:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
      uv run python experiments/T-SNE-3d-takeru-sparse/run_analysis.py
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
DATA_DIR_1 = os.path.join(REPO_ROOT, "activation_data", "takeru_200M")
DATA_DIR_2 = os.path.join(REPO_ROOT, "activation_data", "takeru_200M_extra", "takeru_200M")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
PCA_DIMS = 20
SUBSAMPLE_RATES = [25, 50, 100]
PERPLEXITIES = [50, 75, 100, 120]
MCS_VALUES = [15, 25, 50, 100, 150, 200, 300, 400]
MS_VALUES = [3, 5, 10, 25]

VIEWING_ANGLES = [
    (30, 45),
    (30, 135),
    (30, 225),
    (60, 0),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_combined(subsample_rate):
    """Load both Takeru datasets combined at a given subsample rate."""
    print(f"  Loading dataset 1: {DATA_DIR_1} (sub={subsample_rate})...", flush=True)
    r1, t1 = load_data(DATA_DIR_1, subsample_rate=subsample_rate)
    a1, f1, m1 = compute_features(r1)
    del r1
    gc.collect()
    print(f"    Dataset 1: {len(a1)} points", flush=True)

    print(f"  Loading dataset 2: {DATA_DIR_2} (sub={subsample_rate})...", flush=True)
    r2, t2 = load_data(DATA_DIR_2, subsample_rate=subsample_rate)
    a2, f2, m2 = compute_features(r2)
    del r2
    gc.collect()
    print(f"    Dataset 2: {len(a2)} points", flush=True)

    # Combine
    act = np.vstack([a1, a2])
    feat = np.vstack([f1, f2])
    meta = {k: np.concatenate([m1[k], m2[k]]) for k in m1}

    del a1, a2, f1, f2, m1, m2
    gc.collect()

    print(f"    Combined: {len(act)} points", flush=True)
    return act, feat, meta


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


def write_summary_txt(detail, perp, mcs, ms, output_path, N, subsample_rate,
                      transition_rate=None):
    """Write a human-readable summary for a config."""
    stats = detail["stats"]
    with open(output_path, "w") as f:
        f.write(f"Strategy: T-SNE 3D + HDBSCAN (sparse subsampling)\n")
        f.write(f"Policy: Takeru 200M (combined datasets)\n")
        f.write(f"Subsample rate: {subsample_rate}\n")
        f.write(f"PCA dims: {PCA_DIMS}\n")
        f.write(f"T-SNE perplexity: {perp}\n")
        f.write(f"HDBSCAN min_cluster_size: {mcs}\n")
        f.write(f"HDBSCAN min_samples: {ms}\n")
        f.write(f"\nN points: {N}\n")
        f.write(f"Composite score: {detail['composite_score']:.4f}\n")
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


# ---------------------------------------------------------------------------
# Transition rate computation
# ---------------------------------------------------------------------------

def compute_transition_rate(labels, metadata):
    """Compute the fraction of consecutive timesteps where the cluster label changes.

    Groups points by (env_id, agent_id) trajectory, sorts by step, and counts
    transitions. Only counts transitions between non-noise labels.
    """
    traj_indices = defaultdict(list)
    for i in range(len(labels)):
        key = (int(metadata["env_id"][i]), int(metadata["agent_id"][i]))
        traj_indices[key].append(i)

    total_transitions = 0
    total_pairs = 0

    for key, indices in traj_indices.items():
        if len(indices) < 2:
            continue
        indices = sorted(indices, key=lambda i: metadata["step"][i])
        traj_labels = labels[indices]

        for j in range(len(traj_labels) - 1):
            l_cur = traj_labels[j]
            l_next = traj_labels[j + 1]
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
    print("T-SNE 3D SPARSE SUBSAMPLING: Takeru 200M Combined")
    print(f"Subsample rates: {SUBSAMPLE_RATES}")
    print(f"PCA dims: {PCA_DIMS}")
    print(f"Perplexities: {PERPLEXITIES}")
    print(f"HDBSCAN grid: mcs={MCS_VALUES}, ms={MS_VALUES}")
    print("=" * 90)

    # Build HDBSCAN grid (ms <= mcs)
    hdbscan_grid = [(mcs, ms) for mcs, ms in product(MCS_VALUES, MS_VALUES) if ms <= mcs]
    print(f"HDBSCAN combos per perplexity: {len(hdbscan_grid)}")

    # Master results for final comparison table
    master_results = []  # list of dicts

    # Store best config per subsample rate for comparison plot
    comparison_data = []  # (sub_rate, emb, labels, score, detail, n_points)

    for sub_rate in SUBSAMPLE_RATES:
        sub_start = time.time()
        sub_dir = os.path.join(OUTPUT_DIR, f"sub_{sub_rate}")
        os.makedirs(sub_dir, exist_ok=True)

        print(f"\n{'#'*90}")
        print(f"# SUBSAMPLE RATE = {sub_rate}")
        print(f"{'#'*90}")

        # ------------------------------------------------------------------
        # 1. Load combined data
        # ------------------------------------------------------------------
        print(f"\n[1/5] Loading combined data (sub={sub_rate})...", flush=True)
        activations, features, metadata = load_combined(sub_rate)
        N = len(activations)
        print(f"  Total points: {N} (activations shape: {activations.shape})")

        min_trustworthy = max(20, int(0.02 * N))
        print(f"  Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

        # Adjust perplexity if too high for the data
        safe_perplexities = [min(p, N // 5) for p in PERPLEXITIES]
        seen = set()
        deduped = []
        for p in safe_perplexities:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        perplexities = deduped
        if perplexities != PERPLEXITIES:
            print(f"  Adjusted perplexities for N={N}: {PERPLEXITIES} -> {perplexities}")

        # ------------------------------------------------------------------
        # 2. PCA reduction
        # ------------------------------------------------------------------
        print(f"\n[2/5] PCA {PCA_DIMS}D reduction...", flush=True)
        pca = PCA(n_components=PCA_DIMS, random_state=SEED)
        pca_data = pca.fit_transform(activations)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA {PCA_DIMS}D: {explained:.1%} variance explained")

        del activations
        gc.collect()

        # ------------------------------------------------------------------
        # 3. Parallel T-SNE 3D
        # ------------------------------------------------------------------
        print(f"\n[3/5] T-SNE 3D ({len(perplexities)} perplexities, parallelized)...",
              flush=True)

        def fit_tsne(perp):
            from sklearn.manifold import TSNE
            t = TSNE(n_components=3, perplexity=perp, learning_rate="auto",
                     max_iter=1000, random_state=SEED, verbose=0)
            return t.fit_transform(pca_data)

        tsne_cache = {}
        n_workers = min(8, len(perplexities))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(fit_tsne, p): p for p in perplexities}
            for fut in as_completed(futures):
                perp = futures[fut]
                tsne_cache[perp] = fut.result()
                elapsed = time.time() - sub_start
                print(f"    Done: perp={perp}  ({elapsed:.0f}s elapsed)")

        # ------------------------------------------------------------------
        # 4. HDBSCAN sweep
        # ------------------------------------------------------------------
        print(f"\n[4/5] HDBSCAN sweep...", flush=True)

        all_sweep_results = []
        all_scored = []  # (score, detail, labels, perp, mcs, ms, embedding)
        best_per_perp = {}

        from tqdm import tqdm

        for perp in sorted(tsne_cache.keys()):
            emb = tsne_cache[perp]
            print(f"\n  --- perp={perp} ---")

            best_score_here = -1
            best_labels_here = None
            best_detail_here = None
            best_mcs_here = None
            best_ms_here = None

            for mcs, ms in tqdm(hdbscan_grid, desc=f"    HDBSCAN (perp={perp})",
                                leave=False):
                if mcs > N // 2:
                    continue

                clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
                labels = clusterer.fit_predict(emb)
                score, detail = score_combo(
                    labels, features, metadata, min_trustworthy, SEED
                )

                row = {
                    "subsample_rate": sub_rate,
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

        # Save sweep CSV for this subsample rate
        csv_path = os.path.join(sub_dir, "sweep_results.csv")
        fieldnames = ["subsample_rate", "perplexity", "mcs", "ms", "n_clusters",
                      "noise_pct", "mean_weighted_d", "entropy_factor", "score", "passes"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(all_sweep_results, key=lambda x: x["score"], reverse=True):
                writer.writerow(row)
        print(f"\n  Wrote {csv_path} ({len(all_sweep_results)} rows)")

        # Print top configs for this subsample rate
        all_scored.sort(key=lambda x: x[0], reverse=True)

        print(f"\n  {'='*80}")
        print(f"  TOP 20 CONFIGS (sub={sub_rate}):")
        print(f"  {'='*80}")
        print(f"  {'Rank':>5} {'Perp':>5} {'MCS':>5} {'MS':>4} "
              f"{'#Cl':>4} {'Noise%':>7} {'Wtd d':>7} {'Ent':>6} {'Score':>8} {'Pass':>5}")
        print(f"  {'-'*70}")
        for rank, (score, detail, labels, perp, mcs, ms, emb) in enumerate(
            all_scored[:20], start=1
        ):
            print(f"  {rank:5} {perp:5} {mcs:5} {ms:4} "
                  f"{detail['n_clusters']:4} {detail['noise_pct']:6.1f}% "
                  f"{detail['mean_weighted_d']:7.3f} {detail['entropy_factor']:6.3f} "
                  f"{score:8.4f} {'YES' if detail['passes_verification'] else 'no':>5}")

        # ------------------------------------------------------------------
        # 5. Generate outputs for best overall config
        # ------------------------------------------------------------------
        print(f"\n[5/5] Generating outputs for best config (sub={sub_rate})...", flush=True)

        if all_scored:
            best_score, best_detail, best_labels, best_perp, best_mcs, best_ms, best_emb = all_scored[0]

            # Transition rate
            tr = compute_transition_rate(best_labels, metadata)
            print(f"  Best config transition rate: {tr:.4f}")

            config_str = (
                f"Takeru 200M combined (sub={sub_rate}, N={N}): "
                f"perp={best_perp}, mcs={best_mcs}, ms={best_ms} (score={best_score:.3f})"
            )

            # 3D scatter
            plot_3d_scatter(
                best_emb, best_labels, config_str,
                os.path.join(sub_dir, "tsne_3d_scatter.png"),
            )

            # Cohen's d heatmap
            plot_feature_heatmap(
                best_detail["stats"],
                os.path.join(sub_dir, "cluster_features.png"),
                title=f"Takeru 200M (sub={sub_rate}): Cohen's d (perp={best_perp}, mcs={best_mcs}, ms={best_ms})",
            )

            # Summary
            write_summary_txt(
                best_detail, best_perp, best_mcs, best_ms,
                os.path.join(sub_dir, "summary.txt"), N, sub_rate,
                transition_rate=tr,
            )

            # Save arrays
            np.save(os.path.join(sub_dir, "cluster_labels.npy"), best_labels)
            np.save(os.path.join(sub_dir, "env_id.npy"), metadata["env_id"])
            np.save(os.path.join(sub_dir, "agent_id.npy"), metadata["agent_id"])
            np.save(os.path.join(sub_dir, "step.npy"), metadata["step"])
            print(f"  Saved .npy arrays to {sub_dir}/")

            # Record for master table
            master_results.append({
                "subsample_rate": sub_rate,
                "n_points": N,
                "best_perp": best_perp,
                "best_mcs": best_mcs,
                "best_ms": best_ms,
                "n_clusters": best_detail["n_clusters"],
                "noise_pct": best_detail["noise_pct"],
                "score": best_score,
                "passes_verification": best_detail["passes_verification"],
                "transition_rate": tr,
                "mean_weighted_d": best_detail["mean_weighted_d"],
                "entropy_factor": best_detail["entropy_factor"],
            })

            # Store for comparison plot
            comparison_data.append((sub_rate, best_emb, best_labels, best_score,
                                    best_detail, N))

        # Free memory before next subsample rate
        del pca_data, features, metadata, tsne_cache, all_scored, all_sweep_results
        gc.collect()

        sub_elapsed = time.time() - sub_start
        print(f"\n  Sub={sub_rate} completed in {sub_elapsed:.0f}s")

    # ------------------------------------------------------------------
    # Comparison plot: 1x3 grid showing best config per subsample rate
    # ------------------------------------------------------------------
    print(f"\n{'='*90}")
    print("Generating comparison plot across subsample rates...")
    print(f"{'='*90}")

    if comparison_data:
        n_panels = len(comparison_data)
        fig = plt.figure(figsize=(6 * n_panels, 6))

        for idx, (sub_rate, emb, labels, score, detail, n_pts) in enumerate(comparison_data):
            ax = fig.add_subplot(1, n_panels, idx + 1, projection="3d")

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
                f"sub={sub_rate} (N={n_pts})\n"
                f"{n_clusters} cl, {noise_pct:.0f}% noise\n"
                f"score={score:.3f}",
                fontsize=9,
            )
            ax.tick_params(labelsize=6)

        fig.suptitle(
            "Takeru 200M Combined: T-SNE 3D Sparse Subsampling Comparison",
            fontsize=13, y=1.02,
        )
        fig.tight_layout()
        comp_path = os.path.join(OUTPUT_DIR, "comparison.png")
        fig.savefig(comp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {comp_path}")

    # ------------------------------------------------------------------
    # Master results table
    # ------------------------------------------------------------------
    total_elapsed = time.time() - t_start

    print(f"\n{'='*90}")
    print("MASTER RESULTS TABLE")
    print(f"{'='*90}")
    print(f"{'Sub':>5} {'N':>7} {'Perp':>5} {'MCS':>5} {'MS':>4} "
          f"{'#Cl':>4} {'Noise%':>7} {'Wtd d':>7} {'Ent':>6} {'Score':>8} "
          f"{'Pass':>5} {'Trans':>6}")
    print("-" * 90)
    for r in master_results:
        print(f"{r['subsample_rate']:5} {r['n_points']:7} "
              f"{r['best_perp']:5} {r['best_mcs']:5} {r['best_ms']:4} "
              f"{r['n_clusters']:4} {r['noise_pct']:6.1f}% "
              f"{r['mean_weighted_d']:7.3f} {r['entropy_factor']:6.3f} "
              f"{r['score']:8.4f} {'YES' if r['passes_verification'] else 'no':>5} "
              f"{r['transition_rate']:6.4f}")

    print(f"\n{'='*90}")
    print(f"ANALYSIS COMPLETE in {total_elapsed:.0f}s")
    print(f"{'='*90}")
    print(f"\nKey question: Does sparse subsampling help Takeru like it helped LSTM cell state?")
    print(f"  LSTM cell (yaofeng): sub=5 -> sub=50: noise 43%->29%, trans 0.42->0.28")
    for r in master_results:
        status = "PASSES" if r["passes_verification"] else "FAILS"
        print(f"  Takeru sub={r['subsample_rate']}: {r['n_clusters']} clusters, "
              f"{r['noise_pct']:.1f}% noise, trans={r['transition_rate']:.4f}, "
              f"score={r['score']:.4f} [{status}]")

    print(f"\nAll outputs in: {OUTPUT_DIR}/")
    print(f"Total time: {total_elapsed:.0f}s")
    print("=" * 90)


if __name__ == "__main__":
    main()
