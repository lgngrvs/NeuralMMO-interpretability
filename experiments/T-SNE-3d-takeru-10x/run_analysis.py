#!/usr/bin/env python
"""T-SNE 3D + HDBSCAN analysis on 10x Takeru 200M activation data.

Loads all datasets (original + extra + 8 batches), runs PCA -> T-SNE 3D -> HDBSCAN,
scores with verification metrics, and produces visualizations.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analyze_activations import (
    FEATURE_NAMES,
    compute_cluster_stats,
    compute_features,
    load_data,
)
from sweep_cluster_params import score_combo

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def load_dataset(path, subsample_rate=100):
    """Load a dataset, falling back to .npz cache when source json is absent."""
    try:
        records, _ = load_data(path, subsample_rate=subsample_rate)
        act, feat, meta = compute_features(records)
        return act, feat, meta
    except FileNotFoundError:
        pass

    # Fallback: load directly from .npz cache
    data_dir = Path(path)
    npz_files = list(data_dir.rglob("activations.cache.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No data or cache found in {path}")

    all_acts, all_feats, all_steps, all_env, all_agent = [], [], [], [], []
    for npz_file in npz_files:
        print(f"  Loading cache directly: {npz_file}...", flush=True)
        data = np.load(npz_file)
        all_acts.append(data["activations"])
        all_feats.append(data["features"])
        all_steps.append(data["steps"])
        all_env.append(data["env_ids"])
        all_agent.append(data["agent_ids"])

    acts = np.concatenate(all_acts)
    feats = np.concatenate(all_feats)
    steps = np.concatenate(all_steps)
    env_ids = np.concatenate(all_env)
    agent_ids = np.concatenate(all_agent)

    # Subsample per trajectory
    if subsample_rate and subsample_rate > 1:
        traj_keys = env_ids.astype(np.int64) * 1_000_000 + agent_ids.astype(np.int64)
        unique_keys = np.unique(traj_keys)
        keep_mask = np.zeros(len(acts), dtype=bool)
        for key in unique_keys:
            mask = traj_keys == key
            indices = np.where(mask)[0]
            sorted_order = np.argsort(steps[indices])
            sorted_indices = indices[sorted_order]
            keep_mask[sorted_indices[::subsample_rate]] = True
        acts = acts[keep_mask]
        feats = feats[keep_mask]
        steps = steps[keep_mask]
        env_ids = env_ids[keep_mask]
        agent_ids = agent_ids[keep_mask]
        print(f"  {keep_mask.sum()} records after subsampling (rate={subsample_rate})", flush=True)

    meta = {"step": steps, "env_id": env_ids, "agent_id": agent_ids}
    return acts, feats, meta


# ---------------------------------------------------------------------------
# 3D Visualization helpers (adapted from sweep_tsne_3d.py)
# ---------------------------------------------------------------------------
VIEWING_ANGLES = [(30, 45), (30, 135), (30, 225), (60, 0)]


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
            ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                       embedding[noise_mask, 2], c="lightgray", s=1, alpha=0.3,
                       label="noise", rasterized=True)
        for i, cl in enumerate(cluster_ids):
            mask = labels == cl
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                       c=[cmap(i)], s=3, alpha=0.5,
                       label=f"C{cl} (n={mask.sum()})", rasterized=True)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("T-SNE 1", fontsize=8)
        ax.set_ylabel("T-SNE 2", fontsize=8)
        ax.set_zlabel("T-SNE 3", fontsize=8)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle(f"{config_str}\n{n_clusters} clusters, {noise_pct:.1f}% noise",
                 fontsize=13, y=0.98)
    handles, leg_labels = fig.axes[0].get_legend_handles_labels()
    if n_clusters <= 15:
        fig.legend(handles, leg_labels, loc="lower center",
                   ncol=min(n_clusters + 1, 6), markerscale=4, fontsize=8, framealpha=0.9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def plot_feature_heatmap(stats, output_path, title="Cohen's d per Cluster"):
    """Save a Cohen's-d heatmap."""
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
        f.write(f"Strategy: T-SNE 3D + HDBSCAN (10x Takeru 200M)\n")
        f.write(f"PCA dims: {config_row['pca_dims']}\n")
        f.write(f"T-SNE perplexity: {config_row['tsne_perplexity']}\n")
        f.write(f"HDBSCAN min_cluster_size: {config_row['k_or_mcs']}\n")
        f.write(f"HDBSCAN min_samples: {config_row['min_samples']}\n")
        f.write(f"Total data points: {N}\n")
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


def compute_transition_rates(labels, metadata, N):
    """Compute per-cluster transition rates (how often agents switch clusters)."""
    # Group by trajectory (env_id, agent_id)
    traj_keys = metadata["env_id"].astype(np.int64) * 1_000_000 + metadata["agent_id"].astype(np.int64)
    unique_trajs = np.unique(traj_keys)

    total_transitions = 0
    total_steps = 0
    cluster_transitions = {}  # from_cluster -> to_cluster -> count

    for key in unique_trajs:
        mask = traj_keys == key
        indices = np.where(mask)[0]
        sorted_order = np.argsort(metadata["step"][indices])
        sorted_indices = indices[sorted_order]
        traj_labels = labels[sorted_indices]

        # Count transitions
        for i in range(1, len(traj_labels)):
            if traj_labels[i] == -1 or traj_labels[i-1] == -1:
                continue
            total_steps += 1
            if traj_labels[i] != traj_labels[i-1]:
                total_transitions += 1
                from_c = traj_labels[i-1]
                to_c = traj_labels[i]
                if from_c not in cluster_transitions:
                    cluster_transitions[from_c] = {}
                cluster_transitions[from_c][to_c] = cluster_transitions[from_c].get(to_c, 0) + 1

    transition_rate = total_transitions / max(total_steps, 1)
    return transition_rate, cluster_transitions, total_transitions, total_steps


def plot_comparison(embeddings_by_perp, labels_by_perp, configs_by_perp, output_path):
    """Side-by-side comparison of best config per perplexity."""
    n = len(embeddings_by_perp)
    if n == 0:
        return

    fig = plt.figure(figsize=(6 * n, 6))
    for idx, (perp, emb, labels, cfg_str) in enumerate(
        zip(configs_by_perp.keys(), embeddings_by_perp, labels_by_perp, configs_by_perp.values())
    ):
        ax = fig.add_subplot(1, n, idx + 1, projection="3d")
        cluster_ids = sorted(set(labels) - {-1})
        n_clusters = len(cluster_ids)
        noise_pct = (labels == -1).sum() / len(labels) * 100
        cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(emb[noise_mask, 0], emb[noise_mask, 1], emb[noise_mask, 2],
                       c="lightgray", s=1, alpha=0.3, rasterized=True)
        for i, cl in enumerate(cluster_ids):
            mask = labels == cl
            ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                       c=[cmap(i)], s=3, alpha=0.5,
                       label=f"C{cl} (n={mask.sum()})", rasterized=True)
        ax.view_init(elev=30, azim=45)
        ax.set_xlabel("T-SNE 1", fontsize=8)
        ax.set_ylabel("T-SNE 2", fontsize=8)
        ax.set_zlabel("T-SNE 3", fontsize=8)
        ax.set_title(f"perp={perp}\n{cfg_str}\n{n_clusters} cl, {noise_pct:.0f}% noise", fontsize=9)
        ax.tick_params(labelsize=7)
        if n_clusters <= 10:
            ax.legend(markerscale=3, fontsize=6, loc="best", framealpha=0.8)

    fig.suptitle("Takeru 200M (10x data): Best per Perplexity", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # ==================================================================
    # Step 1: Load all datasets
    # ==================================================================
    print("=" * 60)
    print("Step 1: Loading all datasets at subsample=100")
    print("=" * 60)

    base_dir = str(Path(__file__).resolve().parent.parent.parent)
    datasets = [
        os.path.join(base_dir, 'activation_data/takeru_200M/'),
        os.path.join(base_dir, 'activation_data/takeru_200M_extra/takeru_200M/'),
        os.path.join(base_dir, 'activation_data/takeru_200M_batch_3/takeru_200M/'),
        os.path.join(base_dir, 'activation_data/takeru_200M_batch_4/takeru_200M/'),
        os.path.join(base_dir, 'activation_data/takeru_200M_batch_5/takeru_200M/'),
        os.path.join(base_dir, 'activation_data/takeru_200M_batch_6/takeru_200M/'),
        os.path.join(base_dir, 'activation_data/takeru_200M_batch_7/takeru_200M/'),
    ]

    all_acts, all_feats, all_metas = [], [], []
    for path in datasets:
        try:
            act, feat, meta = load_dataset(path, subsample_rate=100)
            all_acts.append(act)
            all_feats.append(feat)
            all_metas.append(meta)
            gc.collect()
            print(f"  {path}: {act.shape[0]} points")
        except Exception as e:
            print(f"  SKIP {path}: {e}")

    if not all_acts:
        print("ERROR: No data loaded!")
        return

    activations = np.vstack(all_acts)
    features = np.vstack(all_feats)
    metadata = {k: np.concatenate([m[k] for m in all_metas]) for k in all_metas[0]}
    N = len(activations)
    print(f"\nCombined: {activations.shape} ({N} points, {activations.shape[1]}D)")
    del all_acts, all_feats, all_metas
    gc.collect()

    min_trustworthy = max(20, int(0.02 * N))
    print(f"Min trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # ==================================================================
    # Step 2: PCA reduction
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 2: PCA reduction to 20D")
    print("=" * 60)

    pca = PCA(n_components=20, random_state=SEED)
    pca_20 = pca.fit_transform(activations)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA 20D: {explained:.1%} variance explained")

    # ==================================================================
    # Step 3: T-SNE 3D (parallel for 4 perplexities)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 3: T-SNE 3D embeddings (parallel)")
    print("=" * 60)

    PERPLEXITIES = [50, 75, 100, 120]
    tsne_cache = {}

    def _fit_tsne(perp):
        from sklearn.manifold import TSNE as _TSNE
        print(f"  Starting T-SNE perplexity={perp}...", flush=True)
        t = _TSNE(n_components=3, perplexity=perp, learning_rate="auto",
                  max_iter=1000, random_state=SEED, verbose=0)
        result = t.fit_transform(pca_20)
        print(f"  Done T-SNE perplexity={perp}", flush=True)
        return perp, result

    n_workers = min(len(PERPLEXITIES), 8)
    print(f"Running {len(PERPLEXITIES)} T-SNE embeddings with {n_workers} workers...")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fit_tsne, perp): perp for perp in PERPLEXITIES}
        for fut in as_completed(futures):
            perp, emb = fut.result()
            tsne_cache[perp] = emb

    print(f"All T-SNE embeddings computed: {list(tsne_cache.keys())}")

    # ==================================================================
    # Step 4: HDBSCAN sweep
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 4: HDBSCAN sweep")
    print("=" * 60)

    MCS_VALUES = [15, 25, 50, 100, 150, 200, 300, 400]
    MS_VALUES = [3, 5, 10, 25]

    grid = list(product(PERPLEXITIES, MCS_VALUES, MS_VALUES))
    print(f"Sweeping {len(grid)} combos (4 perplexities x {len(MCS_VALUES)} mcs x {len(MS_VALUES)} ms)")

    results = []
    for perp, mcs, ms in grid:
        if ms > mcs:
            continue
        emb = tsne_cache[perp]
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms)
        labels = clusterer.fit_predict(emb)
        score, detail = score_combo(labels, features, metadata, min_trustworthy, SEED)

        row = {
            "strategy": "tsne3d_hdbscan",
            "pca_dims": 20,
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
        results.append((score, row, detail, labels, perp))

        if score > 0:
            status = "PASS" if detail["passes_verification"] else "    "
            print(f"  [{status}] perp={perp} mcs={mcs:3d} ms={ms:2d}: "
                  f"score={score:.3f} cl={detail['n_clusters']} "
                  f"noise={detail['noise_pct']:.0f}% d={detail['mean_weighted_d']:.2f}")

    results.sort(key=lambda x: x[0], reverse=True)

    # ==================================================================
    # Step 5: Write CSV
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
    # Step 6: Console summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("Top 20 configs:")
    print("=" * 60)
    print(f"{'perp':>5} {'mcs':>4} {'ms':>3} {'cl':>3} {'noise%':>7} "
          f"{'wt_d':>6} {'ent':>5} {'score':>7} {'pass':>5}")
    for score, row, detail, _, _ in results[:20]:
        print(f"{row['tsne_perplexity']:5} {row['k_or_mcs']:4d} {row['min_samples']:3d} "
              f"{row['n_clusters']:3d} {row['noise_pct']:6.1f}% "
              f"{row['mean_weighted_d']:6.3f} {row['entropy_factor']:5.3f} "
              f"{score:7.3f} {'YES' if row['passes_verification'] else ' no'}")

    n_passing = sum(1 for _, r, _, _, _ in results if r["passes_verification"])
    print(f"\nPassing configs: {n_passing} / {len(results)}")

    # ==================================================================
    # Step 7: Visualizations for top 5 configs
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 7: Visualizations for top 5 configs")
    print("=" * 60)

    for rank, (score, row, detail, labels, perp) in enumerate(results[:5]):
        if "stats" not in detail or not detail["stats"]:
            print(f"  Rank {rank+1}: skipping (no clusters)")
            continue
        emb = tsne_cache[perp]
        config_str = (f"PCA20 + T-SNE3D(perp={perp}) + HDBSCAN(mcs={row['k_or_mcs']}, "
                      f"ms={row['min_samples']})")
        prefix = f"rank{rank+1}_p{perp}_mcs{row['k_or_mcs']}_ms{row['min_samples']}"

        # 3D scatter
        scatter_path = os.path.join(OUTPUT_DIR, f"{prefix}_tsne_3d_scatter.png")
        plot_3d_clusters(emb, labels, config_str, scatter_path)

        # Feature heatmap
        heatmap_path = os.path.join(OUTPUT_DIR, f"{prefix}_cluster_features.png")
        plot_feature_heatmap(detail["stats"], heatmap_path,
                             title=f"Rank {rank+1}: {config_str}")

        # Summary text
        summary_path = os.path.join(OUTPUT_DIR, f"{prefix}_summary.txt")
        write_summary_txt(detail, row, summary_path, N)

        # Save labels and metadata
        np.save(os.path.join(OUTPUT_DIR, f"{prefix}_cluster_labels.npy"), labels)
        for key in metadata:
            np.save(os.path.join(OUTPUT_DIR, f"{prefix}_{key}.npy"), metadata[key])

        # Transition analysis
        tr_rate, tr_matrix, total_tr, total_steps = compute_transition_rates(
            labels, metadata, N)
        with open(os.path.join(OUTPUT_DIR, f"{prefix}_transitions.txt"), "w") as f:
            f.write(f"Config: {config_str}\n")
            f.write(f"Transition rate: {tr_rate:.4f} ({total_tr}/{total_steps} steps)\n\n")
            f.write("Transition matrix:\n")
            for from_c in sorted(tr_matrix.keys()):
                for to_c in sorted(tr_matrix[from_c].keys()):
                    f.write(f"  C{from_c} -> C{to_c}: {tr_matrix[from_c][to_c]}\n")
        print(f"  Rank {rank+1}: transition rate = {tr_rate:.4f}")

    # ==================================================================
    # Step 8: Comparison plot (best per perplexity)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 8: Best-per-perplexity comparison")
    print("=" * 60)

    best_per_perp = {}
    for score, row, detail, labels, perp in results:
        if perp not in best_per_perp or score > best_per_perp[perp][0]:
            best_per_perp[perp] = (score, row, detail, labels, perp)

    embeddings_list = []
    labels_list = []
    configs_dict = {}
    for perp in sorted(best_per_perp.keys()):
        score, row, detail, labels, p = best_per_perp[perp]
        embeddings_list.append(tsne_cache[perp])
        labels_list.append(labels)
        configs_dict[perp] = (f"mcs={row['k_or_mcs']}, ms={row['min_samples']}, "
                              f"score={score:.3f}")

    comparison_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plot_comparison(embeddings_list, labels_list, configs_dict, comparison_path)

    # ==================================================================
    # Final summary
    # ==================================================================
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total data points: {N}")
    print(f"Activation dim: {activations.shape[1]}")
    print(f"PCA variance explained: {explained:.1%}")
    print(f"Configs tested: {len(results)}")
    print(f"Passing verification: {n_passing}")
    if results:
        best = results[0]
        print(f"Best score: {best[0]:.4f}")
        print(f"  Config: perp={best[4]}, mcs={best[1]['k_or_mcs']}, ms={best[1]['min_samples']}")
        print(f"  Clusters: {best[2]['n_clusters']}, Noise: {best[2]['noise_pct']:.1f}%")
        print(f"  Weighted d: {best[2]['mean_weighted_d']:.3f}")
        print(f"  Passes: {best[2]['passes_verification']}")
    print(f"Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
