#!/usr/bin/env python
"""T-SNE 3D + HDBSCAN analysis on ALL available Takeru 200M activation data.

Combines takeru_200M + takeru_200M_extra + takeru_200M_batch_3..9 (~35GB of
raw data, plus ~30GB of raw jsonl in batch_8+9). Subsamples per trajectory at
rate=100, runs PCA(20) -> T-SNE 3D (sweep perplexities) -> HDBSCAN (sweep
min_cluster_size / min_samples), scores each combo with score_combo, and
produces visualizations + transition-rate analysis for the best configs.

Hypothesis: prior takeru clustering runs failed to find stable clusters due
to insufficient data. With ~4x more data, we may expose real behavioral modes.
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
import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from analyze_activations import (  # noqa: E402
    FEATURE_NAMES,
    compute_cluster_stats,  # noqa: F401
    compute_features,
    load_data,
)
from sweep_cluster_params import score_combo  # noqa: E402

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PERPLEXITIES = [50, 75, 100, 125, 150]
MCS_VALUES = [50, 100, 200, 400, 800]
MS_VALUES = [3, 5, 10, 25]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path, subsample_rate=100):
    """Load one dataset directory. Prefers load_data (which handles source json +
    cache + subsampling), falls back to direct cache load if the source is gone.
    """
    try:
        records, _ = load_data(path, subsample_rate=subsample_rate)
        act, feat, meta = compute_features(records)
        return act, feat, meta
    except FileNotFoundError:
        pass

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
        print(f"  {keep_mask.sum()} records after subsampling (rate={subsample_rate})",
              flush=True)

    meta = {"step": steps, "env_id": env_ids, "agent_id": agent_ids}
    return acts, feats, meta


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
VIEWING_ANGLES = [(30, 45), (30, 135), (30, 225), (60, 0)]


def plot_3d_clusters(embedding, labels, config_str, output_path):
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
    cluster_ids = sorted(stats.keys())
    if not cluster_ids:
        return
    n_feats = len(FEATURE_NAMES)
    d_matrix = np.array([stats[c]["cohens_d"] for c in cluster_ids])

    fig, ax = plt.subplots(figsize=(max(12, n_feats * 0.8),
                                    max(4, len(cluster_ids) * 0.6)))
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
    stats = detail["stats"]
    with open(output_path, "w") as f:
        f.write("Strategy: T-SNE 3D + HDBSCAN (ALL Takeru 200M data: base + extra + batch_3..9)\n")
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
        f.write(f"\n{'=' * 80}\n")
        for label in sorted(stats.keys()):
            s = stats[label]
            f.write(f"\nCluster {label} (n={s['size']}, {s['size'] / N * 100:.1f}%):\n")
            f.write(f"  Step entropy: {s['step_entropy']:.3f}\n")
            f.write(f"  Agent entropy: {s['agent_entropy']:.3f}\n")
            f.write("  Top features:\n")
            for rank, idx in enumerate(s["top_features"][:5]):
                d = s["cohens_d"][idx]
                f.write(f"    {rank + 1}. {FEATURE_NAMES[idx]:25s} d={d:+.2f}\n")
    print(f"  Wrote {output_path}")


def compute_transition_rates(labels, metadata):
    """Fraction of consecutive within-trajectory timesteps where label changes
    (excluding noise-labeled endpoints).
    """
    traj_keys = (metadata["env_id"].astype(np.int64) * 1_000_000
                 + metadata["agent_id"].astype(np.int64))
    unique_trajs = np.unique(traj_keys)

    total_transitions = 0
    total_steps = 0
    cluster_transitions = {}

    for key in unique_trajs:
        mask = traj_keys == key
        indices = np.where(mask)[0]
        sorted_order = np.argsort(metadata["step"][indices])
        sorted_indices = indices[sorted_order]
        traj_labels = labels[sorted_indices]

        for i in range(1, len(traj_labels)):
            if traj_labels[i] == -1 or traj_labels[i - 1] == -1:
                continue
            total_steps += 1
            if traj_labels[i] != traj_labels[i - 1]:
                total_transitions += 1
                from_c = int(traj_labels[i - 1])
                to_c = int(traj_labels[i])
                cluster_transitions.setdefault(from_c, {})
                cluster_transitions[from_c][to_c] = cluster_transitions[from_c].get(to_c, 0) + 1

    transition_rate = total_transitions / max(total_steps, 1)
    return transition_rate, cluster_transitions, total_transitions, total_steps


def plot_comparison(embeddings_by_perp, labels_by_perp, configs_by_perp, output_path):
    n = len(embeddings_by_perp)
    if n == 0:
        return

    fig = plt.figure(figsize=(5.5 * n, 6))
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
        ax.set_title(f"perp={perp}\n{cfg_str}\n{n_clusters} cl, {noise_pct:.0f}% noise",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        if n_clusters <= 10:
            ax.legend(markerscale=3, fontsize=6, loc="best", framealpha=0.8)

    fig.suptitle("Takeru 200M (all batches): Best per Perplexity", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


def plot_summary(best_detail, best_row, best_emb, best_labels, score_per_perp,
                 transition_rate, N, output_path):
    """2x2 summary: best 3D scatter, Cohen's d heatmap, score-vs-perplexity, text."""
    fig = plt.figure(figsize=(16, 13))

    # Top-left: best 3D scatter
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    cluster_ids = sorted(set(best_labels) - {-1})
    n_clusters = len(cluster_ids)
    noise_pct = (best_labels == -1).sum() / len(best_labels) * 100
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))
    noise_mask = best_labels == -1
    if noise_mask.any():
        ax.scatter(best_emb[noise_mask, 0], best_emb[noise_mask, 1], best_emb[noise_mask, 2],
                   c="lightgray", s=1, alpha=0.3, rasterized=True)
    for i, cl in enumerate(cluster_ids):
        mask = best_labels == cl
        ax.scatter(best_emb[mask, 0], best_emb[mask, 1], best_emb[mask, 2],
                   c=[cmap(i)], s=3, alpha=0.5, label=f"C{cl} (n={mask.sum()})",
                   rasterized=True)
    ax.view_init(elev=30, azim=45)
    ax.set_xlabel("T-SNE 1", fontsize=8)
    ax.set_ylabel("T-SNE 2", fontsize=8)
    ax.set_zlabel("T-SNE 3", fontsize=8)
    ax.set_title(f"Best: perp={best_row['tsne_perplexity']}, "
                 f"mcs={best_row['k_or_mcs']}, ms={best_row['min_samples']}", fontsize=11)
    if n_clusters <= 10:
        ax.legend(markerscale=3, fontsize=7, loc="best", framealpha=0.85)

    # Top-right: Cohen's d heatmap for best
    stats = best_detail["stats"]
    ax = fig.add_subplot(2, 2, 2)
    cids = sorted(stats.keys())
    if cids:
        d_mat = np.array([stats[c]["cohens_d"] for c in cids])
        im = ax.imshow(d_mat, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_xticks(range(len(FEATURE_NAMES)))
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(cids)))
        ax.set_yticklabels([f"C{c} (n={stats[c]['size']})" for c in cids], fontsize=7)
        fig.colorbar(im, ax=ax, label="Cohen's d", fraction=0.03)
    ax.set_title("Best config — Cohen's d per feature", fontsize=11)

    # Bottom-left: score vs perplexity curve
    ax = fig.add_subplot(2, 2, 3)
    perps_sorted = sorted(score_per_perp.keys())
    scores_sorted = [score_per_perp[p] for p in perps_sorted]
    ax.plot(perps_sorted, scores_sorted, marker="o", linewidth=2, color="tab:blue")
    ax.set_xlabel("T-SNE perplexity", fontsize=10)
    ax.set_ylabel("Best composite score @ this perplexity", fontsize=10)
    ax.set_title("Score vs. perplexity", fontsize=11)
    ax.grid(True, alpha=0.3)
    for p, s in zip(perps_sorted, scores_sorted):
        ax.annotate(f"{s:.3f}", (p, s), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)

    # Bottom-right: text summary
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    text = (
        f"T-SNE 3D + HDBSCAN clustering\n"
        f"Takeru 200M: ALL batches (base + extra + batch_3..9)\n"
        f"\n"
        f"Total points (subsample=100): {N:,}\n"
        f"Trajectories: see LOG.md\n"
        f"\n"
        f"BEST CONFIG\n"
        f"  perplexity       : {best_row['tsne_perplexity']}\n"
        f"  min_cluster_size : {best_row['k_or_mcs']}\n"
        f"  min_samples      : {best_row['min_samples']}\n"
        f"\n"
        f"METRICS\n"
        f"  composite score  : {best_detail['composite_score']:.4f}\n"
        f"  #clusters        : {best_detail['n_clusters']} "
        f"(raw {best_detail['n_clusters_raw']})\n"
        f"  noise %          : {best_detail['noise_pct']:.1f}%\n"
        f"  weighted |d|     : {best_detail['mean_weighted_d']:.3f}\n"
        f"  min step entropy : {best_detail['min_step_entropy']:.3f}\n"
        f"  min agent entropy: {best_detail['min_agent_entropy']:.3f}\n"
        f"  passes verif.    : {best_detail['passes_verification']}\n"
        f"\n"
        f"TRANSITION RATE (best config)\n"
        f"  {transition_rate:.4f}\n"
        f"  Prior takeru runs: ~0.96\n"
    )
    ax.text(0.02, 0.98, text, family="monospace", fontsize=10,
            va="top", ha="left", transform=ax.transAxes)

    fig.suptitle("Takeru 200M all-data T-SNE 3D + HDBSCAN — Summary", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load all datasets
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Step 1: Loading ALL takeru_200M datasets at subsample=100")
    print("=" * 70)

    base_dir = str(Path(__file__).resolve().parent.parent.parent)
    datasets = [
        os.path.join(base_dir, "activation_data/takeru_200M/"),
        os.path.join(base_dir, "activation_data/takeru_200M_extra/takeru_200M/"),
        os.path.join(base_dir, "activation_data/takeru_200M_batch_3/takeru_200M/"),
        os.path.join(base_dir, "activation_data/takeru_200M_batch_4/takeru_200M/"),
        os.path.join(base_dir, "activation_data/takeru_200M_batch_5/takeru_200M/"),
        os.path.join(base_dir, "activation_data/takeru_200M_batch_6/takeru_200M/"),
        os.path.join(base_dir, "activation_data/takeru_200M_batch_7/takeru_200M/"),
        os.path.join(base_dir, "activation_data/takeru_200M_batch_8/takeru_200M/"),
        os.path.join(base_dir, "activation_data/takeru_200M_batch_9/takeru_200M/"),
    ]

    all_acts, all_feats, all_metas = [], [], []
    per_dataset_counts = {}
    for path in datasets:
        try:
            print(f"\n--- Loading {path} ---")
            act, feat, meta = load_dataset(path, subsample_rate=100)
            all_acts.append(act)
            all_feats.append(feat)
            all_metas.append(meta)
            per_dataset_counts[path] = act.shape[0]
            gc.collect()
            print(f"  -> {act.shape[0]} points after subsampling")
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            per_dataset_counts[path] = 0

    if not all_acts:
        print("ERROR: No data loaded!")
        return

    # To prevent cross-dataset collisions on (env_id, agent_id) keys, offset
    # env_ids by dataset index so transitions won't accidentally bridge datasets.
    offset = 100_000
    for i, m in enumerate(all_metas):
        m["env_id"] = m["env_id"].astype(np.int64) + i * offset

    activations = np.vstack(all_acts).astype(np.float32)
    features = np.vstack(all_feats)
    metadata = {k: np.concatenate([m[k] for m in all_metas]) for k in all_metas[0]}
    N = len(activations)
    print(f"\nCombined: {activations.shape} ({N} points, {activations.shape[1]}D)")
    print("Per-dataset counts:")
    for path, n in per_dataset_counts.items():
        print(f"  {n:>7d}  {Path(path).parts[-2] if 'takeru_200M' not in Path(path).name else Path(path).parts[-3]}")
    n_trajs = len(np.unique(metadata["env_id"].astype(np.int64) * 1_000_000
                            + metadata["agent_id"].astype(np.int64)))
    print(f"Unique trajectories: {n_trajs}")

    del all_acts, all_feats, all_metas
    gc.collect()

    min_trustworthy = max(20, int(0.02 * N))
    print(f"\nMin trustworthy cluster size: {min_trustworthy} (2% of {N})")

    # ------------------------------------------------------------------
    # Step 2: PCA
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2: PCA reduction to 20D")
    print("=" * 70)
    pca = PCA(n_components=20, random_state=SEED)
    pca_20 = pca.fit_transform(activations)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA 20D: {explained:.1%} variance explained")
    # free the full-dim activations; we only need PCA for T-SNE
    del activations
    gc.collect()

    # ------------------------------------------------------------------
    # Step 3: T-SNE 3D (parallel per perplexity)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Step 3: T-SNE 3D embeddings (parallel across perplexities {PERPLEXITIES})")
    print("=" * 70)

    tsne_cache = {}

    def _fit_tsne(perp):
        from sklearn.manifold import TSNE as _TSNE
        t0 = time.time()
        print(f"  Starting T-SNE perplexity={perp}...", flush=True)
        t = _TSNE(n_components=3, perplexity=perp, learning_rate="auto",
                  max_iter=1000, random_state=SEED, verbose=0)
        result = t.fit_transform(pca_20)
        print(f"  Done T-SNE perplexity={perp} in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        return perp, result

    n_workers = min(len(PERPLEXITIES), 5)
    print(f"Running {len(PERPLEXITIES)} T-SNE embeddings with {n_workers} workers "
          f"(4 BLAS threads each = ~20 threads total)...")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fit_tsne, perp): perp for perp in PERPLEXITIES}
        for fut in as_completed(futures):
            perp, emb = fut.result()
            tsne_cache[perp] = emb

    print(f"All T-SNE embeddings computed: {sorted(tsne_cache.keys())}")

    # ------------------------------------------------------------------
    # Step 4: HDBSCAN sweep
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: HDBSCAN sweep")
    print("=" * 70)
    grid = [(p, mcs, ms) for p, mcs, ms in product(PERPLEXITIES, MCS_VALUES, MS_VALUES)
            if ms <= mcs]
    print(f"Sweeping {len(grid)} combos")

    results = []
    for perp, mcs, ms in grid:
        emb = tsne_cache[perp]
        clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                        core_dist_n_jobs=4)
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
        status = "PASS" if detail["passes_verification"] else "    "
        print(f"  [{status}] perp={perp} mcs={mcs:4d} ms={ms:3d}: "
              f"score={score:.3f} cl={detail['n_clusters']:2d} "
              f"noise={detail['noise_pct']:5.1f}% d={detail['mean_weighted_d']:.2f}")

    results.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # Step 5: Write sweep CSV
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 6: Console top-20
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Top 20 configs:")
    print("=" * 70)
    print(f"{'perp':>5} {'mcs':>5} {'ms':>3} {'cl':>3} {'noise%':>7} "
          f"{'wt_d':>6} {'ent':>5} {'score':>7} {'pass':>5}")
    for score, row, detail, _, _ in results[:20]:
        print(f"{row['tsne_perplexity']:5} {row['k_or_mcs']:5d} {row['min_samples']:3d} "
              f"{row['n_clusters']:3d} {row['noise_pct']:6.1f}% "
              f"{row['mean_weighted_d']:6.3f} {row['entropy_factor']:5.3f} "
              f"{score:7.3f} {'YES' if row['passes_verification'] else ' no'}")
    n_passing = sum(1 for _, r, _, _, _ in results if r["passes_verification"])
    print(f"\nPassing configs: {n_passing} / {len(results)}")

    # ------------------------------------------------------------------
    # Step 7: Top-3 detailed outputs (+ keep #1 transition rate)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 7: Detailed visualizations for top 3 configs")
    print("=" * 70)
    top_transition_rate = None
    top_tr_matrix = None

    for rank, (score, row, detail, labels, perp) in enumerate(results[:3]):
        config_dir = os.path.join(OUTPUT_DIR, f"config_{rank + 1}")
        os.makedirs(config_dir, exist_ok=True)
        if "stats" not in detail or not detail["stats"]:
            print(f"  Rank {rank + 1}: no trustworthy clusters, skipping plots")
            with open(os.path.join(config_dir, "summary.txt"), "w") as f:
                f.write(f"rank={rank + 1} perp={perp} mcs={row['k_or_mcs']} "
                        f"ms={row['min_samples']} -- no clusters survived\n")
            continue
        emb = tsne_cache[perp]
        config_str = (f"PCA20 + T-SNE3D(perp={perp}) + "
                      f"HDBSCAN(mcs={row['k_or_mcs']}, ms={row['min_samples']})")

        plot_3d_clusters(emb, labels, config_str,
                         os.path.join(config_dir, "tsne_3d_scatter.png"))
        plot_feature_heatmap(detail["stats"],
                             os.path.join(config_dir, "cluster_features.png"),
                             title=f"Rank {rank + 1}: {config_str}")
        write_summary_txt(detail, row, os.path.join(config_dir, "summary.txt"), N)

        # Cluster labels with metadata rows: [env_id, agent_id, step, label]
        labels_with_meta = np.stack([
            metadata["env_id"].astype(np.int64),
            metadata["agent_id"].astype(np.int64),
            metadata["step"].astype(np.int64),
            labels.astype(np.int64),
        ], axis=1)
        np.save(os.path.join(config_dir, "cluster_labels.npy"), labels_with_meta)

        tr_rate, tr_matrix, total_tr, total_steps = compute_transition_rates(
            labels, metadata)
        with open(os.path.join(config_dir, "transitions.txt"), "w") as f:
            f.write(f"Config: {config_str}\n")
            f.write(f"Transition rate: {tr_rate:.4f} "
                    f"({total_tr}/{total_steps} within-trajectory steps)\n\n")
            f.write("Transition matrix:\n")
            for from_c in sorted(tr_matrix.keys()):
                for to_c in sorted(tr_matrix[from_c].keys()):
                    f.write(f"  C{from_c} -> C{to_c}: {tr_matrix[from_c][to_c]}\n")
        print(f"  Rank {rank + 1}: transition rate = {tr_rate:.4f}")
        if rank == 0:
            top_transition_rate = tr_rate
            top_tr_matrix = tr_matrix

    # ------------------------------------------------------------------
    # Step 8: Best-per-perplexity comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 8: Best-per-perplexity comparison")
    print("=" * 70)

    best_per_perp = {}
    for score, row, detail, labels, perp in results:
        if perp not in best_per_perp or score > best_per_perp[perp][0]:
            best_per_perp[perp] = (score, row, detail, labels, perp)

    embeddings_list = []
    labels_list = []
    configs_dict = {}
    score_per_perp = {}
    for perp in sorted(best_per_perp.keys()):
        score, row, detail, labels, p = best_per_perp[perp]
        embeddings_list.append(tsne_cache[perp])
        labels_list.append(labels)
        configs_dict[perp] = (f"mcs={row['k_or_mcs']}, ms={row['min_samples']}, "
                              f"score={score:.3f}")
        score_per_perp[perp] = score

    comparison_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plot_comparison(embeddings_list, labels_list, configs_dict, comparison_path)

    # ------------------------------------------------------------------
    # Step 9: summary.png
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 9: Writing summary.png")
    print("=" * 70)
    best = results[0]
    best_score, best_row, best_detail, best_labels, best_perp = best
    plot_summary(best_detail, best_row, tsne_cache[best_perp], best_labels,
                 score_per_perp,
                 top_transition_rate if top_transition_rate is not None else float("nan"),
                 N,
                 os.path.join(OUTPUT_DIR, "summary.png"))

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total data points: {N}")
    print(f"Unique trajectories: {n_trajs}")
    print(f"PCA variance explained: {explained:.1%}")
    print(f"Configs tested: {len(results)}")
    print(f"Passing verification: {n_passing}")
    if results:
        print(f"Best score: {best_score:.4f}")
        print(f"  Config: perp={best_perp}, mcs={best_row['k_or_mcs']}, "
              f"ms={best_row['min_samples']}")
        print(f"  Clusters: {best_detail['n_clusters']}, "
              f"Noise: {best_detail['noise_pct']:.1f}%")
        print(f"  Weighted d: {best_detail['mean_weighted_d']:.3f}")
        print(f"  Passes: {best_detail['passes_verification']}")
        if top_transition_rate is not None:
            print(f"  Transition rate (#1 config): {top_transition_rate:.4f}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
