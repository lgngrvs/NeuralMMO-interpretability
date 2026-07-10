#!/usr/bin/env python
"""Per-cluster T-SNE analysis: embed each cluster in isolation to reveal internal structure.

For both LSTM cell-state and action-decoder representations, isolates each cluster
(0, 1, 2) and runs T-SNE 2D/3D to visualize internal structure, colored by the
top-varying behavioral features within that cluster.

Run:
    cd /workspace/NeuralMMO-interpretability && \
    OMP_NUM_THREADS=4 uv run python experiments/per_cluster_tsne/run_analysis.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Make scripts/ importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from analyze_activations import FEATURE_NAMES, compute_cluster_stats  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PCA_DIMS = 30
MAX_POINTS = 3000
N_JOBS = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def subsample_to_n(arrays, n, rng):
    """Subsample a list of arrays (same first dim) to at most n rows."""
    total = arrays[0].shape[0]
    if total <= n:
        return arrays
    idx = rng.choice(total, size=n, replace=False)
    idx.sort()
    return [a[idx] for a in arrays]


def top_varying_features(feats, k=6):
    """Return indices of the k features with highest std across rows."""
    stds = np.std(feats, axis=0)
    return np.argsort(stds)[::-1][:k]


def top_cohens_d_feature(feats, labels, cluster_id):
    """Return the feature index with highest |Cohen's d| for cluster_id vs rest."""
    mask = labels == cluster_id
    if mask.sum() < 2 or (~mask).sum() < 2:
        return 0
    cluster_feats = feats[mask]
    rest_feats = feats[~mask]
    c_mean = cluster_feats.mean(axis=0)
    r_mean = rest_feats.mean(axis=0)
    c_std = cluster_feats.std(axis=0)
    r_std = rest_feats.std(axis=0)
    pooled = np.sqrt((c_std**2 + r_std**2) / 2)
    d = np.where(pooled > 1e-8, (c_mean - r_mean) / pooled, 0.0)
    return int(np.argmax(np.abs(d)))


def run_tsne(pca_data, n_components, seed=42):
    """Run T-SNE with adaptive perplexity."""
    n = pca_data.shape[0]
    perp = min(50, n // 5)
    perp = max(5, perp)
    print(f"    T-SNE {n_components}D: n={n}, perplexity={perp}")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perp,
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    return tsne.fit_transform(pca_data)


def plot_2d_grid(embedding, feats, top_feat_idx, title_prefix, output_path):
    """2x3 grid of 2D T-SNE scatter, each colored by one feature."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, fidx in enumerate(top_feat_idx[:6]):
        ax = axes[i]
        vals = feats[:, fidx]
        vmin, vmax = np.percentile(vals, [2, 98])
        if vmin == vmax:
            vmin, vmax = vals.min(), vals.max()
        if vmin == vmax:
            vmax = vmin + 1
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=vals,
            cmap="viridis",
            s=4,
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(f"{FEATURE_NAMES[fidx]} (std={np.std(vals):.2f})", fontsize=11)
        ax.set_xlabel("T-SNE 1", fontsize=8)
        ax.set_ylabel("T-SNE 2", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title_prefix, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Wrote {output_path}")


def plot_3d_scatter(embedding, feats, feat_idx, title_prefix, output_path):
    """Single 3D scatter colored by one feature, shown from 2 angles."""
    fig = plt.figure(figsize=(16, 7))
    vals = feats[:, feat_idx]
    vmin, vmax = np.percentile(vals, [2, 98])
    if vmin == vmax:
        vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        vmax = vmin + 1

    for pi, (elev, azim) in enumerate([(30, 45), (30, 135)]):
        ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=vals,
            cmap="viridis",
            s=4,
            alpha=0.6,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("T-SNE 1", fontsize=8)
        ax.set_ylabel("T-SNE 2", fontsize=8)
        ax.set_zlabel("T-SNE 3", fontsize=8)
        ax.set_title(f"elev={elev}, azim={azim}", fontsize=10)
        ax.tick_params(labelsize=7)
    fig.colorbar(sc, ax=fig.axes, fraction=0.02, pad=0.08, label=FEATURE_NAMES[feat_idx])
    fig.suptitle(
        f"{title_prefix}\nColored by: {FEATURE_NAMES[feat_idx]}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Wrote {output_path}")


def process_cluster(
    acts, feats, labels, cluster_id, repr_name, out_subdir, all_feats, all_labels
):
    """Process one cluster: subsample, PCA, T-SNE 2D/3D, plot."""
    rng = np.random.RandomState(SEED + cluster_id)
    mask = labels == cluster_id
    c_acts = acts[mask]
    c_feats = feats[mask]
    n_total = c_acts.shape[0]
    print(f"\n  Cluster {cluster_id}: {n_total} points")

    # Subsample
    [c_acts, c_feats] = subsample_to_n([c_acts, c_feats], MAX_POINTS, rng)
    n_used = c_acts.shape[0]
    print(f"    Using {n_used} points (subsampled={n_used < n_total})")

    # PCA
    pca_dims = min(PCA_DIMS, n_used - 1, c_acts.shape[1])
    pca = PCA(n_components=pca_dims, random_state=SEED)
    pca_data = pca.fit_transform(c_acts)
    expl = pca.explained_variance_ratio_.sum()
    print(f"    PCA {pca_dims}D explains {expl:.1%} variance")

    # Top-varying features (within this cluster)
    top6 = top_varying_features(c_feats, k=6)
    # Top discriminating feature (cluster vs rest) for summary
    top_discrim = top_cohens_d_feature(all_feats, all_labels, cluster_id)
    print(
        f"    Top-varying: {[FEATURE_NAMES[i] for i in top6]}"
    )
    print(f"    Top discriminating (Cohen's d): {FEATURE_NAMES[top_discrim]}")

    # T-SNE 2D
    emb_2d = run_tsne(pca_data, 2, seed=SEED)
    title_2d = f"{repr_name} — Cluster {cluster_id} (n={n_total}) — T-SNE 2D"
    path_2d = os.path.join(out_subdir, f"cluster_{cluster_id}_2d.png")
    plot_2d_grid(emb_2d, c_feats, top6, title_2d, path_2d)

    # T-SNE 3D
    emb_3d = run_tsne(pca_data, 3, seed=SEED)
    title_3d = f"{repr_name} — Cluster {cluster_id} (n={n_total}) — T-SNE 3D"
    path_3d = os.path.join(out_subdir, f"cluster_{cluster_id}_3d.png")
    plot_3d_scatter(emb_3d, c_feats, top6[0], title_3d, path_3d)

    return {
        "cluster_id": int(cluster_id),
        "n_total": int(n_total),
        "n_used": int(n_used),
        "pca_explained": float(expl),
        "top_varying_features": [FEATURE_NAMES[i] for i in top6],
        "top_discriminating_feature": FEATURE_NAMES[top_discrim],
        "top_discrim_idx": int(top_discrim),
        "emb_2d": emb_2d,
        "feats_sub": c_feats,
    }


def generate_action_decoder_clusters(acts, feats, steps, env_ids, agent_ids):
    """Generate cluster labels for action decoder activations.

    Pipeline: subsample rate 5 -> PCA(30) -> T-SNE 3D (perp=50) -> HDBSCAN(mcs=200,ms=3)
    -> KNN propagate to full resolution.

    Uses simple every-5th-record subsampling (matching experiments 2-3 config)
    and PCA(30) which reliably produces the classic 3-cluster structure.
    """
    import hdbscan as hdbscan_lib

    N = len(acts)
    print(f"  Action decoder: {N} total points")

    # Simple subsample at rate 5 (every 5th record, matching original experiments)
    sub_rate = 5
    sub_idx = np.arange(0, N, sub_rate)
    sub_acts = acts[sub_idx]
    n_sub = sub_acts.shape[0]
    print(f"  Subsampled: {n_sub} points (rate {sub_rate}, simple stride)")

    # PCA 30 (matches experiment 3 config that produced 3 clusters)
    pca = PCA(n_components=30, random_state=SEED)
    pca_sub = pca.fit_transform(sub_acts).astype(np.float32)
    print(f"  PCA 30D explains {pca.explained_variance_ratio_.sum():.1%} variance")

    # T-SNE 3D
    tsne = TSNE(
        n_components=3,
        perplexity=50,
        learning_rate="auto",
        max_iter=1000,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    emb = tsne.fit_transform(pca_sub)
    print(f"  T-SNE 3D done: {emb.shape}")

    # HDBSCAN
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=200, min_samples=3)
    sub_labels = clusterer.fit_predict(emb)
    cluster_ids = sorted(set(sub_labels.tolist()) - {-1})
    noise_pct = (sub_labels == -1).sum() / n_sub * 100
    print(f"  HDBSCAN: {len(cluster_ids)} clusters, {noise_pct:.1f}% noise")
    for cid in cluster_ids:
        print(f"    C{cid}: n={(sub_labels == cid).sum()}")

    # KNN propagate
    pca_full = pca.transform(acts).astype(np.float32)
    train_mask = sub_labels != -1
    knn = KNeighborsClassifier(
        n_neighbors=min(5, int(train_mask.sum())),
        n_jobs=N_JOBS,
    )
    knn.fit(pca_sub[train_mask], sub_labels[train_mask])
    full_labels = knn.predict(pca_full).astype(np.int32)
    print(f"  KNN propagated: {len(full_labels)} labels")
    for cid in cluster_ids:
        print(f"    C{cid} full: n={(full_labels == cid).sum()}")

    return full_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    results = {}

    # ===================================================================
    # LSTM cell-state
    # ===================================================================
    print("=" * 80)
    print("PART 1: LSTM CELL-STATE PER-CLUSTER T-SNE")
    print("=" * 80)

    lstm_npz = os.path.join(
        REPO_ROOT,
        "activation_data",
        "yaofeng_200M_lstm_cell",
        "yaofeng_200M",
        "activations.cache.npz",
    )
    cluster_labels_path = os.path.join(
        REPO_ROOT, "experiments", "yaofeng_lstm_cell_lifetime", "cluster_labels.npy"
    )

    print("\nLoading LSTM data...")
    data = np.load(lstm_npz)
    lstm_acts = data["activations"].astype(np.float32)
    lstm_feats = data["features"].astype(np.float32)
    lstm_steps = data["steps"].astype(np.int64)
    lstm_env_ids = data["env_ids"].astype(np.int64)
    lstm_agent_ids = data["agent_ids"].astype(np.int64)
    print(f"  Activations: {lstm_acts.shape}, Features: {lstm_feats.shape}")

    # Load pre-existing cluster labels
    print("Loading cluster labels...")
    label_arr = np.load(cluster_labels_path)  # [N, 4] = [env_id, agent_id, step, label]
    print(f"  Label array: {label_arr.shape}")

    # Join on (env_id, agent_id, step)
    # Both are from the same extraction run, so index-based join should work
    # Verify alignment
    match_count = np.sum(
        (label_arr[:, 0] == lstm_env_ids)
        & (label_arr[:, 1] == lstm_agent_ids)
        & (label_arr[:, 2] == lstm_steps)
    )
    print(f"  Direct index match: {match_count}/{len(lstm_acts)}")

    if match_count == len(lstm_acts):
        lstm_labels = label_arr[:, 3].astype(np.int32)
    else:
        # Fall back to key-based join
        print("  Using key-based join...")
        key_to_label = {}
        for i in range(len(label_arr)):
            key = (int(label_arr[i, 0]), int(label_arr[i, 1]), int(label_arr[i, 2]))
            key_to_label[key] = int(label_arr[i, 3])
        lstm_labels = np.array(
            [
                key_to_label.get(
                    (int(lstm_env_ids[i]), int(lstm_agent_ids[i]), int(lstm_steps[i])), -1
                )
                for i in range(len(lstm_acts))
            ],
            dtype=np.int32,
        )
        matched = (lstm_labels != -1).sum()
        print(f"  Key-based match: {matched}/{len(lstm_acts)}")

    lstm_cluster_ids = sorted(set(lstm_labels.tolist()) - {-1})
    print(f"  Cluster IDs: {lstm_cluster_ids}")
    for cid in lstm_cluster_ids:
        print(f"    C{cid}: n={(lstm_labels == cid).sum()}")

    lstm_out = os.path.join(OUT_DIR, "lstm_cell")
    os.makedirs(lstm_out, exist_ok=True)
    results["lstm_cell"] = {}

    for cid in lstm_cluster_ids:
        info = process_cluster(
            lstm_acts,
            lstm_feats,
            lstm_labels,
            cid,
            "LSTM Cell",
            lstm_out,
            lstm_feats,
            lstm_labels,
        )
        results["lstm_cell"][cid] = info

    # ===================================================================
    # ACTION DECODER
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 2: ACTION DECODER PER-CLUSTER T-SNE")
    print("=" * 80)

    ad_npz = os.path.join(
        REPO_ROOT, "activation_data", "yaofeng_200M", "activations.cache.npz"
    )
    print("\nLoading action decoder data...")
    data = np.load(ad_npz)
    ad_acts = data["activations"].astype(np.float32)
    ad_feats = data["features"].astype(np.float32)
    ad_steps = data["steps"].astype(np.int64)
    ad_env_ids = data["env_ids"].astype(np.int64)
    ad_agent_ids = data["agent_ids"].astype(np.int64)
    print(f"  Activations: {ad_acts.shape}, Features: {ad_feats.shape}")

    print("\nGenerating action decoder clusters...")
    ad_labels = generate_action_decoder_clusters(
        ad_acts, ad_feats, ad_steps, ad_env_ids, ad_agent_ids
    )
    ad_cluster_ids = sorted(set(ad_labels.tolist()) - {-1})
    print(f"  Cluster IDs: {ad_cluster_ids}")

    ad_out = os.path.join(OUT_DIR, "action_decoder")
    os.makedirs(ad_out, exist_ok=True)
    results["action_decoder"] = {}

    for cid in ad_cluster_ids:
        info = process_cluster(
            ad_acts,
            ad_feats,
            ad_labels,
            cid,
            "Action Decoder",
            ad_out,
            ad_feats,
            ad_labels,
        )
        results["action_decoder"][cid] = info

    # ===================================================================
    # SUMMARY PNG: 2x3 grid
    # ===================================================================
    print("\n" + "=" * 80)
    print("BUILDING SUMMARY")
    print("=" * 80)

    # We need exactly clusters 0,1,2 for each. Handle gracefully if different.
    target_clusters = [0, 1, 2]

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    for row_idx, (repr_name, repr_results, all_feats_arr, all_labels_arr) in enumerate(
        [
            ("LSTM Cell", results["lstm_cell"], lstm_feats, lstm_labels),
            ("Action Decoder", results["action_decoder"], ad_feats, ad_labels),
        ]
    ):
        for col_idx, cid in enumerate(target_clusters):
            ax = axes[row_idx, col_idx]
            if cid not in repr_results:
                ax.text(
                    0.5,
                    0.5,
                    f"Cluster {cid} not found",
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                ax.set_title(f"{repr_name} — C{cid}")
                ax.axis("off")
                continue

            info = repr_results[cid]
            emb = info["emb_2d"]
            feats_sub = info["feats_sub"]
            discrim_idx = info["top_discrim_idx"]
            discrim_name = info["top_discriminating_feature"]

            vals = feats_sub[:, discrim_idx]
            vmin, vmax = np.percentile(vals, [2, 98])
            if vmin == vmax:
                vmin, vmax = vals.min(), vals.max()
            if vmin == vmax:
                vmax = vmin + 1

            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=vals,
                cmap="viridis",
                s=3,
                alpha=0.6,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_title(
                f"{repr_name} C{cid} (n={info['n_total']})\n"
                f"color: {discrim_name}",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xlabel("T-SNE 1", fontsize=8)
            ax.set_ylabel("T-SNE 2", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Per-Cluster T-SNE: Internal Structure of Each Cluster\n"
        "Yaofeng 200M — LSTM Cell-State (top) vs Action Decoder (bottom)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    summary_path = os.path.join(OUT_DIR, "summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {summary_path}")

    # ===================================================================
    # Save stats JSON
    # ===================================================================
    elapsed = time.time() - t0
    stats_out = {}
    for repr_name, repr_results in results.items():
        stats_out[repr_name] = {}
        for cid, info in repr_results.items():
            stats_out[repr_name][str(cid)] = {
                "n_total": info["n_total"],
                "n_used": info["n_used"],
                "pca_explained": info["pca_explained"],
                "top_varying_features": info["top_varying_features"],
                "top_discriminating_feature": info["top_discriminating_feature"],
            }
    stats_out["elapsed_sec"] = elapsed

    stats_path = os.path.join(OUT_DIR, "run_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"  Wrote {stats_path}")

    # ===================================================================
    # Print summary for LOG
    # ===================================================================
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    for repr_name, repr_results in results.items():
        print(f"\n  {repr_name.upper()}:")
        for cid in sorted(repr_results.keys()):
            info = repr_results[cid]
            print(
                f"    C{cid}: n={info['n_total']}, "
                f"PCA var={info['pca_explained']:.1%}, "
                f"top-varying={info['top_varying_features'][:3]}, "
                f"discriminating={info['top_discriminating_feature']}"
            )

    print(f"\nTotal time: {elapsed:.0f}s")
    print(f"\nOutput: {OUT_DIR}/")


if __name__ == "__main__":
    main()
