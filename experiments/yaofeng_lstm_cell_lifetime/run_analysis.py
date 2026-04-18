#!/usr/bin/env python
"""Yaofeng 200M LSTM cell-state lifetime clustering analysis.

Re-runs the cluster analysis at the user-specified config, KNN-propagates labels
to full-resolution, and produces per-agent lifetime plots overlaying the cluster
labels.

Config:
  Data:         activation_data/yaofeng_200M_lstm_cell/yaofeng_200M/activations.cache.npz
  Subsample:    50 per trajectory (sort by step within (env_id,agent_id))
  PCA:          20 dims (random_state=42)
  T-SNE:        3D, perplexity=50, max_iter=1000, learning_rate='auto', random_state=42
  HDBSCAN:      min_cluster_size=200, min_samples=3
  KNN prop:     k=5 on PCA-20 space, -1 (noise) points excluded from training.

Run:
  cd /workspace/NeuralMMO-interpretability && \\
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    uv run python experiments/yaofeng_lstm_cell_lifetime/run_analysis.py
"""

import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import hdbscan as hdbscan_lib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Make the scripts/ package importable for FEATURE_NAMES + compute_cluster_stats.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from analyze_activations import FEATURE_NAMES, compute_cluster_stats  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NPZ_PATH = os.path.join(
    REPO_ROOT,
    "activation_data",
    "yaofeng_200M_lstm_cell",
    "yaofeng_200M",
    "activations.cache.npz",
)
LIFE_DATA_DIR = os.path.join(REPO_ROOT, "activation_data", "yaofeng_200M")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PER_AGENT_DIR = os.path.join(OUT_DIR, "per_agent")
os.makedirs(PER_AGENT_DIR, exist_ok=True)

SUBSAMPLE_RATE = 50
PCA_DIMS = 20
PERPLEXITY = 50
TSNE_MAX_ITER = 1000
HDBSCAN_MCS = 200
HDBSCAN_MS = 3
SEED = 42
KNN_K = 5
NUM_AGENTS = 5

VIEWING_ANGLES = [(30, 45), (30, 135), (30, 225), (60, 0)]


# ---------------------------------------------------------------------------
# Plotting helpers (adapted from experiments/T-SNE-3d-yaofeng-lstm-cell-sparse)
# ---------------------------------------------------------------------------
def plot_3d_scatter(embedding, labels, title_str, output_path):
    cluster_ids = sorted(set(labels.tolist()) - {-1})
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
                embedding[noise_mask, 0], embedding[noise_mask, 1], embedding[noise_mask, 2],
                c="lightgray", s=1, alpha=0.3, label="noise", rasterized=True,
            )
        for i, cl in enumerate(cluster_ids):
            mask = labels == cl
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
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
        f"{title_str}\n{n_clusters} clusters, {noise_pct:.1f}% noise",
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


# ---------------------------------------------------------------------------
# Transition rate (for full-resolution, per-trajectory adjacent steps)
# ---------------------------------------------------------------------------
def compute_transition_rate_fullres(labels, env_ids, agent_ids, steps):
    """Compute fraction of consecutive timesteps where the dense cluster label
    changes. Excludes noise->noise and noise transitions by filtering them.
    Operates on FULL-resolution (post-KNN) dense labels.
    """
    traj_keys = env_ids.astype(np.int64) * 1_000_000 + agent_ids.astype(np.int64)
    unique_keys = np.unique(traj_keys)

    total_pairs = 0
    total_trans = 0
    for key in unique_keys:
        mask = traj_keys == key
        idx = np.where(mask)[0]
        if len(idx) < 2:
            continue
        order = np.argsort(steps[idx])
        idx_sorted = idx[order]
        lbl = labels[idx_sorted]
        for j in range(len(lbl) - 1):
            if lbl[j] == -1 or lbl[j + 1] == -1:
                continue
            total_pairs += 1
            if lbl[j] != lbl[j + 1]:
                total_trans += 1
    if total_pairs == 0:
        return 0.0
    return total_trans / total_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    print("=" * 80)
    print("YAOFENG 200M LSTM CELL-STATE LIFETIME ANALYSIS")
    print(f"  data: {NPZ_PATH}")
    print(f"  subsample: {SUBSAMPLE_RATE} / traj")
    print(f"  PCA: {PCA_DIMS}, TSNE: perp={PERPLEXITY}, iters={TSNE_MAX_ITER}")
    print(f"  HDBSCAN: mcs={HDBSCAN_MCS}, ms={HDBSCAN_MS}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load npz cache directly (no activations.json in lstm_cell dir)
    # ------------------------------------------------------------------
    print("\n[1/9] Loading npz cache...", flush=True)
    data = np.load(NPZ_PATH)
    acts = data["activations"].astype(np.float32)
    feats = data["features"].astype(np.float32)
    steps = data["steps"].astype(np.int64)
    env_ids = data["env_ids"].astype(np.int64)
    agent_ids = data["agent_ids"].astype(np.int64)
    N_full = len(acts)
    print(f"  Full-res: {acts.shape}, {N_full} alive records")

    # ------------------------------------------------------------------
    # 2. Subsample per trajectory (every 50th after sort-by-step)
    # ------------------------------------------------------------------
    print(f"\n[2/9] Subsampling per trajectory at rate {SUBSAMPLE_RATE}...", flush=True)
    traj_keys_full = env_ids.astype(np.int64) * 1_000_000 + agent_ids.astype(np.int64)
    unique_keys = np.unique(traj_keys_full)
    keep_mask = np.zeros(N_full, dtype=bool)
    for key in unique_keys:
        mask = traj_keys_full == key
        idx = np.where(mask)[0]
        order = np.argsort(steps[idx])
        idx_sorted = idx[order]
        keep_mask[idx_sorted[::SUBSAMPLE_RATE]] = True

    sub_acts = acts[keep_mask]
    sub_feats = feats[keep_mask]
    sub_steps = steps[keep_mask]
    sub_env = env_ids[keep_mask]
    sub_agent = agent_ids[keep_mask]
    N = len(sub_acts)
    print(f"  Subsampled: {N} points ({N_full} -> {N})")

    metadata_sub = {"step": sub_steps.astype(int),
                    "env_id": sub_env.astype(int),
                    "agent_id": sub_agent.astype(int)}

    # ------------------------------------------------------------------
    # 3. PCA on subsampled activations (fit_transform)
    # ------------------------------------------------------------------
    print(f"\n[3/9] PCA ({PCA_DIMS}D)...", flush=True)
    pca = PCA(n_components=PCA_DIMS, random_state=SEED)
    pca_sub = pca.fit_transform(sub_acts).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA {PCA_DIMS}D explains {explained:.1%} variance")

    # ------------------------------------------------------------------
    # 4. T-SNE 3D
    # ------------------------------------------------------------------
    print(f"\n[4/9] T-SNE 3D (perp={PERPLEXITY}, iters={TSNE_MAX_ITER})...", flush=True)
    tsne = TSNE(
        n_components=3,
        perplexity=PERPLEXITY,
        learning_rate="auto",
        max_iter=TSNE_MAX_ITER,
        random_state=SEED,
        verbose=1,
    )
    tsne_emb = tsne.fit_transform(pca_sub)
    print(f"  T-SNE embedding shape: {tsne_emb.shape}")

    # ------------------------------------------------------------------
    # 5. HDBSCAN on T-SNE embedding
    # ------------------------------------------------------------------
    print(f"\n[5/9] HDBSCAN (mcs={HDBSCAN_MCS}, ms={HDBSCAN_MS})...", flush=True)
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=HDBSCAN_MCS, min_samples=HDBSCAN_MS
    )
    sub_labels = clusterer.fit_predict(tsne_emb)
    unique_sub = sorted(set(sub_labels.tolist()))
    cluster_ids = [c for c in unique_sub if c != -1]
    noise_pct_sub = (sub_labels == -1).sum() / len(sub_labels) * 100
    print(f"  Clusters: {len(cluster_ids)} ({cluster_ids})")
    print(f"  Noise: {noise_pct_sub:.1f}% ({(sub_labels == -1).sum()}/{N})")
    for cid in cluster_ids:
        n_cid = (sub_labels == cid).sum()
        print(f"    C{cid}: n={n_cid} ({n_cid / N * 100:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Cohen's d per cluster per feature
    # ------------------------------------------------------------------
    print(f"\n[6/9] Cohen's d per cluster...", flush=True)
    stats = compute_cluster_stats(sub_feats, sub_labels, metadata_sub,
                                  n_baseline_samples=100, rng_seed=SEED)
    for cid in cluster_ids:
        s = stats[cid]
        top3 = [(FEATURE_NAMES[idx], s["cohens_d"][idx]) for idx in s["top_features"][:3]]
        print(f"    C{cid} top: " +
              ", ".join(f"{n}={d:+.2f}" for n, d in top3))

    # ------------------------------------------------------------------
    # 7. KNN-propagate cluster labels to full resolution
    # ------------------------------------------------------------------
    print(f"\n[7/9] KNN propagation (k={KNN_K}) to full resolution...", flush=True)
    pca_full = pca.transform(acts).astype(np.float32)  # apply fit PCA to all

    # Train KNN only on non-noise subsampled points
    train_mask = sub_labels != -1
    if train_mask.sum() == 0:
        print("  WARNING: all subsampled points are noise, assigning -1 to all full-res points")
        full_labels = np.full(N_full, -1, dtype=np.int32)
    else:
        knn = KNeighborsClassifier(
            n_neighbors=min(KNN_K, int(train_mask.sum())),
            n_jobs=min(16, os.cpu_count() or 1),
        )
        knn.fit(pca_sub[train_mask], sub_labels[train_mask])
        full_labels = knn.predict(pca_full).astype(np.int32)

    # Sanity: dense full_labels should have no -1, but keep cast
    n_noise_full = (full_labels == -1).sum()
    print(f"  Full-res labels: shape={full_labels.shape}, "
          f"-1 count={n_noise_full} (expected 0 when KNN was trainable)")
    for cid in cluster_ids:
        n_cid = (full_labels == cid).sum()
        print(f"    C{cid} full: n={n_cid} ({n_cid / N_full * 100:.2f}%)")

    # Transition rate on full res
    trans_rate = compute_transition_rate_fullres(full_labels, env_ids, agent_ids, steps)
    print(f"  Full-res transition rate (tick-to-tick): {trans_rate:.4f}")

    # ------------------------------------------------------------------
    # 8. Save cluster_labels.npy ([env_id, agent_id, step, label])
    # ------------------------------------------------------------------
    print(f"\n[8/9] Writing cluster_labels.npy...", flush=True)
    labels_out = np.stack(
        [env_ids, agent_ids, steps, full_labels.astype(np.int64)], axis=1
    ).astype(np.int64)
    labels_path = os.path.join(OUT_DIR, "cluster_labels.npy")
    np.save(labels_path, labels_out)
    print(f"  Wrote {labels_path}, shape={labels_out.shape}")

    # ------------------------------------------------------------------
    # Plots: tsne_3d_scatter.png + cluster_features.png
    # ------------------------------------------------------------------
    title = (
        f"yaofeng_200M LSTM cell: perp={PERPLEXITY}, "
        f"sub={SUBSAMPLE_RATE}, mcs={HDBSCAN_MCS}, ms={HDBSCAN_MS}"
    )
    plot_3d_scatter(tsne_emb, sub_labels, title,
                    os.path.join(OUT_DIR, "tsne_3d_scatter.png"))
    plot_feature_heatmap(
        stats, os.path.join(OUT_DIR, "cluster_features.png"),
        title=f"Cohen's d (perp={PERPLEXITY}, mcs={HDBSCAN_MCS}, ms={HDBSCAN_MS})",
    )

    # ------------------------------------------------------------------
    # 9. Run agent_life_visualization.py via uv
    # ------------------------------------------------------------------
    print(f"\n[9/9] Running agent_life_visualization.py for {NUM_AGENTS} agents...",
          flush=True)
    cmd = [
        "uv", "run", "python",
        os.path.join(REPO_ROOT, "scripts", "agent_life_visualization.py"),
        LIFE_DATA_DIR,
        "--num_agents", str(NUM_AGENTS),
        "--cluster_dir", OUT_DIR,
        "--output", PER_AGENT_DIR,
        "--seed", "42",
    ]
    print(" ".join(cmd))
    # Keep env vars so BLAS stays tame
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True)
    print("---- stdout ----")
    print(res.stdout)
    if res.returncode != 0:
        print("---- stderr ----")
        print(res.stderr)
        raise RuntimeError(
            f"agent_life_visualization.py failed with exit code {res.returncode}"
        )
    print("---- stderr (info) ----")
    print(res.stderr[:2000])

    per_agent_pngs = sorted(Path(PER_AGENT_DIR).glob("*.png"))
    print(f"  Per-agent PNGs: {len(per_agent_pngs)}")
    for p in per_agent_pngs:
        print(f"    {p}")

    # ------------------------------------------------------------------
    # Build summary.png: top row [tsne 3D, Cohen's d], bottom row 3 per-agent plots
    # ------------------------------------------------------------------
    print("\nBuilding summary.png...", flush=True)
    summary_path = os.path.join(OUT_DIR, "summary.png")
    tsne_img = mpimg.imread(os.path.join(OUT_DIR, "tsne_3d_scatter.png"))
    heat_img = mpimg.imread(os.path.join(OUT_DIR, "cluster_features.png"))

    # Pick up to 3 per-agent pngs, prefer middle-lifespan ones
    chosen = per_agent_pngs[:3]

    # Figure geometry: top row 2 panels, bottom row up to 3 panels.
    # Size: ~2400px wide at dpi=150 => figsize width 16.
    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(
        3, 6,
        height_ratios=[5, 3, 6 if len(chosen) > 0 else 0.01],
        hspace=0.08, wspace=0.05,
    )

    ax_tsne = fig.add_subplot(gs[0, 0:3])
    ax_tsne.imshow(tsne_img)
    ax_tsne.axis("off")
    ax_tsne.set_title("T-SNE 3D (4 views) -- HDBSCAN clusters", fontsize=14,
                      fontweight="bold")

    ax_heat = fig.add_subplot(gs[0, 3:6])
    ax_heat.imshow(heat_img)
    ax_heat.axis("off")
    ax_heat.set_title("Cohen's d per cluster per feature", fontsize=14,
                      fontweight="bold")

    # Summary text row
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")
    cluster_desc_lines = []
    for cid in cluster_ids:
        s = stats[cid]
        top_strs = []
        for idx in s["top_features"][:4]:
            d = s["cohens_d"][idx]
            top_strs.append(f"{FEATURE_NAMES[idx]}={d:+.1f}")
        size_full = (full_labels == cid).sum()
        cluster_desc_lines.append(
            f"C{cid}:  subsample n={s['size']} ({s['size']/N*100:.0f}%),  "
            f"full n={size_full} ({size_full/N_full*100:.0f}%)\n"
            f"      top features: " + ", ".join(top_strs)
        )
    summary_text = (
        f"Yaofeng 200M — LSTM cell-state, lifetime-overlay clustering\n"
        f"{N_full} full-res records   |   {N} subsampled (rate {SUBSAMPLE_RATE})   "
        f"|   PCA {PCA_DIMS}d ({explained:.0%} var)   "
        f"|   T-SNE perp {PERPLEXITY}   "
        f"|   HDBSCAN mcs={HDBSCAN_MCS}, ms={HDBSCAN_MS}\n"
        f"Clusters: {len(cluster_ids)}   |   subsample noise: {noise_pct_sub:.1f}%   "
        f"|   full-res transition rate: {trans_rate:.3f}\n\n"
        + "\n".join(cluster_desc_lines)
    )
    ax_text.text(
        0.01, 0.98, summary_text, transform=ax_text.transAxes,
        fontsize=11, va="top", ha="left", family="monospace",
    )

    if chosen:
        n_chosen = len(chosen)
        cell_span = 6 // n_chosen
        for i, png in enumerate(chosen):
            img = mpimg.imread(str(png))
            ax = fig.add_subplot(gs[2, i * cell_span:(i + 1) * cell_span])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(png.name, fontsize=11)

    fig.suptitle(
        "Yaofeng LSTM cell-state lifetime clustering", fontsize=18,
        fontweight="bold", y=0.995,
    )
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {summary_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save a small run-stats json for LOG.md
    summary_stats = {
        "n_full": int(N_full),
        "n_sub": int(N),
        "n_clusters": len(cluster_ids),
        "cluster_ids": [int(c) for c in cluster_ids],
        "noise_pct_sub": float(noise_pct_sub),
        "transition_rate_full": float(trans_rate),
        "pca_explained": float(explained),
        "cluster_sizes_sub": {int(c): int((sub_labels == c).sum()) for c in cluster_ids},
        "cluster_sizes_full": {int(c): int((full_labels == c).sum()) for c in cluster_ids},
        "top_features": {
            int(c): [
                (FEATURE_NAMES[idx], float(stats[c]["cohens_d"][idx]))
                for idx in stats[c]["top_features"][:5]
            ]
            for c in cluster_ids
        },
        "elapsed_sec": float(elapsed),
    }
    import json
    with open(os.path.join(OUT_DIR, "run_stats.json"), "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  Wrote {os.path.join(OUT_DIR, 'run_stats.json')}")


if __name__ == "__main__":
    main()
