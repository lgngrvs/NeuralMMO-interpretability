#!/usr/bin/env python
"""Create publication-quality summary visualizations for the T-SNE sweep.

Reads the sweep CSV and per-config outputs to produce:
  1. score_vs_perplexity.png       - Composite scores grouped by T-SNE perplexity
  2. score_vs_min_cluster_size.png - Composite scores grouped by HDBSCAN min_cluster_size
  3. passing_configs_heatmap.png   - Heatmap: perplexity x min_cluster_size
  4. best_clusters_comparison.png  - Side-by-side T-SNE scatters for top 3 configs
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "sweep_results.csv")
VIZ_DIR = os.path.join(SCRIPT_DIR, "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# Add project root so we can import from analyze_activations
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Only keep rows with nonzero composite score
df_nonzero = df[df["composite_score"] > 0].copy()

print(f"Loaded {len(df)} total configs, {len(df_nonzero)} with nonzero score")
print(f"Passing configs: {df['passes_verification'].sum()}")

# Style settings for publication quality
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# 1. Score vs Perplexity (box + scatter)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

perplexities = sorted(df_nonzero["tsne_perplexity"].unique())
data_by_perp = [df_nonzero[df_nonzero["tsne_perplexity"] == p]["composite_score"].values
                for p in perplexities]

bp = ax.boxplot(data_by_perp, positions=range(len(perplexities)),
                widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="lightblue", edgecolor="navy"),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(color="navy"),
                capprops=dict(color="navy"),
                flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5))

# Overlay individual points with jitter
for i, (p, vals) in enumerate(zip(perplexities, data_by_perp)):
    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
    passing_mask = df_nonzero[df_nonzero["tsne_perplexity"] == p]["passes_verification"].values
    ax.scatter(np.full(len(vals), i) + jitter, vals,
               c=["green" if v else "gray" for v in passing_mask],
               s=20, alpha=0.7, zorder=3, edgecolors="none")

ax.set_xticks(range(len(perplexities)))
ax.set_xticklabels([str(int(p)) for p in perplexities])
ax.set_xlabel("T-SNE Perplexity")
ax.set_ylabel("Composite Score")
ax.set_title("Composite Score vs. T-SNE Perplexity\n(green = passes verification)")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)

fig.savefig(os.path.join(VIZ_DIR, "score_vs_perplexity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote score_vs_perplexity.png")

# ---------------------------------------------------------------------------
# 2. Score vs Min Cluster Size (box + scatter)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

mcs_values = sorted(df_nonzero["k_or_mcs"].unique())
data_by_mcs = [df_nonzero[df_nonzero["k_or_mcs"] == m]["composite_score"].values
               for m in mcs_values]

bp = ax.boxplot(data_by_mcs, positions=range(len(mcs_values)),
                widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="lightyellow", edgecolor="darkorange"),
                medianprops=dict(color="red", linewidth=2),
                whiskerprops=dict(color="darkorange"),
                capprops=dict(color="darkorange"),
                flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5))

for i, (m, vals) in enumerate(zip(mcs_values, data_by_mcs)):
    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
    passing_mask = df_nonzero[df_nonzero["k_or_mcs"] == m]["passes_verification"].values
    ax.scatter(np.full(len(vals), i) + jitter, vals,
               c=["green" if v else "gray" for v in passing_mask],
               s=20, alpha=0.7, zorder=3, edgecolors="none")

ax.set_xticks(range(len(mcs_values)))
ax.set_xticklabels([str(int(m)) for m in mcs_values])
ax.set_xlabel("HDBSCAN min_cluster_size")
ax.set_ylabel("Composite Score")
ax.set_title("Composite Score vs. HDBSCAN min_cluster_size\n(green = passes verification)")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(bottom=0)

fig.savefig(os.path.join(VIZ_DIR, "score_vs_min_cluster_size.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote score_vs_min_cluster_size.png")

# ---------------------------------------------------------------------------
# 3. Heatmap: perplexity x min_cluster_size (averaged over PCA dims & min_samples)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Pivot: average composite score for each (perplexity, min_cluster_size) pair
pivot = df_nonzero.groupby(["tsne_perplexity", "k_or_mcs"])["composite_score"].mean().unstack()
pivot = pivot.sort_index(ascending=True)  # perplexity increasing down
pivot = pivot[sorted(pivot.columns)]  # mcs increasing right

im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([str(int(c)) for c in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([str(int(i)) for i in pivot.index])
ax.set_xlabel("HDBSCAN min_cluster_size")
ax.set_ylabel("T-SNE Perplexity")
ax.set_title("Mean Composite Score\n(averaged over PCA dims & min_samples)")

# Annotate cells
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if np.isnan(val):
            continue
        text_color = "white" if val > pivot.values[~np.isnan(pivot.values)].max() * 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color)

cbar = fig.colorbar(im, ax=ax, label="Composite Score")

fig.savefig(os.path.join(VIZ_DIR, "passing_configs_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote passing_configs_heatmap.png")

# ---------------------------------------------------------------------------
# 4. Best Clusters Comparison - Side-by-side T-SNE scatters for top 3 configs
# ---------------------------------------------------------------------------
# We need to re-run T-SNE and HDBSCAN for the top 3 configs to get embeddings
# and labels. Load from the config directories.
from analyze_activations import (
    FEATURE_NAMES,
    compute_features,
    load_data,
    run_hdbscan,
    run_tsne,
)

print("\nLoading activation data for comparison plot...")
records, _ = load_data(os.path.join(PROJECT_ROOT, "activation_data/baseline_10M/"),
                       subsample_rate=5)
activations, features, metadata = compute_features(records)

# Top 3 configs from the sweep (by composite score):
top_configs = [
    {"label": "#1: perp=50, mcs=50, ms=10", "perplexity": 50, "mcs": 50, "ms": 10},
    {"label": "#2: perp=30, mcs=50, ms=10", "perplexity": 30, "mcs": 50, "ms": 10},
    {"label": "#3: perp=50, mcs=100, ms=10", "perplexity": 50, "mcs": 100, "ms": 10},
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (cfg, ax) in enumerate(zip(top_configs, axes)):
    print(f"  Computing T-SNE for {cfg['label']}...")
    emb, _ = run_tsne(activations, perplexity=cfg["perplexity"], random_state=42)
    labels, _ = run_hdbscan(emb, min_cluster_size=cfg["mcs"], min_samples=cfg["ms"])

    # Plot noise
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(emb[noise_mask, 0], emb[noise_mask, 1],
                   c="lightgray", s=2, alpha=0.3, label="noise", rasterized=True)

    cluster_ids = sorted(set(labels) - {-1})
    n_clusters = len(cluster_ids)
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))

    for i, cl in enumerate(cluster_ids):
        mask = labels == cl
        n = mask.sum()
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=[cmap(i)], s=4, alpha=0.6,
                   label=f"C{cl} (n={n})", rasterized=True)

    noise_pct = noise_mask.sum() / len(labels) * 100
    ax.set_title(f"{cfg['label']}\n{n_clusters} clusters, {noise_pct:.0f}% noise",
                 fontsize=11)
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")

    if n_clusters <= 10:
        ax.legend(markerscale=3, fontsize=7, loc="best",
                  framealpha=0.8, edgecolor="gray")

fig.suptitle("Top 3 T-SNE + HDBSCAN Configurations", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "best_clusters_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote best_clusters_comparison.png")

print(f"\nAll visualizations saved to {VIZ_DIR}/")
