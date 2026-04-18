"""Rebuild summary.png from existing artifacts (after agent_life_visualization
produced more per-agent PNGs on the second run)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from analyze_activations import FEATURE_NAMES  # noqa: E402

with open(os.path.join(HERE, "run_stats.json")) as f:
    stats = json.load(f)

tsne_img = mpimg.imread(os.path.join(HERE, "tsne_3d_scatter.png"))
heat_img = mpimg.imread(os.path.join(HERE, "cluster_features.png"))

per_agent_dir = os.path.join(HERE, "per_agent")
# Pick the 3 with the highest cluster-match rate (from run log we know env=13 id=14 has 95%).
# Order by filename, but prefer a spread of lifespans.
candidates = sorted(os.listdir(per_agent_dir))
candidates = [c for c in candidates if c.endswith(".png")]
# Pick 3 with good visual variety: one short, one mid, one long.
# Rank numbers appear in filename "agent-RR-of-NN.png" and higher rank == longer.
ranked = sorted(candidates, key=lambda c: int(c.split("-")[1]))
if len(ranked) >= 3:
    chosen = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
else:
    chosen = ranked
print("Chosen per-agent pngs:", chosen)

fig = plt.figure(figsize=(22, 22))
gs = fig.add_gridspec(
    3, 6,
    height_ratios=[5, 2.8, 6],
    hspace=0.08, wspace=0.05,
)

ax_tsne = fig.add_subplot(gs[0, 0:3])
ax_tsne.imshow(tsne_img)
ax_tsne.axis("off")
ax_tsne.set_title("T-SNE 3D (4 views) — HDBSCAN clusters",
                  fontsize=14, fontweight="bold")

ax_heat = fig.add_subplot(gs[0, 3:6])
ax_heat.imshow(heat_img)
ax_heat.axis("off")
ax_heat.set_title("Cohen's d per cluster per feature",
                  fontsize=14, fontweight="bold")

# Summary text row
ax_text = fig.add_subplot(gs[1, :])
ax_text.axis("off")
cluster_desc_lines = []
for cid_str, top in stats["top_features"].items():
    cid = int(cid_str)
    size_sub = stats["cluster_sizes_sub"][cid_str]
    size_full = stats["cluster_sizes_full"][cid_str]
    n_sub = stats["n_sub"]
    n_full = stats["n_full"]
    top_strs = [f"{name}={d:+.1f}" for name, d in top[:4]]
    cluster_desc_lines.append(
        f"C{cid}:  subsample n={size_sub} ({size_sub/n_sub*100:.0f}%),  "
        f"full n={size_full} ({size_full/n_full*100:.0f}%)    "
        f"top feats: " + ", ".join(top_strs)
    )
summary_text = (
    f"Yaofeng 200M — LSTM cell-state, lifetime-overlay clustering\n"
    f"{stats['n_full']} full-res records   |   {stats['n_sub']} subsampled (rate 50)   "
    f"|   PCA 20d ({stats['pca_explained']:.0%} var)   "
    f"|   T-SNE perp 50   |   HDBSCAN mcs=200, ms=3\n"
    f"{stats['n_clusters']} clusters   |   subsample noise: {stats['noise_pct_sub']:.1f}%   "
    f"|   full-res transition rate (tick-to-tick): {stats['transition_rate_full']:.4f}\n\n"
    + "\n".join(cluster_desc_lines)
)
ax_text.text(0.01, 0.98, summary_text, transform=ax_text.transAxes,
             fontsize=11, va="top", ha="left", family="monospace")

# Bottom row: per-agent plots
n_chosen = len(chosen)
cell_span = 6 // n_chosen if n_chosen else 6
for i, png in enumerate(chosen):
    img = mpimg.imread(os.path.join(per_agent_dir, png))
    ax = fig.add_subplot(gs[2, i * cell_span:(i + 1) * cell_span])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(png, fontsize=11)

fig.suptitle(
    "Yaofeng LSTM cell-state lifetime clustering (perp=50, mcs=200, ms=3)",
    fontsize=18, fontweight="bold", y=0.995,
)
out = os.path.join(HERE, "summary.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out}")
