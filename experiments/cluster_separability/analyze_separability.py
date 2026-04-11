"""Cluster separability analysis for yaofeng_200M T-SNE 3D clusters.

Computes four metrics to determine whether clusters C0 (unequipped) and
C1 (food-stressed) are genuinely distinct from the dominant C2 cluster,
or merely tails of C2's distribution.

Metrics:
  1. Per-cluster silhouette score distribution
  2. K-NN purity (k=20)
  3. Mahalanobis distance from C2 centroid
  4. Density along inter-centroid axis
"""

import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from analyze_activations import load_data, compute_features, FEATURE_NAMES

DATA_DIR = os.path.join(REPO_ROOT, "activation_data", "yaofeng_200M")
LABEL_PATH = os.path.join(
    REPO_ROOT, "experiments", "T-SNE-3d-sweep-yaofeng_200M", "config_1", "cluster_labels.npy"
)
OUT_DIR = os.path.dirname(__file__)

CLUSTER_NAMES = {0: "C0 (unequipped)", 1: "C1 (food-stressed)", 2: "C2 (steady-state)"}
CLUSTER_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#3498db"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data (subsample=5)...")
records, _ = load_data(DATA_DIR, subsample_rate=5)
activations, features, metadata = compute_features(records)

labels = np.load(LABEL_PATH)
print(f"  activations shape: {activations.shape}, labels shape: {labels.shape}")

# Exclude noise points (label == -1)
mask = labels != -1
X = activations[mask]
y = labels[mask]
print(f"  After noise removal: {X.shape[0]} points  (C0={np.sum(y==0)}, C1={np.sum(y==1)}, C2={np.sum(y==2)})")

# PCA to 30 dimensions
pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X)
print(f"  PCA explained variance (30 dims): {pca.explained_variance_ratio_.sum():.3f}")

# ---------------------------------------------------------------------------
# Metric 1: Per-Cluster Silhouette Score Distribution
# ---------------------------------------------------------------------------
print("Computing silhouette scores...")
sil = silhouette_samples(X_pca, y, metric="euclidean")
overall_sil = silhouette_score(X_pca, y, metric="euclidean")

sil_per_cluster = {}
for c in sorted(np.unique(y)):
    sil_per_cluster[c] = sil[y == c]
    med = np.median(sil_per_cluster[c])
    mean = np.mean(sil_per_cluster[c])
    frac_pos = np.mean(sil_per_cluster[c] > 0)
    print(f"  C{c}: median={med:.3f}, mean={mean:.3f}, frac>0={frac_pos:.3f}")

# ---------------------------------------------------------------------------
# Metric 2: K-NN Purity
# ---------------------------------------------------------------------------
K = 20
print(f"Computing {K}-NN purity...")
nn = NearestNeighbors(n_neighbors=K + 1, algorithm="auto").fit(X_pca)
dists, indices = nn.kneighbors(X_pca)

# Exclude self (column 0)
neighbor_labels = y[indices[:, 1:]]
purity = np.mean(neighbor_labels == y[:, None], axis=1)

purity_per_cluster = {}
knn_dists_per_cluster = {}
for c in sorted(np.unique(y)):
    cm = y == c
    purity_per_cluster[c] = purity[cm]
    # Mean distance to k-NN (excluding self)
    knn_dists_per_cluster[c] = dists[cm, 1:].mean(axis=1)
    print(f"  C{c}: mean_purity={purity_per_cluster[c].mean():.3f}, "
          f"median_purity={np.median(purity_per_cluster[c]):.3f}")

# ---------------------------------------------------------------------------
# Metric 3: Mahalanobis Distance from C2 Centroid
# ---------------------------------------------------------------------------
print("Computing Mahalanobis distances from C2...")
mu_c2 = X_pca[y == 2].mean(axis=0)
cov_c2 = np.cov(X_pca[y == 2], rowvar=False)
cov_inv = np.linalg.pinv(cov_c2)

maha_all = np.zeros(len(X_pca))
for i in range(len(X_pca)):
    maha_all[i] = mahalanobis(X_pca[i], mu_c2, cov_inv)

maha_per_cluster = {}
expected_mean = np.sqrt(30)  # ~5.48 under multivariate Gaussian
for c in sorted(np.unique(y)):
    maha_per_cluster[c] = maha_all[y == c]
    print(f"  C{c}: mean_maha={maha_per_cluster[c].mean():.2f}, "
          f"median={np.median(maha_per_cluster[c]):.2f}  "
          f"(expected under Gaussian: ~{expected_mean:.1f})")

# ---------------------------------------------------------------------------
# Metric 4: Density Along Inter-Centroid Axis
# ---------------------------------------------------------------------------
print("Computing density along inter-centroid axes...")
centroids = {c: X_pca[y == c].mean(axis=0) for c in sorted(np.unique(y))}

density_data = {}
for c in [0, 1]:
    axis = centroids[c] - centroids[2]
    axis = axis / np.linalg.norm(axis)
    proj = X_pca @ axis

    # Per-cluster projections
    proj_per_cluster = {}
    for cc in sorted(np.unique(y)):
        proj_per_cluster[cc] = proj[y == cc]

    # Fit KDEs
    kdes = {}
    x_grid = np.linspace(np.percentile(proj, 1), np.percentile(proj, 99), 500)
    for cc in sorted(np.unique(y)):
        kde = gaussian_kde(proj_per_cluster[cc])
        kdes[cc] = kde(x_grid)

    density_data[c] = {
        "proj_per_cluster": proj_per_cluster,
        "x_grid": x_grid,
        "kdes": kdes,
        "axis_label": f"C{c}--C2 axis",
    }

# ---------------------------------------------------------------------------
# Visualization: 2x2 panel
# ---------------------------------------------------------------------------
print("Generating figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Cluster Separability Analysis (yaofeng_200M, 30-dim PCA)", fontsize=14, y=0.98)

# --- Panel 1: Silhouette violin plot ---
ax = axes[0, 0]
violin_data = [sil_per_cluster[c] for c in sorted(np.unique(y))]
parts = ax.violinplot(violin_data, positions=[0, 1, 2], showmedians=True, showextrema=False)
for i, c in enumerate(sorted(np.unique(y))):
    parts["bodies"][i].set_facecolor(CLUSTER_COLORS[c])
    parts["bodies"][i].set_alpha(0.7)
parts["cmedians"].set_color("black")
ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels([CLUSTER_NAMES[c] for c in [0, 1, 2]], fontsize=9)
ax.set_ylabel("Silhouette Score")
ax.set_title(f"Silhouette Distribution (overall={overall_sil:.3f})")
# Annotate medians
for i, c in enumerate(sorted(np.unique(y))):
    med = np.median(sil_per_cluster[c])
    ax.text(i, med + 0.03, f"med={med:.2f}", ha="center", fontsize=8)

# --- Panel 2: K-NN Purity histogram ---
ax = axes[0, 1]
bins = np.linspace(0, 1, 30)
for c in sorted(np.unique(y)):
    ax.hist(purity_per_cluster[c], bins=bins, alpha=0.55, color=CLUSTER_COLORS[c],
            label=f"{CLUSTER_NAMES[c]} (mean={purity_per_cluster[c].mean():.2f})",
            density=True)
ax.set_xlabel(f"k-NN Purity (k={K})")
ax.set_ylabel("Density")
ax.set_title("K-NN Purity Distribution")
ax.legend(fontsize=8, loc="upper left")

# --- Panel 3: Mahalanobis distance histograms ---
ax = axes[1, 0]
bins_maha = np.linspace(0, max(maha_all.max(), 20), 60)
for c in sorted(np.unique(y)):
    ax.hist(maha_per_cluster[c], bins=bins_maha, alpha=0.55, color=CLUSTER_COLORS[c],
            label=f"{CLUSTER_NAMES[c]} (mean={maha_per_cluster[c].mean():.1f})",
            density=True)
ax.axvline(expected_mean, color="black", ls="--", lw=1.2, label=f"Expected sqrt(30)={expected_mean:.1f}")
ax.set_xlabel("Mahalanobis Distance from C2 Centroid")
ax.set_ylabel("Density")
ax.set_title("Mahalanobis Distance from C2")
ax.legend(fontsize=8)

# --- Panel 4: Density along inter-centroid axes ---
ax = axes[1, 1]
linestyles = {0: "-", 1: "--"}
for c_target in [0, 1]:
    dd = density_data[c_target]
    x_grid = dd["x_grid"]
    for cc in sorted(np.unique(y)):
        label_str = f"{CLUSTER_NAMES[cc]} on {dd['axis_label']}" if c_target == 0 else None
        alpha = 0.9 if c_target == 0 else 0.5
        ax.plot(x_grid, dd["kdes"][cc], color=CLUSTER_COLORS[cc],
                ls=linestyles[c_target], alpha=alpha, lw=1.5, label=label_str)

# Add annotation for axes
ax.text(0.02, 0.95, "Solid: C0--C2 axis\nDashed: C1--C2 axis",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.set_xlabel("Projection onto Inter-Centroid Axis")
ax.set_ylabel("Density")
ax.set_title("Density Along Inter-Centroid Axes")
ax.legend(fontsize=8, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(OUT_DIR, "separability_analysis.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved figure: {fig_path}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------
summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("CLUSTER SEPARABILITY ANALYSIS -- yaofeng_200M (subsample=5)")
summary_lines.append("=" * 70)
summary_lines.append(f"Total points (non-noise): {len(y)}")
for c in sorted(np.unique(y)):
    summary_lines.append(f"  C{c}: {np.sum(y==c)} points")
summary_lines.append(f"PCA explained variance (30 dims): {pca.explained_variance_ratio_.sum():.3f}")
summary_lines.append("")

summary_lines.append("--- Metric 1: Silhouette Scores ---")
summary_lines.append(f"Overall silhouette: {overall_sil:.4f}")
for c in sorted(np.unique(y)):
    s = sil_per_cluster[c]
    summary_lines.append(
        f"  C{c}: median={np.median(s):.3f}, mean={np.mean(s):.3f}, "
        f"frac>0={np.mean(s>0):.3f}, n={len(s)}"
    )
summary_lines.append("Interpretation: Genuine clusters have positive median silhouette.")
summary_lines.append("")

summary_lines.append(f"--- Metric 2: {K}-NN Purity ---")
for c in sorted(np.unique(y)):
    p = purity_per_cluster[c]
    summary_lines.append(
        f"  C{c}: mean={p.mean():.3f}, median={np.median(p):.3f}, "
        f"mean_knn_dist={knn_dists_per_cluster[c].mean():.3f}"
    )
summary_lines.append("Interpretation: High purity = points surrounded by same-cluster neighbors.")
summary_lines.append("")

summary_lines.append("--- Metric 3: Mahalanobis Distance from C2 ---")
summary_lines.append(f"Expected under Gaussian: mean ~ sqrt(30) = {expected_mean:.2f}")
for c in sorted(np.unique(y)):
    m = maha_per_cluster[c]
    summary_lines.append(
        f"  C{c}: mean={m.mean():.2f}, median={np.median(m):.2f}, "
        f"std={m.std():.2f}, frac_above_8={np.mean(m>8):.3f}"
    )
summary_lines.append("Interpretation: If C0/C1 have Mahalanobis >> sqrt(30), they lie outside C2's distribution.")
summary_lines.append("")

summary_lines.append("--- Metric 4: Density Along Inter-Centroid Axis ---")
for c_target in [0, 1]:
    dd = density_data[c_target]
    # Check for dip: look at density of combined data between centroids
    proj_c2 = dd["proj_per_cluster"][2]
    proj_ct = dd["proj_per_cluster"][c_target]
    c2_center = np.mean(proj_c2)
    ct_center = np.mean(proj_ct)
    lo, hi = sorted([c2_center, ct_center])
    # Check if total density has a dip in the inter-centroid region
    midpoint = (lo + hi) / 2
    x_grid = dd["x_grid"]
    total_kde = gaussian_kde(np.concatenate([proj_c2, proj_ct]))
    total_density = total_kde(x_grid)
    # Density at midpoint vs at centroid peaks
    mid_idx = np.argmin(np.abs(x_grid - midpoint))
    lo_idx = np.argmin(np.abs(x_grid - lo))
    hi_idx = np.argmin(np.abs(x_grid - hi))
    d_mid = total_density[mid_idx]
    d_lo = total_density[lo_idx]
    d_hi = total_density[hi_idx]
    dip_ratio = d_mid / max(d_lo, d_hi)
    summary_lines.append(
        f"  C{c_target}--C2 axis: centroid_sep={np.linalg.norm(centroids[c_target]-centroids[2]):.2f}, "
        f"density_at_midpoint/peak={dip_ratio:.3f}"
    )
    if dip_ratio < 0.5:
        summary_lines.append(f"    -> Clear density dip (ratio={dip_ratio:.3f} < 0.5): genuine separation")
    elif dip_ratio < 0.8:
        summary_lines.append(f"    -> Moderate dip (ratio={dip_ratio:.3f}): partial separation")
    else:
        summary_lines.append(f"    -> No clear dip (ratio={dip_ratio:.3f} >= 0.8): likely tail of C2")
summary_lines.append("")

# Overall verdict
summary_lines.append("=" * 70)
summary_lines.append("OVERALL VERDICT")
summary_lines.append("=" * 70)

for c in [0, 1]:
    name = CLUSTER_NAMES[c]
    s_med = np.median(sil_per_cluster[c])
    p_mean = purity_per_cluster[c].mean()
    m_mean = maha_per_cluster[c].mean()
    # Score: how many metrics suggest genuine separation?
    score = 0
    reasons = []
    if s_med > 0.05:
        score += 1
        reasons.append(f"positive silhouette (med={s_med:.3f})")
    else:
        reasons.append(f"weak silhouette (med={s_med:.3f})")
    if p_mean > 0.7:
        score += 1
        reasons.append(f"high kNN purity ({p_mean:.3f})")
    else:
        reasons.append(f"low kNN purity ({p_mean:.3f})")
    if m_mean > expected_mean + 2:
        score += 1
        reasons.append(f"high Mahalanobis ({m_mean:.1f} >> {expected_mean:.1f})")
    else:
        reasons.append(f"Mahalanobis near expected ({m_mean:.1f} vs {expected_mean:.1f})")

    if score >= 2:
        verdict = "GENUINELY DISTINCT"
    elif score == 1:
        verdict = "WEAKLY DISTINCT / BORDERLINE"
    else:
        verdict = "LIKELY TAIL OF C2"

    summary_lines.append(f"\n{name}: {verdict}")
    for r in reasons:
        summary_lines.append(f"  - {r}")

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

summary_path = os.path.join(OUT_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved summary: {summary_path}")
