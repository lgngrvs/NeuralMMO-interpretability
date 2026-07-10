#!/usr/bin/env python
"""
Linear probes to predict 3 action-decoder clusters from action-decoder activations.

Prior run (PCA=20, min_samples=3) found only 2 clusters with 99.2% accuracy.
This run uses PCA=30 and min_samples=10, which yields 3 clusters. Key question:
does the third cluster (fragile food-stressed mode in PCA dims 21-30) remain
linearly separable?

Clustering config (exact):
  - Data: activation_data/yaofeng_200M/activations.cache.npz (36,501 records)
  - Subsample: rate 5, per-trajectory
  - PCA: 30 dims (random_state=42)
  - T-SNE 3D: perplexity=50, max_iter=1000, learning_rate='auto', random_state=42
  - HDBSCAN: min_cluster_size=200, min_samples=10
  - KNN-propagate (k=5) to all 36,501 records via PCA-30 space

Run:
    cd /workspace/NeuralMMO-interpretability && \
    OMP_NUM_THREADS=4 uv run python experiments/action_decoder_cluster_probes_3cl/run_analysis.py
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

# Ensure we run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)

OUTPUT_DIR = REPO_ROOT / "experiments" / "action_decoder_cluster_probes_3cl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_JOBS = 4


# ============================================================================
# STEP 1: Generate 3-cluster labels via PCA(30)->T-SNE 3D->HDBSCAN->KNN
# ============================================================================

def generate_ad_clusters(acts, steps, env_ids, agent_ids):
    """Generate 3-cluster labels via subsample->PCA(30)->T-SNE 3D->HDBSCAN->KNN propagation."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    import hdbscan as hdbscan_lib

    N = len(acts)
    print(f"  Total points: {N}")

    # Subsample at rate 5, per-trajectory (group by env_id*1_000_000 + agent_id, sort by step)
    sub_rate = 5
    traj_keys = env_ids.astype(np.int64) * 1_000_000 + agent_ids.astype(np.int64)
    unique_keys = np.unique(traj_keys)
    keep_mask = np.zeros(N, dtype=bool)
    for key in unique_keys:
        mask_k = traj_keys == key
        idx = np.where(mask_k)[0]
        order = np.argsort(steps[idx])
        idx_sorted = idx[order]
        keep_mask[idx_sorted[::sub_rate]] = True

    sub_acts = acts[keep_mask]
    sub_indices = np.where(keep_mask)[0]
    n_sub = sub_acts.shape[0]
    print(f"  Subsampled: {n_sub} points (rate {sub_rate})")

    # PCA 30 (changed from 20)
    pca = PCA(n_components=30, random_state=SEED)
    pca_sub = pca.fit_transform(sub_acts).astype(np.float32)
    print(f"  PCA 30D explains {pca.explained_variance_ratio_.sum():.1%} variance")

    # T-SNE 3D
    print("  Running T-SNE 3D (perplexity=50, max_iter=1000)...")
    t_tsne = time.time()
    tsne = TSNE(
        n_components=3,
        perplexity=50,
        learning_rate="auto",
        max_iter=1000,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    emb = tsne.fit_transform(pca_sub)
    print(f"  T-SNE done in {time.time() - t_tsne:.1f}s: shape {emb.shape}")

    # HDBSCAN with min_samples=10 (changed from 3)
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=200, min_samples=10)
    sub_labels = clusterer.fit_predict(emb)
    cluster_ids = sorted(set(sub_labels.tolist()) - {-1})
    noise_pct = (sub_labels == -1).sum() / n_sub * 100
    print(f"  HDBSCAN: {len(cluster_ids)} clusters, {noise_pct:.1f}% noise")
    for cid in cluster_ids:
        print(f"    C{cid}: n={(sub_labels == cid).sum()}")

    # KNN propagate labels to ALL full-res records via PCA-30 space
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

    return full_labels, pca


print("=" * 80)
print("STEP 1: Generate 3-cluster labels (PCA=30, min_samples=10)")
print("=" * 80)

t_start = time.time()

# Load action-decoder data
cache = np.load("activation_data/yaofeng_200M/activations.cache.npz")
ad_acts = cache["activations"]      # (36501, 256)
ad_feats = cache["features"]        # (36501, 35)
ad_steps = cache["steps"]           # (36501,)
ad_env_ids = cache["env_ids"]       # (36501,)
ad_agent_ids = cache["agent_ids"]   # (36501,)

print(f"Action-decoder cache: {ad_acts.shape[0]} records, {ad_acts.shape[1]}-dim")

# Generate clusters
ad_labels, pca_30 = generate_ad_clusters(ad_acts, ad_steps, ad_env_ids, ad_agent_ids)

# Save cluster labels in [env_id, agent_id, step, label] format
label_arr = np.column_stack([
    ad_env_ids.astype(np.float64),
    ad_agent_ids.astype(np.float64),
    ad_steps.astype(np.float64),
    ad_labels.astype(np.float64),
])
np.save(OUTPUT_DIR / "cluster_labels.npy", label_arr)
print(f"  Saved cluster_labels.npy: {label_arr.shape}")

n_clusters = len(set(ad_labels.tolist()) - {-1})
label_dist = dict(zip(*np.unique(ad_labels, return_counts=True)))
print(f"  Cluster distribution: {label_dist}")


# ============================================================================
# STEP 2: Trajectory-based train/test split
# ============================================================================

print(f"\n{'='*80}")
print("STEP 2: Trajectory-based train/test split")
print("=" * 80)

from sklearn.model_selection import GroupShuffleSplit

X = ad_acts
y = ad_labels

# Group by (env_id, agent_id) = trajectory
groups = np.array([f"{e}_{a}" for e, a in zip(ad_env_ids, ad_agent_ids)])
unique_groups = np.unique(groups)
print(f"  Unique trajectories: {len(unique_groups)}")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"  Train: {len(X_train)} samples ({len(np.unique(groups[train_idx]))} trajectories)")
print(f"  Test:  {len(X_test)} samples ({len(np.unique(groups[test_idx]))} trajectories)")
print(f"  Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  Test  label dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")


# ============================================================================
# STEP 3: Probes
# ============================================================================

print(f"\n{'='*80}")
print("STEP 3: Train logistic regression probes")
print("=" * 80)

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Standardize activations
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3a. Logistic regression on raw 256-dim ---
print("\n  [3a] Logistic regression on raw 256-dim activations...")
lr_raw = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=SEED,
    n_jobs=N_JOBS,
)
lr_raw.fit(X_train_scaled, y_train)
y_pred_raw = lr_raw.predict(X_test_scaled)
acc_raw = accuracy_score(y_test, y_pred_raw)
f1_raw = f1_score(y_test, y_pred_raw, average="macro")
print(f"    Accuracy: {acc_raw:.4f}")
print(f"    Macro F1: {f1_raw:.4f}")
print(classification_report(y_test, y_pred_raw, digits=4))

cm_raw = confusion_matrix(y_test, y_pred_raw)

# --- 3b. PCA(50) + Logistic regression ---
print("\n  [3b] PCA(50) + Logistic regression...")
pca = PCA(n_components=50, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"    PCA explained variance (50 components): {pca.explained_variance_ratio_.sum():.4f}")

lr_pca = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=SEED,
    n_jobs=N_JOBS,
)
lr_pca.fit(X_train_pca, y_train)
y_pred_pca = lr_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca, average="macro")
print(f"    Accuracy: {acc_pca:.4f}")
print(f"    Macro F1: {f1_pca:.4f}")
print(classification_report(y_test, y_pred_pca, digits=4))

# --- 3c. Shuffle baseline ---
print("\n  [3c] Shuffle baseline (permuted labels)...")
rng = np.random.RandomState(SEED)
y_train_shuffled = rng.permutation(y_train)
lr_shuffle = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=SEED,
    n_jobs=N_JOBS,
)
lr_shuffle.fit(X_train_scaled, y_train_shuffled)
y_pred_shuffle = lr_shuffle.predict(X_test_scaled)
acc_shuffle = accuracy_score(y_test, y_pred_shuffle)
f1_shuffle = f1_score(y_test, y_pred_shuffle, average="macro")
print(f"    Shuffle accuracy: {acc_shuffle:.4f}")
print(f"    Shuffle macro F1: {f1_shuffle:.4f}")


# ============================================================================
# STEP 4: Linear Discriminant Analysis
# ============================================================================

print(f"\n{'='*80}")
print("STEP 4: LDA projection")
print("=" * 80)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda_n_components = min(2, n_clusters - 1)
print(f"  LDA n_components = {lda_n_components} (n_clusters = {n_clusters})")
lda = LinearDiscriminantAnalysis(n_components=lda_n_components)
X_lda_train = lda.fit_transform(X_train_scaled, y_train)
X_lda_test = lda.transform(X_test_scaled)

# LDA classification accuracy (for reference)
y_pred_lda = lda.predict(X_test_scaled)
acc_lda = accuracy_score(y_test, y_pred_lda)
print(f"  LDA classification accuracy: {acc_lda:.4f}")

# Full dataset LDA projection for plotting
X_all_scaled = scaler.transform(X)
X_lda_all = lda.transform(X_all_scaled)


# ============================================================================
# STEP 5: Probe direction analysis
# ============================================================================

print(f"\n{'='*80}")
print("STEP 5: Probe weight vs PCA direction analysis")
print("=" * 80)

# Weight vectors from raw logistic regression
W = lr_raw.coef_  # (n_classes, 256) for multiclass
n_probe_rows = W.shape[0]

# Top-20 PCA directions (in original scaled space)
pca_full = PCA(n_components=20, random_state=SEED)
pca_full.fit(X_train_scaled)
pca_dirs = pca_full.components_  # (20, 256)

# Cosine similarity between each probe weight vector and each PCA direction
from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(W, pca_dirs)  # (n_classes, 20)
print(f"  Probe weight shapes: {W.shape}")
print(f"  PCA directions shape: {pca_dirs.shape}")
print(f"  Cosine similarity matrix shape: {cos_sim.shape}")
print(f"  Max |cos_sim| per probe class: {np.abs(cos_sim).max(axis=1)}")
print(f"  PCA component with max alignment per class: {np.abs(cos_sim).argmax(axis=1)}")


# ============================================================================
# STEP 6: Generate plots
# ============================================================================

print(f"\n{'='*80}")
print("STEP 6: Generate plots")
print("=" * 80)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

cluster_ids = sorted(set(y.tolist()))
n_cls = len(cluster_ids)
cluster_names = [f"Cluster {c}" for c in cluster_ids]
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:n_cls]

# --- 6a. Confusion matrix ---
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
im = ax_cm.imshow(cm_raw, interpolation="nearest", cmap="Blues")
ax_cm.set_title(
    "Confusion Matrix\n(Logistic Regression, 256-dim AD -> 3 AD clusters)",
    fontsize=12,
)
ax_cm.set_xlabel("Predicted", fontsize=11)
ax_cm.set_ylabel("True", fontsize=11)
ax_cm.set_xticks(range(n_cls))
ax_cm.set_yticks(range(n_cls))
ax_cm.set_xticklabels(cluster_names, fontsize=9)
ax_cm.set_yticklabels(cluster_names, fontsize=9)
for i in range(n_cls):
    for j in range(n_cls):
        val = cm_raw[i, j]
        total_row = cm_raw[i].sum()
        pct = val / total_row * 100
        color = "white" if val > cm_raw.max() / 2 else "black"
        ax_cm.text(
            j, i, f"{val}\n({pct:.1f}%)",
            ha="center", va="center", color=color, fontsize=10,
        )
fig_cm.colorbar(im, ax=ax_cm)
fig_cm.tight_layout()
fig_cm.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig_cm)
print("  Saved confusion_matrix.png")

# --- 6b. LDA projection ---
fig_lda, ax_lda = plt.subplots(figsize=(8, 6))
if lda_n_components >= 2:
    for idx, label in enumerate(cluster_ids):
        mask = y == label
        ax_lda.scatter(
            X_lda_all[mask, 0],
            X_lda_all[mask, 1],
            c=cluster_colors[idx],
            label=f"Cluster {label} (n={mask.sum()})",
            alpha=0.3,
            s=8,
            edgecolors="none",
        )
    ax_lda.set_xlabel("LDA Component 1", fontsize=11)
    ax_lda.set_ylabel("LDA Component 2", fontsize=11)
else:
    # 1D LDA: histogram/strip plot
    for idx, label in enumerate(cluster_ids):
        mask = y == label
        jitter = np.random.RandomState(SEED).normal(0, 0.1, size=mask.sum())
        ax_lda.scatter(
            X_lda_all[mask, 0],
            jitter,
            c=cluster_colors[idx],
            label=f"Cluster {label} (n={mask.sum()})",
            alpha=0.2,
            s=6,
            edgecolors="none",
        )
    ax_lda.set_xlabel("LDA Component 1", fontsize=11)
    ax_lda.set_ylabel("(jitter)", fontsize=11)
ax_lda.set_title(
    "LDA Projection of Action-Decoder Activations\nColored by 3 AD Clusters (PCA=30)",
    fontsize=12,
)
ax_lda.legend(fontsize=9, markerscale=3)
fig_lda.tight_layout()
fig_lda.savefig(OUTPUT_DIR / "lda_projection.png", dpi=150, bbox_inches="tight")
plt.close(fig_lda)
print("  Saved lda_projection.png")

# --- 6c. Probe weights vs PCA directions ---
probe_row_labels = [f"C{i} weight" for i in cluster_ids[:n_probe_rows]]
fig_pw, ax_pw = plt.subplots(figsize=(10, max(3, n_probe_rows + 1)))
im_pw = ax_pw.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_pw.set_xlabel("PCA Component", fontsize=11)
ax_pw.set_ylabel("Probe Weight Vector", fontsize=11)
ax_pw.set_yticks(range(n_probe_rows))
ax_pw.set_yticklabels(probe_row_labels, fontsize=9)
ax_pw.set_xticks(range(20))
ax_pw.set_xticklabels([f"PC{i}" for i in range(20)], fontsize=8, rotation=45)
ax_pw.set_title(
    "Cosine Similarity: Probe Weights vs Top-20 PCA Directions\n(3 clusters, PCA=30)",
    fontsize=12,
)
fig_pw.colorbar(im_pw, ax=ax_pw, label="Cosine Similarity")
for i in range(n_probe_rows):
    for j in range(20):
        if abs(cos_sim[i, j]) > 0.15:
            ax_pw.text(
                j, i, f"{cos_sim[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
            )
fig_pw.tight_layout()
fig_pw.savefig(OUTPUT_DIR / "probe_weights_pca.png", dpi=150, bbox_inches="tight")
plt.close(fig_pw)
print("  Saved probe_weights_pca.png")

# --- 6d. Summary (2x2 grid) ---
fig_summary = plt.figure(figsize=(16, 14))
gs = GridSpec(2, 2, figure=fig_summary, hspace=0.35, wspace=0.3)

# Top-left: Confusion matrix
ax1 = fig_summary.add_subplot(gs[0, 0])
im1 = ax1.imshow(cm_raw, interpolation="nearest", cmap="Blues")
ax1.set_title(
    "Confusion Matrix (256-dim LogReg)\nAD activations -> 3 AD clusters",
    fontsize=11, fontweight="bold",
)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("True")
ax1.set_xticks(range(n_cls))
ax1.set_yticks(range(n_cls))
ax1.set_xticklabels(cluster_names, fontsize=8)
ax1.set_yticklabels(cluster_names, fontsize=8)
for i in range(n_cls):
    for j in range(n_cls):
        val = cm_raw[i, j]
        total_row = cm_raw[i].sum()
        pct = val / total_row * 100
        color = "white" if val > cm_raw.max() / 2 else "black"
        ax1.text(
            j, i, f"{val}\n({pct:.1f}%)",
            ha="center", va="center", color=color, fontsize=9,
        )
fig_summary.colorbar(im1, ax=ax1, fraction=0.046)

# Top-right: LDA projection
ax2 = fig_summary.add_subplot(gs[0, 1])
if lda_n_components >= 2:
    for idx, label in enumerate(cluster_ids):
        mask = y == label
        ax2.scatter(
            X_lda_all[mask, 0],
            X_lda_all[mask, 1],
            c=cluster_colors[idx],
            label=f"Cluster {label}",
            alpha=0.3,
            s=6,
            edgecolors="none",
        )
    ax2.set_xlabel("LDA Component 1")
    ax2.set_ylabel("LDA Component 2")
else:
    for idx, label in enumerate(cluster_ids):
        mask = y == label
        jitter = np.random.RandomState(SEED + idx).normal(0, 0.1, size=mask.sum())
        ax2.scatter(
            X_lda_all[mask, 0],
            jitter,
            c=cluster_colors[idx],
            label=f"Cluster {label}",
            alpha=0.2,
            s=4,
            edgecolors="none",
        )
    ax2.set_xlabel("LDA Component 1")
    ax2.set_ylabel("(jitter)")
ax2.set_title(
    "LDA Projection (Action-Decoder Space)\nColored by 3 AD clusters (PCA=30)",
    fontsize=11, fontweight="bold",
)
ax2.legend(fontsize=8, markerscale=3)

# Bottom-left: Probe weights vs PCA
ax3 = fig_summary.add_subplot(gs[1, 0])
im3 = ax3.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax3.set_xlabel("PCA Component")
ax3.set_ylabel("Probe Weight")
ax3.set_yticks(range(n_probe_rows))
ax3.set_yticklabels(probe_row_labels, fontsize=8)
ax3.set_xticks(range(20))
ax3.set_xticklabels([f"PC{i}" for i in range(20)], fontsize=7, rotation=45)
ax3.set_title("Probe Weights vs PCA Directions", fontsize=11, fontweight="bold")
fig_summary.colorbar(im3, ax=ax3, fraction=0.046, label="Cosine Sim.")
for i in range(n_probe_rows):
    for j in range(20):
        if abs(cos_sim[i, j]) > 0.2:
            ax3.text(
                j, i, f"{cos_sim[i, j]:.2f}",
                ha="center", va="center", fontsize=6,
            )

# Bottom-right: Metrics text panel
ax4 = fig_summary.add_subplot(gs[1, 1])
ax4.axis("off")

# Per-class metrics from classification report
report_raw = classification_report(y_test, y_pred_raw, output_dict=True, digits=4)
report_pca = classification_report(y_test, y_pred_pca, output_dict=True, digits=4)

metrics_text = (
    "3-Cluster Action-Decoder Probe Results\n"
    "=" * 44 + "\n\n"
    f"Config: PCA=30, HDBSCAN min_samples=10\n"
    f"Total samples: {len(X):,}\n"
    f"Train / Test: {len(X_train):,} / {len(X_test):,}\n"
    f"Trajectories: {len(unique_groups)} total\n"
    f"Clusters: {n_cls}\n\n"
    "--- AD -> 3 AD-clusters (LogReg 256-dim) ---\n"
    f"  Accuracy:  {acc_raw:.4f}\n"
    f"  Macro F1:  {f1_raw:.4f}\n"
)
for c in [str(c) for c in cluster_ids]:
    p = report_raw[c]["precision"]
    r = report_raw[c]["recall"]
    f = report_raw[c]["f1-score"]
    metrics_text += f"  C{c}: P={p:.3f} R={r:.3f} F1={f:.3f}\n"

metrics_text += (
    f"\n--- AD -> 3 AD-clusters (LogReg PCA-50) ---\n"
    f"  Accuracy:  {acc_pca:.4f}\n"
    f"  Macro F1:  {f1_pca:.4f}\n"
    f"  PCA var:   {pca.explained_variance_ratio_.sum():.4f}\n"
)

metrics_text += (
    f"\n--- LDA Classification ---\n"
    f"  Accuracy:  {acc_lda:.4f}\n"
)

metrics_text += (
    f"\n--- Shuffle Baseline ---\n"
    f"  Accuracy:  {acc_shuffle:.4f}\n"
    f"  Macro F1:  {f1_shuffle:.4f}\n"
)

# Comparison panel
metrics_text += (
    f"\n{'='*44}\n"
    f"COMPARISON:\n"
    f"  Prior 2-cluster (PCA=20):  99.2% acc, F1=0.989\n"
    f"  This 3-cluster (PCA=30):   {acc_raw:.1%} acc, F1={f1_raw:.3f}\n"
    f"  LSTM cluster from AD:      70.9% acc, F1=0.664\n"
)

if acc_raw > 0.95:
    verdict = "YES - all 3 clusters are nearly linear"
elif acc_raw > 0.85:
    verdict = "MOSTLY - high but not perfect separation"
elif acc_raw > 0.7:
    verdict = "PARTIALLY - some nonlinear structure"
else:
    verdict = "NO - clusters are genuinely nonlinear"
metrics_text += (
    f"\nAll 3 clusters linearly separable?\n  {verdict}\n"
    f"Max |cos_sim| with PCA: {np.abs(cos_sim).max():.3f}\n"
)

ax4.text(
    0.05, 0.95, metrics_text,
    transform=ax4.transAxes,
    fontsize=9,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
)
ax4.set_title("Metrics Summary", fontsize=11, fontweight="bold")

fig_summary.suptitle(
    "Are 3 Action-Decoder T-SNE/HDBSCAN Clusters (PCA=30)\n"
    "Linearly Separable in Action-Decoder Space?",
    fontsize=14, fontweight="bold", y=0.98,
)
fig_summary.savefig(OUTPUT_DIR / "summary.png", dpi=150, bbox_inches="tight")
plt.close(fig_summary)
print("  Saved summary.png")


# ============================================================================
# STEP 7: Save numerical results
# ============================================================================

print(f"\n{'='*80}")
print("STEP 7: Save results")
print("=" * 80)

results = {
    "config": {
        "pca_clustering_dims": 30,
        "hdbscan_min_cluster_size": 200,
        "hdbscan_min_samples": 10,
        "tsne_perplexity": 50,
        "tsne_max_iter": 1000,
        "subsample_rate": 5,
        "knn_k": 5,
    },
    "n_total": int(len(X)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_trajectories": int(len(unique_groups)),
    "n_clusters": n_cls,
    "cluster_distribution": {str(k): int(v) for k, v in label_dist.items()},
    # AD -> AD-cluster probes
    "ad_acc_raw_256": float(acc_raw),
    "ad_f1_raw_256": float(f1_raw),
    "ad_acc_pca_50": float(acc_pca),
    "ad_f1_pca_50": float(f1_pca),
    "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
    "ad_acc_lda": float(acc_lda),
    "ad_acc_shuffle": float(acc_shuffle),
    "ad_f1_shuffle": float(f1_shuffle),
    "confusion_matrix_raw": cm_raw.tolist(),
    "cosine_sim_probe_pca": cos_sim.tolist(),
    "max_abs_cosine_sim": float(np.abs(cos_sim).max()),
    # Per-class metrics
    "per_class_report_raw": {
        str(c): {
            "precision": float(report_raw[str(c)]["precision"]),
            "recall": float(report_raw[str(c)]["recall"]),
            "f1": float(report_raw[str(c)]["f1-score"]),
            "support": int(report_raw[str(c)]["support"]),
        }
        for c in cluster_ids
    },
    # Comparison references
    "comparison": {
        "prior_2cluster_pca20_acc": 0.992,
        "prior_2cluster_pca20_f1": 0.989,
        "lstm_cluster_from_ad_acc": 0.709,
        "lstm_cluster_from_ad_f1": 0.664,
    },
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved results.json")

elapsed = time.time() - t_start

print(f"\n{'='*80}")
print("DONE!")
print(f"{'='*80}")
print(f"  Total time: {elapsed:.0f}s")
print(f"  Output dir: {OUTPUT_DIR}")
print(f"\n  AD -> 3 AD-cluster accuracy (256-dim): {acc_raw:.4f}")
print(f"  AD -> 3 AD-cluster accuracy (PCA-50):  {acc_pca:.4f}")
print(f"  AD -> 3 AD-cluster accuracy (LDA):     {acc_lda:.4f}")
print(f"  Shuffle baseline:                       {acc_shuffle:.4f}")
print(f"  Max |cos_sim| probe vs PCA:             {np.abs(cos_sim).max():.4f}")
print(f"\n  Comparison:")
print(f"    Prior 2-cluster (PCA=20): 99.2% acc, macro F1=0.989")
print(f"    This  3-cluster (PCA=30): {acc_raw:.1%} acc, macro F1={f1_raw:.3f}")
print(f"    LSTM cluster from AD:     70.9% acc, macro F1=0.664")
