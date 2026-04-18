#!/usr/bin/env python
"""
Linear probes to predict action-decoder T-SNE/HDBSCAN clusters from action-decoder activations.

Tests whether the nonlinear clusters found via PCA->T-SNE->HDBSCAN on action-decoder
activations are linearly separable in the same 256-dim action-decoder space. This is the
"same-space" counterpart to Experiment 24 (lstm_cluster_probes), which predicted LSTM
clusters from action-decoder activations (70.9% accuracy).

Bonus: also trains logistic regression on LSTM cell-state activations to predict these
action-decoder clusters (cross-representation probe).

Data:
  - Action-decoder activations: activation_data/yaofeng_200M/activations.cache.npz (36,501 records)
  - LSTM cell-state activations: activation_data/yaofeng_200M_lstm_cell/yaofeng_200M/activations.cache.npz (170,775 records)
  - Cluster labels: generated via PCA(20)->T-SNE 3D->HDBSCAN on action-decoder activations

Run:
    cd /workspace/NeuralMMO-interpretability && \
    OMP_NUM_THREADS=4 uv run python experiments/action_decoder_cluster_probes/run_analysis.py
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

OUTPUT_DIR = REPO_ROOT / "experiments" / "action_decoder_cluster_probes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_JOBS = 4


# ============================================================================
# STEP 1: Generate action-decoder cluster labels
# ============================================================================

def generate_ad_clusters(acts, steps, env_ids, agent_ids):
    """Generate 3-cluster labels via subsample->PCA(20)->T-SNE 3D->HDBSCAN->KNN propagation.

    Replicates the pipeline from experiments 2-3 (yaofeng action-decoder 3-cluster structure).
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.neighbors import KNeighborsClassifier
    import hdbscan as hdbscan_lib

    N = len(acts)
    print(f"  Total points: {N}")

    # Subsample at rate 5, per-trajectory (group by env_id, agent_id, sort by step)
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

    # PCA 20
    pca = PCA(n_components=20, random_state=SEED)
    pca_sub = pca.fit_transform(sub_acts).astype(np.float32)
    print(f"  PCA 20D explains {pca.explained_variance_ratio_.sum():.1%} variance")

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

    # HDBSCAN
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=200, min_samples=3)
    sub_labels = clusterer.fit_predict(emb)
    cluster_ids = sorted(set(sub_labels.tolist()) - {-1})
    noise_pct = (sub_labels == -1).sum() / n_sub * 100
    print(f"  HDBSCAN: {len(cluster_ids)} clusters, {noise_pct:.1f}% noise")
    for cid in cluster_ids:
        print(f"    C{cid}: n={(sub_labels == cid).sum()}")

    # KNN propagate labels to ALL full-res records via PCA-20 space
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
print("STEP 1: Generate action-decoder cluster labels")
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
ad_labels, pca_20 = generate_ad_clusters(ad_acts, ad_steps, ad_env_ids, ad_agent_ids)

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
# For binary classification, sklearn only stores 1 row; replicate for plotting
W = lr_raw.coef_  # (1, 256) for binary, (n_classes, 256) for multiclass
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
# STEP 6: Cross-representation probe (LSTM -> AD clusters)
# ============================================================================

print(f"\n{'='*80}")
print("STEP 6: Cross-representation probe (LSTM cell-state -> AD clusters)")
print("=" * 80)

lstm_cache = np.load(
    "activation_data/yaofeng_200M_lstm_cell/yaofeng_200M/activations.cache.npz"
)
lstm_acts = lstm_cache["activations"]     # (170775, 256)
lstm_env_ids = lstm_cache["env_ids"]      # (170775,)
lstm_agent_ids = lstm_cache["agent_ids"]  # (170775,)
lstm_steps = lstm_cache["steps"]          # (170775,)

print(f"  LSTM cache: {lstm_acts.shape[0]} records, {lstm_acts.shape[1]}-dim")

# Build lookup: (env_id, agent_id, step) -> ad_cluster_label
ad_label_lookup = {}
for i in range(len(ad_env_ids)):
    key = (int(ad_env_ids[i]), int(ad_agent_ids[i]), int(ad_steps[i]))
    ad_label_lookup[key] = int(ad_labels[i])

# Join LSTM records with AD cluster labels
lstm_matched_idx = []
lstm_matched_labels = []
for i in range(len(lstm_env_ids)):
    key = (int(lstm_env_ids[i]), int(lstm_agent_ids[i]), int(lstm_steps[i]))
    if key in ad_label_lookup:
        lstm_matched_idx.append(i)
        lstm_matched_labels.append(ad_label_lookup[key])

lstm_matched_idx = np.array(lstm_matched_idx)
lstm_matched_labels = np.array(lstm_matched_labels)
X_lstm = lstm_acts[lstm_matched_idx]
y_lstm = lstm_matched_labels
lstm_env_matched = lstm_env_ids[lstm_matched_idx]
lstm_agent_matched = lstm_agent_ids[lstm_matched_idx]

print(f"  Matched LSTM samples: {len(X_lstm)}")
print(f"  Label distribution: {dict(zip(*np.unique(y_lstm, return_counts=True)))}")

# Trajectory-based split for LSTM cross-probe
groups_lstm = np.array([f"{e}_{a}" for e, a in zip(lstm_env_matched, lstm_agent_matched)])
unique_groups_lstm = np.unique(groups_lstm)
print(f"  Unique LSTM trajectories: {len(unique_groups_lstm)}")

splitter_lstm = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx_lstm, test_idx_lstm = next(splitter_lstm.split(X_lstm, y_lstm, groups_lstm))

X_lstm_train, X_lstm_test = X_lstm[train_idx_lstm], X_lstm[test_idx_lstm]
y_lstm_train, y_lstm_test = y_lstm[train_idx_lstm], y_lstm[test_idx_lstm]

print(f"  LSTM Train: {len(X_lstm_train)} / Test: {len(X_lstm_test)}")
print(f"  LSTM Train dist: {dict(zip(*np.unique(y_lstm_train, return_counts=True)))}")
print(f"  LSTM Test dist:  {dict(zip(*np.unique(y_lstm_test, return_counts=True)))}")

scaler_lstm = StandardScaler()
X_lstm_train_s = scaler_lstm.fit_transform(X_lstm_train)
X_lstm_test_s = scaler_lstm.transform(X_lstm_test)

lr_lstm = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=SEED,
    n_jobs=N_JOBS,
)
lr_lstm.fit(X_lstm_train_s, y_lstm_train)
y_pred_lstm = lr_lstm.predict(X_lstm_test_s)
acc_lstm_cross = accuracy_score(y_lstm_test, y_pred_lstm)
f1_lstm_cross = f1_score(y_lstm_test, y_pred_lstm, average="macro")
print(f"  LSTM -> AD-cluster accuracy: {acc_lstm_cross:.4f}")
print(f"  LSTM -> AD-cluster macro F1: {f1_lstm_cross:.4f}")
print(classification_report(y_lstm_test, y_pred_lstm, digits=4))


# ============================================================================
# STEP 7: Generate plots
# ============================================================================

print(f"\n{'='*80}")
print("STEP 7: Generate plots")
print("=" * 80)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

cluster_ids = sorted(set(y.tolist()))
n_cls = len(cluster_ids)
cluster_names = [f"Cluster {c}" for c in cluster_ids]
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:n_cls]

# --- 7a. Confusion matrix ---
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
im = ax_cm.imshow(cm_raw, interpolation="nearest", cmap="Blues")
ax_cm.set_title("Confusion Matrix\n(Logistic Regression, 256-dim AD -> AD clusters)", fontsize=12)
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
        ax_cm.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10)
fig_cm.colorbar(im, ax=ax_cm)
fig_cm.tight_layout()
fig_cm.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(fig_cm)
print("  Saved confusion_matrix.png")

# --- 7b. LDA projection ---
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
ax_lda.set_title("LDA Projection of Action-Decoder Activations\nColored by AD Cluster", fontsize=12)
ax_lda.legend(fontsize=9, markerscale=3)
fig_lda.tight_layout()
fig_lda.savefig(OUTPUT_DIR / "lda_projection.png", dpi=150, bbox_inches="tight")
plt.close(fig_lda)
print("  Saved lda_projection.png")

# --- 7c. Probe weights vs PCA directions ---
probe_row_labels = [f"Weight {i}" for i in range(n_probe_rows)]
fig_pw, ax_pw = plt.subplots(figsize=(10, max(3, n_probe_rows + 1)))
im_pw = ax_pw.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_pw.set_xlabel("PCA Component", fontsize=11)
ax_pw.set_ylabel("Probe Weight Vector", fontsize=11)
ax_pw.set_yticks(range(n_probe_rows))
ax_pw.set_yticklabels(probe_row_labels, fontsize=9)
ax_pw.set_xticks(range(20))
ax_pw.set_xticklabels([f"PC{i}" for i in range(20)], fontsize=8, rotation=45)
ax_pw.set_title("Cosine Similarity: Probe Weights vs Top-20 PCA Directions", fontsize=12)
fig_pw.colorbar(im_pw, ax=ax_pw, label="Cosine Similarity")
for i in range(n_probe_rows):
    for j in range(20):
        if abs(cos_sim[i, j]) > 0.15:
            ax_pw.text(j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig_pw.tight_layout()
fig_pw.savefig(OUTPUT_DIR / "probe_weights_pca.png", dpi=150, bbox_inches="tight")
plt.close(fig_pw)
print("  Saved probe_weights_pca.png")

# --- 7d. Summary (2x2 grid) ---
fig_summary = plt.figure(figsize=(16, 14))
gs = GridSpec(2, 2, figure=fig_summary, hspace=0.35, wspace=0.3)

# Top-left: Confusion matrix
ax1 = fig_summary.add_subplot(gs[0, 0])
im1 = ax1.imshow(cm_raw, interpolation="nearest", cmap="Blues")
ax1.set_title("Confusion Matrix (256-dim LogReg)\nAD activations -> AD clusters", fontsize=11, fontweight="bold")
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
        ax1.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9)
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
ax2.set_title("LDA Projection (Action-Decoder Space)\nColored by AD cluster", fontsize=11, fontweight="bold")
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
            ax3.text(j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center", fontsize=6)

# Bottom-right: Metrics text panel
ax4 = fig_summary.add_subplot(gs[1, 1])
ax4.axis("off")

# Per-class metrics from classification report
report_raw = classification_report(y_test, y_pred_raw, output_dict=True, digits=4)
report_pca = classification_report(y_test, y_pred_pca, output_dict=True, digits=4)

metrics_text = (
    "Action-Decoder Cluster Probe Results\n"
    "=" * 44 + "\n\n"
    f"Total samples: {len(X):,}\n"
    f"Train / Test: {len(X_train):,} / {len(X_test):,}\n"
    f"Trajectories: {len(unique_groups)} total\n"
    f"Clusters: {n_cls}\n\n"
    "--- AD -> AD-cluster (LogReg 256-dim) ---\n"
    f"  Accuracy:  {acc_raw:.4f}\n"
    f"  Macro F1:  {f1_raw:.4f}\n"
)
for c in [str(c) for c in cluster_ids]:
    p = report_raw[c]["precision"]
    r = report_raw[c]["recall"]
    f = report_raw[c]["f1-score"]
    metrics_text += f"  Cluster {c}: P={p:.3f} R={r:.3f} F1={f:.3f}\n"

metrics_text += (
    f"\n--- AD -> AD-cluster (LogReg PCA-50) ---\n"
    f"  Accuracy:  {acc_pca:.4f}\n"
    f"  Macro F1:  {f1_pca:.4f}\n"
    f"  PCA var:   {pca.explained_variance_ratio_.sum():.4f}\n"
)

metrics_text += (
    f"\n--- LDA Classification ---\n"
    f"  Accuracy:  {acc_lda:.4f}\n"
)

metrics_text += (
    f"\n--- LSTM -> AD-cluster (cross-repr) ---\n"
    f"  Accuracy:  {acc_lstm_cross:.4f}\n"
    f"  Macro F1:  {f1_lstm_cross:.4f}\n"
    f"  (joined on {len(X_lstm):,} samples)\n"
)

metrics_text += (
    f"\n--- Shuffle Baseline ---\n"
    f"  Accuracy:  {acc_shuffle:.4f}\n"
    f"  Macro F1:  {f1_shuffle:.4f}\n"
)

# Comparison with LSTM cluster probes (Exp 24)
metrics_text += (
    f"\n{'='*44}\n"
    f"COMPARISON:\n"
    f"  LSTM-cluster from AD (Exp 24): 70.9%\n"
    f"  AD-cluster from AD (this):     {acc_raw:.1%}\n"
    f"  AD-cluster from LSTM (cross):  {acc_lstm_cross:.1%}\n"
)

if acc_raw > 0.9:
    verdict = "YES - clusters are nearly linear"
elif acc_raw > 0.7:
    verdict = "PARTIALLY - some nonlinear structure"
else:
    verdict = "NO - clusters are genuinely nonlinear"
metrics_text += (
    f"\nAD clusters linearly separable? {verdict}\n"
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
    "Are Action-Decoder T-SNE/HDBSCAN Clusters Linearly Separable\n"
    "in Action-Decoder Space?",
    fontsize=14, fontweight="bold", y=0.98,
)
fig_summary.savefig(OUTPUT_DIR / "summary.png", dpi=150, bbox_inches="tight")
plt.close(fig_summary)
print("  Saved summary.png")


# ============================================================================
# STEP 8: Save numerical results
# ============================================================================

print(f"\n{'='*80}")
print("STEP 8: Save results")
print("=" * 80)

results = {
    "n_total": int(len(X)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_trajectories": int(len(unique_groups)),
    "n_clusters": n_cls,
    "cluster_distribution": {str(k): int(v) for k, v in label_dist.items()},
    # AD -> AD-cluster probes
    "ad_ad_acc_raw_256": float(acc_raw),
    "ad_ad_f1_raw_256": float(f1_raw),
    "ad_ad_acc_pca_50": float(acc_pca),
    "ad_ad_f1_pca_50": float(f1_pca),
    "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
    "ad_ad_acc_lda": float(acc_lda),
    "ad_ad_acc_shuffle": float(acc_shuffle),
    "ad_ad_f1_shuffle": float(f1_shuffle),
    "confusion_matrix_raw": cm_raw.tolist(),
    "cosine_sim_probe_pca": cos_sim.tolist(),
    "max_abs_cosine_sim": float(np.abs(cos_sim).max()),
    # LSTM -> AD-cluster cross-probe
    "lstm_ad_n_matched": int(len(X_lstm)),
    "lstm_ad_acc": float(acc_lstm_cross),
    "lstm_ad_f1": float(f1_lstm_cross),
    # Reference: LSTM cluster probe (Exp 24)
    "ref_lstm_cluster_probe_acc": 0.709,
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
print(f"\n  AD -> AD-cluster accuracy (256-dim): {acc_raw:.4f}")
print(f"  AD -> AD-cluster accuracy (PCA-50):  {acc_pca:.4f}")
print(f"  AD -> AD-cluster accuracy (LDA):     {acc_lda:.4f}")
print(f"  LSTM -> AD-cluster accuracy:          {acc_lstm_cross:.4f}")
print(f"  Shuffle baseline:                     {acc_shuffle:.4f}")
print(f"  Max |cos_sim| probe vs PCA:           {np.abs(cos_sim).max():.4f}")
print(f"\n  Comparison: LSTM-cluster from AD (Exp 24) was 70.9%")
print(f"  This experiment: AD-cluster from AD = {acc_raw:.1%}")
