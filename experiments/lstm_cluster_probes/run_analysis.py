"""
Linear probes to predict LSTM cell-state clusters from action-decoder activations.

Tests whether the behavioral modes identified in LSTM cell state (3 clusters from
KMeans on UMAP of LSTM cell states) are also linearly readable from the feed-forward
action-decoder hidden state (256-dim).

Data:
  - Action-decoder activations: activation_data/yaofeng_200M/activations.cache.npz
  - LSTM cluster labels: experiments/yaofeng_lstm_cell_lifetime/cluster_labels.npy
  - Join on (env_id, agent_id, step) -> ~25,920 matched samples
"""

import os
import sys
import numpy as np
from pathlib import Path

# Ensure we run from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)

OUTPUT_DIR = REPO_ROOT / "experiments" / "lstm_cluster_probes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load and join ──────────────────────────────────────────────────────────

print("=== Step 1: Load and join data ===")

cache = np.load("activation_data/yaofeng_200M/activations.cache.npz")
activations_all = cache["activations"]  # (36501, 256)
env_ids_cache = cache["env_ids"]        # (36501,)
agent_ids_cache = cache["agent_ids"]    # (36501,)
steps_cache = cache["steps"]            # (36501,)

labels_all = np.load("experiments/yaofeng_lstm_cell_lifetime/cluster_labels.npy")
# columns: [env_id, agent_id, step, label]

print(f"  Action-decoder cache: {activations_all.shape[0]} records, {activations_all.shape[1]}-dim")
print(f"  LSTM cluster labels: {labels_all.shape[0]} records")

# Build lookup: (env_id, agent_id, step) -> cluster_label
label_lookup = {}
for row in labels_all:
    key = (int(row[0]), int(row[1]), int(row[2]))
    label_lookup[key] = int(row[3])

# Join
matched_indices = []
matched_labels = []
for i in range(len(env_ids_cache)):
    key = (int(env_ids_cache[i]), int(agent_ids_cache[i]), int(steps_cache[i]))
    if key in label_lookup:
        matched_indices.append(i)
        matched_labels.append(label_lookup[key])

matched_indices = np.array(matched_indices)
X = activations_all[matched_indices]  # (N, 256)
y = np.array(matched_labels)          # (N,) in {0, 1, 2}
matched_env_ids = env_ids_cache[matched_indices]
matched_agent_ids = agent_ids_cache[matched_indices]

print(f"  Matched samples: {len(X)}")
print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ── 2. Trajectory-based train/test split ──────────────────────────────────────

print("\n=== Step 2: Trajectory-based train/test split ===")

from sklearn.model_selection import GroupShuffleSplit

# Group by (env_id, agent_id) = trajectory
groups = np.array([f"{e}_{a}" for e, a in zip(matched_env_ids, matched_agent_ids)])
unique_groups = np.unique(groups)
print(f"  Unique trajectories: {len(unique_groups)}")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"  Train: {len(X_train)} samples ({len(np.unique(groups[train_idx]))} trajectories)")
print(f"  Test:  {len(X_test)} samples ({len(np.unique(groups[test_idx]))} trajectories)")
print(f"  Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"  Test  label dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# ── 3. Probes ─────────────────────────────────────────────────────────────────

print("\n=== Step 3: Train logistic regression probes ===")

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
    random_state=42,
    n_jobs=4,
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
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"    PCA explained variance (50 components): {pca.explained_variance_ratio_.sum():.4f}")

lr_pca = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
    n_jobs=4,
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
rng = np.random.RandomState(42)
y_train_shuffled = rng.permutation(y_train)
lr_shuffle = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    random_state=42,
    n_jobs=4,
)
lr_shuffle.fit(X_train_scaled, y_train_shuffled)
y_pred_shuffle = lr_shuffle.predict(X_test_scaled)
acc_shuffle = accuracy_score(y_test, y_pred_shuffle)
f1_shuffle = f1_score(y_test, y_pred_shuffle, average="macro")
print(f"    Shuffle accuracy: {acc_shuffle:.4f}")
print(f"    Shuffle macro F1: {f1_shuffle:.4f}")

# ── 4. Linear Discriminant Analysis ──────────────────────────────────────────

print("\n=== Step 4: LDA projection ===")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda_train = lda.fit_transform(X_train_scaled, y_train)
X_lda_test = lda.transform(X_test_scaled)

# LDA classification accuracy (for reference)
y_pred_lda = lda.predict(X_test_scaled)
acc_lda = accuracy_score(y_test, y_pred_lda)
print(f"  LDA classification accuracy: {acc_lda:.4f}")

# Full dataset LDA projection for plotting
X_all_scaled = scaler.transform(X)
X_lda_all = lda.transform(X_all_scaled)

# ── 5. Probe direction analysis ──────────────────────────────────────────────

print("\n=== Step 5: Probe weight vs PCA direction analysis ===")

# Weight vectors from raw logistic regression (3 classes x 256 dims)
W = lr_raw.coef_  # (3, 256)

# Top-20 PCA directions (in original scaled space)
pca_full = PCA(n_components=20, random_state=42)
pca_full.fit(X_train_scaled)
pca_dirs = pca_full.components_  # (20, 256)

# Cosine similarity between each probe weight vector and each PCA direction
from sklearn.metrics.pairwise import cosine_similarity

cos_sim = cosine_similarity(W, pca_dirs)  # (3, 20)
print(f"  Probe weight shapes: {W.shape}")
print(f"  PCA directions shape: {pca_dirs.shape}")
print(f"  Cosine similarity matrix shape: {cos_sim.shape}")
print(f"  Max |cos_sim| per probe class: {np.abs(cos_sim).max(axis=1)}")
print(f"  PCA component with max alignment per class: {np.abs(cos_sim).argmax(axis=1)}")

# ── 6. Generate plots ────────────────────────────────────────────────────────

print("\n=== Step 6: Generate plots ===")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

cluster_names = ["Cluster 0", "Cluster 1", "Cluster 2"]
cluster_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# --- 6a. Confusion matrix ---
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
im = ax_cm.imshow(cm_raw, interpolation="nearest", cmap="Blues")
ax_cm.set_title("Confusion Matrix\n(Logistic Regression, 256-dim)", fontsize=12)
ax_cm.set_xlabel("Predicted", fontsize=11)
ax_cm.set_ylabel("True", fontsize=11)
ax_cm.set_xticks([0, 1, 2])
ax_cm.set_yticks([0, 1, 2])
ax_cm.set_xticklabels(cluster_names, fontsize=9)
ax_cm.set_yticklabels(cluster_names, fontsize=9)
# Annotate cells
for i in range(3):
    for j in range(3):
        val = cm_raw[i, j]
        total_row = cm_raw[i].sum()
        pct = val / total_row * 100
        color = "white" if val > cm_raw.max() / 2 else "black"
        ax_cm.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10)
fig_cm.colorbar(im, ax=ax_cm)
fig_cm.tight_layout()
fig_cm.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
print("  Saved confusion_matrix.png")

# --- 6b. LDA projection ---
fig_lda, ax_lda = plt.subplots(figsize=(8, 6))
for label in [0, 1, 2]:
    mask = y == label
    ax_lda.scatter(
        X_lda_all[mask, 0],
        X_lda_all[mask, 1],
        c=cluster_colors[label],
        label=f"Cluster {label} (n={mask.sum()})",
        alpha=0.3,
        s=8,
        edgecolors="none",
    )
ax_lda.set_xlabel("LDA Component 1", fontsize=11)
ax_lda.set_ylabel("LDA Component 2", fontsize=11)
ax_lda.set_title("LDA Projection of Action-Decoder Activations\nColored by LSTM Cluster", fontsize=12)
ax_lda.legend(fontsize=9, markerscale=3)
fig_lda.tight_layout()
fig_lda.savefig(OUTPUT_DIR / "lda_projection.png", dpi=150, bbox_inches="tight")
print("  Saved lda_projection.png")

# --- 6c. Probe weights vs PCA directions ---
fig_pw, ax_pw = plt.subplots(figsize=(10, 4))
im_pw = ax_pw.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_pw.set_xlabel("PCA Component", fontsize=11)
ax_pw.set_ylabel("Probe Weight Vector", fontsize=11)
ax_pw.set_yticks([0, 1, 2])
ax_pw.set_yticklabels(cluster_names, fontsize=9)
ax_pw.set_xticks(range(20))
ax_pw.set_xticklabels([f"PC{i}" for i in range(20)], fontsize=8, rotation=45)
ax_pw.set_title("Cosine Similarity: Probe Weights vs Top-20 PCA Directions", fontsize=12)
fig_pw.colorbar(im_pw, ax=ax_pw, label="Cosine Similarity")
# Annotate significant values
for i in range(3):
    for j in range(20):
        if abs(cos_sim[i, j]) > 0.15:
            ax_pw.text(j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig_pw.tight_layout()
fig_pw.savefig(OUTPUT_DIR / "probe_weights_pca.png", dpi=150, bbox_inches="tight")
print("  Saved probe_weights_pca.png")

# --- 6d. Summary (2x2 grid) ---
fig_summary = plt.figure(figsize=(16, 14))
gs = GridSpec(2, 2, figure=fig_summary, hspace=0.35, wspace=0.3)

# Top-left: Confusion matrix
ax1 = fig_summary.add_subplot(gs[0, 0])
im1 = ax1.imshow(cm_raw, interpolation="nearest", cmap="Blues")
ax1.set_title("Confusion Matrix (256-dim LogReg)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("True")
ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])
ax1.set_xticklabels(cluster_names, fontsize=8)
ax1.set_yticklabels(cluster_names, fontsize=8)
for i in range(3):
    for j in range(3):
        val = cm_raw[i, j]
        total_row = cm_raw[i].sum()
        pct = val / total_row * 100
        color = "white" if val > cm_raw.max() / 2 else "black"
        ax1.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9)
fig_summary.colorbar(im1, ax=ax1, fraction=0.046)

# Top-right: LDA projection
ax2 = fig_summary.add_subplot(gs[0, 1])
for label in [0, 1, 2]:
    mask = y == label
    ax2.scatter(
        X_lda_all[mask, 0],
        X_lda_all[mask, 1],
        c=cluster_colors[label],
        label=f"Cluster {label}",
        alpha=0.3,
        s=6,
        edgecolors="none",
    )
ax2.set_xlabel("LDA Component 1")
ax2.set_ylabel("LDA Component 2")
ax2.set_title("LDA Projection (Action-Decoder Space)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8, markerscale=3)

# Bottom-left: Probe weights vs PCA
ax3 = fig_summary.add_subplot(gs[1, 0])
im3 = ax3.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax3.set_xlabel("PCA Component")
ax3.set_ylabel("Probe Weight")
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(cluster_names, fontsize=8)
ax3.set_xticks(range(20))
ax3.set_xticklabels([f"PC{i}" for i in range(20)], fontsize=7, rotation=45)
ax3.set_title("Probe Weights vs PCA Directions", fontsize=11, fontweight="bold")
fig_summary.colorbar(im3, ax=ax3, fraction=0.046, label="Cosine Sim.")
for i in range(3):
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
    "LSTM Cluster Probe Results\n"
    "=" * 40 + "\n\n"
    f"Matched samples: {len(X):,}\n"
    f"Train / Test: {len(X_train):,} / {len(X_test):,}\n"
    f"Trajectories: {len(unique_groups)} total\n\n"
    "--- Logistic Regression (256-dim) ---\n"
    f"  Accuracy:  {acc_raw:.4f}\n"
    f"  Macro F1:  {f1_raw:.4f}\n"
)
for c in ["0", "1", "2"]:
    p = report_raw[c]["precision"]
    r = report_raw[c]["recall"]
    f = report_raw[c]["f1-score"]
    metrics_text += f"  Cluster {c}: P={p:.3f} R={r:.3f} F1={f:.3f}\n"

metrics_text += (
    f"\n--- Logistic Regression (PCA-50) ---\n"
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

verdict = "YES" if acc_raw > 0.9 else ("PARTIALLY" if acc_raw > 0.7 else "NO")
metrics_text += (
    f"\n{'=' * 40}\n"
    f"VERDICT: LSTM clusters linearly decodable\n"
    f"from action-decoder space? {verdict}\n"
    f"(threshold: >90% accuracy)\n"
)

ax4.text(
    0.05, 0.95, metrics_text,
    transform=ax4.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
)
ax4.set_title("Metrics Summary", fontsize=11, fontweight="bold")

fig_summary.suptitle(
    "Can Action-Decoder Activations Predict LSTM Cell-State Clusters?",
    fontsize=14, fontweight="bold", y=0.98,
)
fig_summary.savefig(OUTPUT_DIR / "summary.png", dpi=150, bbox_inches="tight")
print("  Saved summary.png")

plt.close("all")

# ── 7. Save numerical results ────────────────────────────────────────────────

print("\n=== Step 7: Save results ===")

results = {
    "n_matched": int(len(X)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_trajectories": int(len(unique_groups)),
    "acc_raw_256": float(acc_raw),
    "f1_raw_256": float(f1_raw),
    "acc_pca_50": float(acc_pca),
    "f1_pca_50": float(f1_pca),
    "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
    "acc_lda": float(acc_lda),
    "acc_shuffle": float(acc_shuffle),
    "f1_shuffle": float(f1_shuffle),
    "confusion_matrix": cm_raw.tolist(),
    "cosine_sim_probe_pca": cos_sim.tolist(),
}

import json

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved results.json")

print("\n=== Done! ===")
print(f"  All outputs in: {OUTPUT_DIR}")
print(f"  Accuracy (256-dim): {acc_raw:.4f}")
print(f"  Accuracy (PCA-50):  {acc_pca:.4f}")
print(f"  Shuffle baseline:   {acc_shuffle:.4f}")
