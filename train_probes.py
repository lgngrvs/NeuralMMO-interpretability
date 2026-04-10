"""Train linear probes to find directions in activation space for behavioral metrics.

For each feature from analyze_activations.py, trains a linear probe
(ridge regression for continuous, logistic regression for binary) on the 256-dim
activation vectors.  Evaluates on held-out *trajectories* to avoid temporal leakage.

Reports:
  - Continuous features:  R², MAE
  - Binary features:      Accuracy, AUC, Precision, Recall, F1
  - Baselines:            Shuffle (permuted labels) and PCA-reduced activations

Saves learned weight vectors (the "directions") and a summary visualization.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Reuse data loading and feature extraction from the analysis pipeline
from analyze_activations import FEATURE_NAMES, compute_features, load_data

# Features that are binary (0/1) — use logistic regression
BINARY_FEATURES = {"in_combat", "is_moving", "is_attacking", "is_trading", "is_using_item"}


# ---------------------------------------------------------------------------
# Train / test split by trajectory
# ---------------------------------------------------------------------------

def trajectory_split(metadata, test_frac=0.2, seed=42):
    """Split indices by trajectory so no (env_id, agent_id) straddles train/test."""
    rng = np.random.RandomState(seed)
    traj_keys = list(
        set(zip(metadata["env_id"].tolist(), metadata["agent_id"].tolist()))
    )
    rng.shuffle(traj_keys)
    n_test = max(1, int(len(traj_keys) * test_frac))
    test_set = set(map(tuple, traj_keys[:n_test]))

    train_idx, test_idx = [], []
    for i in range(len(metadata["env_id"])):
        key = (int(metadata["env_id"][i]), int(metadata["agent_id"][i]))
        if key in test_set:
            test_idx.append(i)
        else:
            train_idx.append(i)

    return np.array(train_idx), np.array(test_idx)


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def _compute_sample_weights_balanced(y, n_bins=10):
    """Compute sample weights inversely proportional to quantile-bin frequency."""
    try:
        bin_edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.clip(np.digitize(y, bin_edges, right=True), 1, n_bins)
    except Exception:
        return np.ones(len(y))
    bin_counts = np.maximum(np.bincount(bin_indices, minlength=n_bins + 1), 1)
    return len(y) / (n_bins * bin_counts[bin_indices].astype(float))


def train_probe(X_train, y_train, X_test, y_test, binary=False, alpha=1.0,
                class_weight=None, seed=42):
    """Train a single linear probe and return metrics + weight vector.

    Returns dict with metrics and the learned direction (coef).
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    result = {}

    if binary:
        # Guard against single-class splits
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return None

        model = LogisticRegression(
            C=1.0 / max(alpha, 1e-8),
            max_iter=2000,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=seed,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        result["accuracy"] = accuracy_score(y_test, y_pred)
        result["auc"] = roc_auc_score(y_test, y_prob)
        result["precision"] = precision_score(y_test, y_pred, zero_division=0)
        result["recall"] = recall_score(y_test, y_pred, zero_division=0)
        result["f1"] = f1_score(y_test, y_pred, zero_division=0)
        result["coef"] = model.coef_[0]  # shape (d,)
        result["intercept"] = model.intercept_[0]
        # Proportion of positive class in train/test for context
        result["train_pos_rate"] = y_train.mean()
        result["test_pos_rate"] = y_test.mean()
    else:
        model = Ridge(alpha=alpha, random_state=seed)
        sample_weight = None
        if class_weight == "balanced":
            sample_weight = _compute_sample_weights_balanced(y_train)
        model.fit(X_train_s, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_test_s)

        result["r2"] = r2_score(y_test, y_pred)
        result["mae"] = mean_absolute_error(y_test, y_pred)
        result["coef"] = model.coef_  # shape (d,)
        result["intercept"] = model.intercept_
        result["y_std"] = float(y_test.std())

    result["scaler_mean"] = scaler.mean_
    result["scaler_scale"] = scaler.scale_
    return result


def train_all_probes(activations, features, metadata, n_pca=50, alpha=1.0,
                     test_frac=0.2, class_weight=None, seed=42):
    """Train probes for all features under three conditions: full, PCA, shuffle.

    Returns a dict: {feature_name: {"full": result, "pca": result, "shuffle": result}}
    """
    from sklearn.decomposition import PCA

    train_idx, test_idx = trajectory_split(metadata, test_frac=test_frac, seed=seed)
    print(f"Split: {len(train_idx)} train, {len(test_idx)} test "
          f"({len(test_idx) / (len(train_idx) + len(test_idx)) * 100:.1f}% test)")

    # PCA reduction
    pca = PCA(n_components=min(n_pca, activations.shape[1]), random_state=seed)
    act_pca = pca.fit_transform(activations)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {pca.n_components_} components, {var_explained * 100:.1f}% variance explained")

    results = {}
    rng = np.random.RandomState(seed)

    for fi, fname in enumerate(FEATURE_NAMES):
        binary = fname in BINARY_FEATURES
        y = features[:, fi]

        y_train, y_test = y[train_idx], y[test_idx]

        conditions = {
            "full": (activations[train_idx], activations[test_idx], y_train, y_test),
            "pca": (act_pca[train_idx], act_pca[test_idx], y_train, y_test),
        }

        # Shuffle baseline: permute train labels, keep test labels real
        y_train_shuf = y_train.copy()
        rng.shuffle(y_train_shuf)
        conditions["shuffle"] = (
            activations[train_idx], activations[test_idx], y_train_shuf, y_test
        )

        results[fname] = {}
        for cond_name, (Xtr, Xte, ytr, yte) in conditions.items():
            res = train_probe(Xtr, ytr, Xte, yte, binary=binary, alpha=alpha,
                              class_weight=class_weight, seed=seed)
            results[fname][cond_name] = res

        # Progress line
        if binary:
            full = results[fname]["full"]
            tag = f"AUC={full['auc']:.3f}" if full else "SKIP"
        else:
            full = results[fname]["full"]
            tag = f"R²={full['r2']:.3f}" if full else "SKIP"
        print(f"  {fname:25s}  {tag}")

    return results, pca


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(results, output_dir):
    """Create summary visualization of probe results."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    continuous = [f for f in FEATURE_NAMES if f not in BINARY_FEATURES]
    binary = [f for f in FEATURE_NAMES if f in BINARY_FEATURES]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- Continuous features: R² ---
    ax = axes[0]
    names_c = []
    vals_full, vals_pca, vals_shuf = [], [], []
    for f in continuous:
        r = results[f]
        if r["full"] is None:
            continue
        names_c.append(f)
        vals_full.append(r["full"]["r2"])
        vals_pca.append(r["pca"]["r2"])
        vals_shuf.append(r["shuffle"]["r2"] if r["shuffle"] else 0)

    x = np.arange(len(names_c))
    w = 0.25

    # Clip y-axis to a readable range; annotate outliers
    all_vals = vals_full + vals_pca + vals_shuf
    y_min = min(-0.15, min(v for v in all_vals if v >= -1.0) - 0.05)
    y_max = max(all_vals) + 0.05

    # Clamp bars for display and annotate out-of-range values
    def clamp_and_annotate(ax, x_pos, vals, y_min):
        for xi, v in zip(x_pos, vals):
            if v < y_min:
                ax.annotate(f"{v:.1f}", xy=(xi, y_min), fontsize=7,
                            ha="center", va="bottom", color="red",
                            fontweight="bold")

    vals_shuf_clamped = [max(v, y_min) for v in vals_shuf]
    vals_pca_clamped = [max(v, y_min) for v in vals_pca]
    vals_full_clamped = [max(v, y_min) for v in vals_full]

    ax.bar(x - w, vals_full_clamped, w, label="Full (256d)", color="#2563eb")
    ax.bar(x, vals_pca_clamped, w, label="PCA", color="#7c3aed")
    ax.bar(x + w, vals_shuf_clamped, w, label="Shuffle", color="#9ca3af")
    clamp_and_annotate(ax, x - w, vals_full, y_min)
    clamp_and_annotate(ax, x, vals_pca, y_min)
    clamp_and_annotate(ax, x + w, vals_shuf, y_min)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(names_c, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Linear Probe Performance — Continuous Features (R²)")
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5)

    # --- Binary features: AUC ---
    ax = axes[1]
    names_b = []
    auc_full, auc_pca, auc_shuf = [], [], []
    for f in binary:
        r = results[f]
        if r["full"] is None:
            continue
        names_b.append(f)
        auc_full.append(r["full"]["auc"])
        auc_pca.append(r["pca"]["auc"])
        auc_shuf.append(r["shuffle"]["auc"] if r["shuffle"] else 0.5)

    x = np.arange(len(names_b))
    ax.bar(x - w, auc_full, w, label="Full (256d)", color="#2563eb")
    ax.bar(x, auc_pca, w, label="PCA", color="#7c3aed")
    ax.bar(x + w, auc_shuf, w, label="Shuffle", color="#9ca3af")
    ax.set_xticks(x)
    ax.set_xticklabels(names_b, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_title("Linear Probe Performance — Binary Features (AUC)")
    ax.legend()
    ax.axhline(0.5, color="k", linewidth=0.5, linestyle="--", label="chance")

    plt.tight_layout()
    fig.savefig(output_dir / "probe_summary.png", dpi=180)
    print(f"Saved {output_dir / 'probe_summary.png'}")
    plt.close(fig)

    # --- Detailed metrics table for binary features ---
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.axis("off")
    if names_b:
        cell_text = []
        for f in names_b:
            r = results[f]["full"]
            if r is None:
                continue
            cell_text.append([
                f,
                f"{r['accuracy']:.3f}",
                f"{r['auc']:.3f}",
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['f1']:.3f}",
                f"{r['train_pos_rate']:.2%}",
                f"{r['test_pos_rate']:.2%}",
            ])
        col_labels = ["Feature", "Accuracy", "AUC", "Precision", "Recall",
                       "F1", "Train +rate", "Test +rate"]
        table = ax2.table(cellText=cell_text, colLabels=col_labels,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        ax2.set_title("Binary Feature Probe — Detailed Metrics", pad=20)

    plt.tight_layout()
    fig2.savefig(output_dir / "probe_binary_details.png", dpi=180)
    print(f"Saved {output_dir / 'probe_binary_details.png'}")
    plt.close(fig2)

    # --- Direction cosine similarity heatmap ---
    directions = {}
    for f in FEATURE_NAMES:
        r = results[f]["full"]
        if r is not None:
            d = r["coef"]
            directions[f] = d / (np.linalg.norm(d) + 1e-12)

    if len(directions) > 1:
        dir_names = list(directions.keys())
        mat = np.zeros((len(dir_names), len(dir_names)))
        for i, a in enumerate(dir_names):
            for j, b in enumerate(dir_names):
                mat[i, j] = np.dot(directions[a], directions[b])

        fig3, ax3 = plt.subplots(figsize=(10, 8))
        im = ax3.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax3.set_xticks(range(len(dir_names)))
        ax3.set_yticks(range(len(dir_names)))
        ax3.set_xticklabels(dir_names, rotation=45, ha="right", fontsize=9)
        ax3.set_yticklabels(dir_names, fontsize=9)
        for i in range(len(dir_names)):
            for j in range(len(dir_names)):
                ax3.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                         fontsize=7, color="white" if abs(mat[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax3, label="Cosine similarity")
        ax3.set_title("Cosine Similarity Between Learned Probe Directions")
        plt.tight_layout()
        fig3.savefig(output_dir / "probe_direction_similarity.png", dpi=180)
        print(f"Saved {output_dir / 'probe_direction_similarity.png'}")
        plt.close(fig3)


def print_summary(results):
    """Print a text summary table of all probe results."""
    print("\n" + "=" * 90)
    print("LINEAR PROBE RESULTS SUMMARY")
    print("=" * 90)

    print(f"\n{'Feature':25s} {'Type':6s} {'Metric':6s} {'Full':>8s} {'PCA':>8s} {'Shuffle':>8s}")
    print("-" * 70)

    for fname in FEATURE_NAMES:
        r = results[fname]
        binary = fname in BINARY_FEATURES

        if r["full"] is None:
            print(f"{fname:25s} {'bin' if binary else 'cont':6s}  SKIPPED (single-class)")
            continue

        if binary:
            metric = "AUC"
            v_full = r["full"]["auc"]
            v_pca = r["pca"]["auc"] if r["pca"] else float("nan")
            v_shuf = r["shuffle"]["auc"] if r["shuffle"] else float("nan")
        else:
            metric = "R²"
            v_full = r["full"]["r2"]
            v_pca = r["pca"]["r2"] if r["pca"] else float("nan")
            v_shuf = r["shuffle"]["r2"] if r["shuffle"] else float("nan")

        print(f"{fname:25s} {'bin' if binary else 'cont':6s} {metric:6s} "
              f"{v_full:8.4f} {v_pca:8.4f} {v_shuf:8.4f}")

    # Extra detail for binary features
    print("\nBinary feature detail:")
    print(f"{'Feature':25s} {'Acc':>7s} {'AUC':>7s} {'Prec':>7s} {'Rec':>7s} "
          f"{'F1':>7s} {'Train+':>7s} {'Test+':>7s}")
    print("-" * 80)
    for fname in FEATURE_NAMES:
        if fname not in BINARY_FEATURES:
            continue
        r = results[fname]["full"]
        if r is None:
            continue
        print(f"{fname:25s} {r['accuracy']:7.3f} {r['auc']:7.3f} "
              f"{r['precision']:7.3f} {r['recall']:7.3f} {r['f1']:7.3f} "
              f"{r['train_pos_rate']:7.1%} {r['test_pos_rate']:7.1%}")

    print()


# ---------------------------------------------------------------------------
# Save / load probe artifacts
# ---------------------------------------------------------------------------

def save_probes(results, pca, output_dir):
    """Save learned directions and PCA model for downstream use."""
    import joblib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save directions as a numpy archive
    directions = {}
    for fname in FEATURE_NAMES:
        r = results[fname]["full"]
        if r is not None:
            directions[f"{fname}_coef"] = r["coef"]
            directions[f"{fname}_intercept"] = np.array([r["intercept"]])
            directions[f"{fname}_scaler_mean"] = r["scaler_mean"]
            directions[f"{fname}_scaler_scale"] = r["scaler_scale"]
    np.savez(output_dir / "probe_directions.npz", **directions)
    print(f"Saved {output_dir / 'probe_directions.npz'}")

    joblib.dump(pca, output_dir / "probe_pca.pkl")
    print(f"Saved {output_dir / 'probe_pca.pkl'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train linear probes on activation vectors for behavioral metrics.")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation_data/ folders")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Keep every Nth step per trajectory (default: 10)")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of trajectories for test set (default: 0.2)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Regularization strength (Ridge alpha / 1/C for logistic)")
    parser.add_argument("--n-pca", type=int, default=50,
                        help="Number of PCA components for reduced baseline (default: 50)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <data_dir>/probe_results)")
    parser.add_argument("--class-weight", type=str, default="none",
                        choices=["none", "balanced"],
                        help="Class reweighting strategy (default: none)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.data_dir) / "probe_results")

    # Load data
    print("Loading activation data...")
    records, total = load_data(args.data_dir, subsample_rate=args.subsample)
    print(f"Total raw records: {total}, after subsampling: {len(records)}")

    # Compute features
    print("Computing features...")
    activations, features, metadata = compute_features(records)
    print(f"Activations shape: {activations.shape}")
    print(f"Features shape:    {features.shape}")

    # Train probes
    print("\nTraining linear probes...")
    class_weight = None if args.class_weight == "none" else args.class_weight

    results, pca = train_all_probes(
        activations, features, metadata,
        n_pca=args.n_pca, alpha=args.alpha,
        test_frac=args.test_frac, class_weight=class_weight,
        seed=args.seed,
    )

    # Report
    print_summary(results)

    # Save
    save_probes(results, pca, output_dir)

    # Visualize
    plot_results(results, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
