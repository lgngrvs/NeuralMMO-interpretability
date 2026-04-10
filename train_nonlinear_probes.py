"""Train nonlinear (2-layer ReLU MLP) probes on activation vectors for behavioral metrics.

Same evaluation protocol as train_probes.py (trajectory split, three conditions:
full / PCA / shuffle), but uses a PyTorch MLP instead of sklearn linear models.

Architecture: Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from analyze_activations import FEATURE_NAMES, compute_features, load_data
from train_probes import (
    BINARY_FEATURES,
    _compute_sample_weights_balanced,
    plot_results,
    print_summary,
    trajectory_split,
)


# ---------------------------------------------------------------------------
# MLP probe model
# ---------------------------------------------------------------------------

class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training a single nonlinear probe
# ---------------------------------------------------------------------------

def train_probe(X_train, y_train, X_test, y_test, binary=False,
                class_weight=None, seed=42, lr=1e-3, epochs=100,
                batch_size=512, hidden_factor=4):
    """Train a 2-layer MLP probe and return metrics dict.

    Returns dict with same keys as train_probes.train_probe so that
    print_summary / plot_results remain compatible.
    """
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

    # Single-class guard for binary
    if binary and (len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2):
        return None

    # Standardize inputs
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    y_train_f = y_train.astype(np.float32)

    # Build model
    torch.manual_seed(seed)
    input_dim = X_train_s.shape[1]
    hidden_dim = max(1, input_dim // hidden_factor)
    model = MLPProbe(input_dim, hidden_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and sample weights
    if binary:
        if class_weight == "balanced":
            n_pos = y_train_f.sum()
            n_neg = len(y_train_f) - n_pos
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        else:
            pos_weight = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        sample_weights = None
    else:
        criterion = nn.MSELoss(reduction="none")
        if class_weight == "balanced":
            sample_weights = _compute_sample_weights_balanced(y_train).astype(np.float32)
        else:
            sample_weights = None

    # DataLoader
    X_t = torch.from_numpy(X_train_s)
    y_t = torch.from_numpy(y_train_f)
    if sample_weights is not None:
        w_t = torch.from_numpy(sample_weights)
        dataset = TensorDataset(X_t, y_t, w_t)
    else:
        dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for _epoch in range(epochs):
        for batch in loader:
            if sample_weights is not None:
                xb, yb, wb = [b.to(device) for b in batch]
            else:
                xb, yb = [b.to(device) for b in batch]
                wb = None

            logits = model(xb)
            if binary:
                loss = criterion(logits, yb)
            else:
                per_sample_loss = criterion(logits, yb)
                if wb is not None:
                    loss = (per_sample_loss * wb).mean()
                else:
                    loss = per_sample_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test_s).to(device)
        logits_test = model(X_test_t).cpu().numpy()

    result = {}

    if binary:
        y_prob = 1.0 / (1.0 + np.exp(-logits_test))  # sigmoid
        y_pred = (y_prob >= 0.5).astype(int)
        result["accuracy"] = accuracy_score(y_test, y_pred)
        result["auc"] = roc_auc_score(y_test, y_prob)
        result["precision"] = precision_score(y_test, y_pred, zero_division=0)
        result["recall"] = recall_score(y_test, y_pred, zero_division=0)
        result["f1"] = f1_score(y_test, y_pred, zero_division=0)
        result["train_pos_rate"] = y_train.mean()
        result["test_pos_rate"] = y_test.mean()
    else:
        y_pred = logits_test
        result["r2"] = r2_score(y_test, y_pred)
        result["mae"] = mean_absolute_error(y_test, y_pred)
        result["y_std"] = float(y_test.std())

    # Store model state_dict and scaler info for saving
    result["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    result["input_dim"] = input_dim
    result["hidden_dim"] = hidden_dim
    result["scaler_mean"] = scaler.mean_
    result["scaler_scale"] = scaler.scale_

    # Placeholder coef for compatibility with plot_results direction similarity
    # Use first layer weight's mean direction as a proxy
    w1 = model.net[0].weight.detach().cpu().numpy()  # (hidden, input)
    result["coef"] = w1.mean(axis=0)
    result["intercept"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Train all probes (full / PCA / shuffle)
# ---------------------------------------------------------------------------

def train_all_probes(activations, features, metadata, n_pca=50,
                     test_frac=0.2, class_weight=None, seed=42,
                     lr=1e-3, epochs=100, batch_size=512, hidden_factor=4):
    """Train MLP probes for all features under three conditions."""
    from sklearn.decomposition import PCA

    train_idx, test_idx = trajectory_split(metadata, test_frac=test_frac, seed=seed)
    print(f"Split: {len(train_idx)} train, {len(test_idx)} test "
          f"({len(test_idx) / (len(train_idx) + len(test_idx)) * 100:.1f}% test)")

    pca = PCA(n_components=min(n_pca, activations.shape[1]), random_state=seed)
    act_pca = pca.fit_transform(activations)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {pca.n_components_} components, {var_explained * 100:.1f}% variance explained")

    results = {}
    rng = np.random.RandomState(seed)

    common_kwargs = dict(lr=lr, epochs=epochs, batch_size=batch_size,
                         hidden_factor=hidden_factor, class_weight=class_weight,
                         seed=seed)

    for fi, fname in enumerate(FEATURE_NAMES):
        binary = fname in BINARY_FEATURES
        y = features[:, fi]
        y_train, y_test = y[train_idx], y[test_idx]

        conditions = {
            "full": (activations[train_idx], activations[test_idx], y_train, y_test),
            "pca": (act_pca[train_idx], act_pca[test_idx], y_train, y_test),
        }

        y_train_shuf = y_train.copy()
        rng.shuffle(y_train_shuf)
        conditions["shuffle"] = (
            activations[train_idx], activations[test_idx], y_train_shuf, y_test
        )

        results[fname] = {}
        for cond_name, (Xtr, Xte, ytr, yte) in conditions.items():
            res = train_probe(Xtr, ytr, Xte, yte, binary=binary, **common_kwargs)
            results[fname][cond_name] = res

        if binary:
            full = results[fname]["full"]
            tag = f"AUC={full['auc']:.3f}" if full else "SKIP"
        else:
            full = results[fname]["full"]
            tag = f"R²={full['r2']:.3f}" if full else "SKIP"
        print(f"  {fname:25s}  {tag}")

    return results, pca


# ---------------------------------------------------------------------------
# Save MLP probe artifacts
# ---------------------------------------------------------------------------

def save_probes(results, pca, output_dir):
    """Save MLP state dicts and PCA model."""
    import joblib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all state_dicts into a single dict
    model_data = {}
    for fname in FEATURE_NAMES:
        r = results[fname]["full"]
        if r is not None:
            model_data[fname] = {
                "state_dict": r["state_dict"],
                "input_dim": r["input_dim"],
                "hidden_dim": r["hidden_dim"],
                "scaler_mean": r["scaler_mean"],
                "scaler_scale": r["scaler_scale"],
            }

    torch.save(model_data, output_dir / "mlp_probe_weights.pt")
    print(f"Saved {output_dir / 'mlp_probe_weights.pt'}")

    joblib.dump(pca, output_dir / "mlp_probe_pca.pkl")
    print(f"Saved {output_dir / 'mlp_probe_pca.pkl'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train nonlinear (MLP) probes on activation vectors for behavioral metrics.")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation_data/ folders")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Keep every Nth step per trajectory (default: 10)")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="Fraction of trajectories for test set (default: 0.2)")
    parser.add_argument("--n-pca", type=int, default=50,
                        help="Number of PCA components for reduced baseline (default: 50)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <data_dir>/nonlinear_probe_results)")
    parser.add_argument("--class-weight", type=str, default="none",
                        choices=["none", "balanced"],
                        help="Class reweighting strategy (default: none)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size (default: 512)")
    parser.add_argument("--hidden-factor", type=int, default=4,
                        help="hidden_dim = input_dim // hidden_factor (default: 4)")
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.data_dir) / "nonlinear_probe_results")

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
    print("\nTraining nonlinear (MLP) probes...")
    class_weight = None if args.class_weight == "none" else args.class_weight

    results, pca = train_all_probes(
        activations, features, metadata,
        n_pca=args.n_pca,
        test_frac=args.test_frac, class_weight=class_weight,
        seed=args.seed,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        hidden_factor=args.hidden_factor,
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
