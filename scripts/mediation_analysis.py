"""Causal mediation analysis for linear probe results.

Tests whether high-R² probe results reflect genuine representations or
confounded correlations by residualizing activations and targets on
potential mediator features.

Two analyses:
1. n_visible_players mediation: control for combat, position, time signals
2. Tick control for all features: how much R² is explained by game time
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from analyze_activations import FEATURE_NAMES, compute_features, load_data
from train_probes import trajectory_split, BINARY_FEATURES


# ---------------------------------------------------------------------------
# Mediator sets for n_visible_players analysis
# ---------------------------------------------------------------------------

MEDIATOR_SETS = {
    "Combat":   ["in_combat", "self_damage", "max_combat_level", "is_attacking"],
    "Position": ["self_row", "self_col"],
    "Time":     ["tick", "time_alive"],
}

# Cumulative: each includes all previous
CUMULATIVE_MEDIATOR_SETS = {
    "Combat":              MEDIATOR_SETS["Combat"],
    "Combat+Position":     MEDIATOR_SETS["Combat"] + MEDIATOR_SETS["Position"],
    "Combat+Position+Time": MEDIATOR_SETS["Combat"] + MEDIATOR_SETS["Position"] + MEDIATOR_SETS["Time"],
}


def get_feature_indices(names):
    """Get column indices in the features array for given feature names."""
    return [FEATURE_NAMES.index(n) for n in names]


def residualize(X, Z, X_test=None, Z_test=None, alpha=1.0):
    """Regress X on Z and return residuals.

    X: (n, d) array to residualize (can be activations or target)
    Z: (n, k) mediator features
    Returns residuals of X after removing linear prediction from Z.
    If X_test/Z_test provided, returns (train_resid, test_resid).
    """
    scaler = StandardScaler()
    Z_s = scaler.fit_transform(Z)

    if X.ndim == 1:
        model = Ridge(alpha=alpha)
        model.fit(Z_s, X)
        resid_train = X - model.predict(Z_s)
        if X_test is not None:
            Z_test_s = scaler.transform(Z_test)
            resid_test = X_test - model.predict(Z_test_s)
            return resid_train, resid_test
        return resid_train
    else:
        # For high-dim activations, residualize each dimension
        model = Ridge(alpha=alpha)
        model.fit(Z_s, X)
        resid_train = X - model.predict(Z_s)
        if X_test is not None:
            Z_test_s = scaler.transform(Z_test)
            resid_test = X_test - model.predict(Z_test_s)
            return resid_train, resid_test
        return resid_train


def train_ridge_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """Train Ridge probe and return R²."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    model = Ridge(alpha=alpha)
    model.fit(X_tr_s, y_train)
    y_pred = model.predict(X_te_s)
    return r2_score(y_test, y_pred)


def train_logistic_probe(X_train, y_train, X_test, y_test, alpha=1.0):
    """Train logistic probe and return AUC."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    model = LogisticRegression(C=1.0/max(alpha, 1e-8), max_iter=2000,
                               solver="lbfgs", random_state=42)
    model.fit(X_tr_s, y_train)
    y_prob = model.predict_proba(X_te_s)[:, 1]
    return roc_auc_score(y_test, y_prob)


# ---------------------------------------------------------------------------
# Analysis 1: n_visible_players mediation
# ---------------------------------------------------------------------------

def mediation_analysis_nvp(activations, features, metadata, alpha=1.0):
    """Run mediation analysis for n_visible_players."""
    train_idx, test_idx = trajectory_split(metadata)
    target_idx = FEATURE_NAMES.index("n_visible_players")

    y_train = features[train_idx, target_idx]
    y_test = features[test_idx, target_idx]
    X_train = activations[train_idx]
    X_test = activations[test_idx]

    # Uncontrolled R²
    r2_uncontrolled = train_ridge_probe(X_train, y_train, X_test, y_test, alpha)

    results = {}

    all_sets = {**MEDIATOR_SETS, **CUMULATIVE_MEDIATOR_SETS}
    # Order for display
    display_order = ["Combat", "Position", "Time",
                     "Combat+Position", "Combat+Position+Time"]

    for set_name in display_order:
        mediator_names = all_sets[set_name]
        med_indices = get_feature_indices(mediator_names)
        Z_train = features[train_idx][:, med_indices]
        Z_test = features[test_idx][:, med_indices]

        # Residualize activations on mediators
        X_tr_resid, X_te_resid = residualize(
            X_train, Z_train, X_test, Z_test, alpha=alpha)

        # Residualize target on mediators
        y_tr_resid, y_te_resid = residualize(
            y_train, Z_train, y_test, Z_test, alpha=alpha)

        # Train probe on residuals
        r2_controlled = train_ridge_probe(
            X_tr_resid, y_tr_resid, X_te_resid, y_te_resid, alpha)

        pct_explained = (1 - r2_controlled / r2_uncontrolled) * 100 if r2_uncontrolled > 0 else float('nan')

        results[set_name] = {
            "r2_uncontrolled": r2_uncontrolled,
            "r2_controlled": r2_controlled,
            "pct_explained": pct_explained,
            "mediators": mediator_names,
        }
        print(f"  {set_name:30s}  R²: {r2_uncontrolled:.4f} -> {r2_controlled:.4f}  "
              f"({pct_explained:.1f}% explained by mediators)")

    return r2_uncontrolled, results


# ---------------------------------------------------------------------------
# Analysis 2: Tick control for all features
# ---------------------------------------------------------------------------

def tick_control_analysis(activations, features, metadata, alpha=1.0):
    """For every feature, compare uncontrolled vs tick-controlled probe."""
    train_idx, test_idx = trajectory_split(metadata)
    tick_idx = FEATURE_NAMES.index("tick")

    Z_train = features[train_idx, tick_idx:tick_idx+1]
    Z_test = features[test_idx, tick_idx:tick_idx+1]

    # Residualize activations on tick (once, shared across all features)
    X_train = activations[train_idx]
    X_test = activations[test_idx]
    X_tr_resid, X_te_resid = residualize(X_train, Z_train, X_test, Z_test, alpha=alpha)

    results = {}

    for fi, fname in enumerate(FEATURE_NAMES):
        is_binary = fname in BINARY_FEATURES
        y_train = features[train_idx, fi]
        y_test = features[test_idx, fi]

        # Skip tick itself
        if fname == "tick":
            if is_binary:
                r2_unc = train_logistic_probe(X_train, y_train, X_test, y_test, alpha)
            else:
                r2_unc = train_ridge_probe(X_train, y_train, X_test, y_test, alpha)
            results[fname] = {
                "type": "bin" if is_binary else "cont",
                "uncontrolled": r2_unc,
                "controlled": None,
                "delta": None,
                "pct_via_tick": None,
            }
            print(f"  {fname:25s}  {'AUC' if is_binary else 'R²'}: {r2_unc:.4f}  (tick itself, skip)")
            continue

        if is_binary:
            # Uncontrolled AUC
            auc_unc = train_logistic_probe(X_train, y_train, X_test, y_test, alpha)
            if auc_unc is None:
                results[fname] = {
                    "type": "bin",
                    "uncontrolled": None,
                    "controlled": None,
                    "delta": None,
                    "pct_via_tick": None,
                }
                print(f"  {fname:25s}  SKIP (single class)")
                continue

            # Tick-controlled: residualize activations but keep binary target as-is
            # Use logistic regression on residualized activations
            auc_ctrl = train_logistic_probe(X_tr_resid, y_train, X_te_resid, y_test, alpha)

            delta = auc_unc - (auc_ctrl or 0)
            # For AUC, percentage relative to above-chance performance
            pct = (delta / (auc_unc - 0.5) * 100) if auc_unc > 0.5 else float('nan')

            results[fname] = {
                "type": "bin",
                "uncontrolled": auc_unc,
                "controlled": auc_ctrl,
                "delta": delta,
                "pct_via_tick": pct,
            }
            ctrl_str = f"{auc_ctrl:.4f}" if auc_ctrl is not None else "N/A"
            print(f"  {fname:25s}  AUC: {auc_unc:.4f} -> {ctrl_str:>6}  "
                  f"(delta={delta:.4f}, {pct:.1f}% via tick)")
        else:
            # Uncontrolled R²
            r2_unc = train_ridge_probe(X_train, y_train, X_test, y_test, alpha)

            # Residualize target on tick
            y_tr_resid, y_te_resid = residualize(
                y_train, Z_train, y_test, Z_test, alpha=alpha)

            # Tick-controlled R²
            r2_ctrl = train_ridge_probe(X_tr_resid, y_tr_resid, X_te_resid, y_te_resid, alpha)

            delta = r2_unc - r2_ctrl
            pct = (delta / r2_unc * 100) if r2_unc > 0 else float('nan')

            results[fname] = {
                "type": "cont",
                "uncontrolled": r2_unc,
                "controlled": r2_ctrl,
                "delta": delta,
                "pct_via_tick": pct,
            }
            print(f"  {fname:25s}  R²: {r2_unc:.4f} -> {r2_ctrl:.4f}  "
                  f"(delta={delta:.4f}, {pct:.1f}% via tick)")

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_table1(r2_uncontrolled, mediation_results):
    """Print n_visible_players mediation table."""
    print("\n" + "=" * 95)
    print("TABLE 1: n_visible_players MEDIATION ANALYSIS")
    print("=" * 95)
    print(f"\n{'Mediator Set':30s} | {'R² (uncontrolled)':>18s} | {'R² (residualized)':>18s} | {'% explained':>12s}")
    print("-" * 95)
    for name, res in mediation_results.items():
        print(f"{name:30s} | {res['r2_uncontrolled']:18.4f} | {res['r2_controlled']:18.4f} | {res['pct_explained']:11.1f}%")
    print()


def print_table2(tick_results):
    """Print tick control table for all features."""
    print("\n" + "=" * 100)
    print("TABLE 2: TICK CONTROL FOR ALL FEATURES")
    print("=" * 100)
    print(f"\n{'Feature':25s} | {'Type':4s} | {'Uncontrolled':>12s} | {'Tick-controlled':>15s} | {'Delta':>8s} | {'% via tick':>10s}")
    print("-" * 100)

    for fname in FEATURE_NAMES:
        res = tick_results[fname]
        ftype = res["type"]
        metric = "AUC" if ftype == "bin" else "R²"

        if res["uncontrolled"] is None:
            print(f"{fname:25s} | {ftype:4s} | {'SKIP':>12s} |")
            continue

        unc_str = f"{res['uncontrolled']:.4f}"

        if res["controlled"] is None:
            print(f"{fname:25s} | {ftype:4s} | {unc_str:>12s} | {'N/A':>15s} | {'N/A':>8s} | {'N/A':>10s}")
        else:
            ctrl_str = f"{res['controlled']:.4f}"
            delta_str = f"{res['delta']:.4f}"
            pct_str = f"{res['pct_via_tick']:.1f}%"
            print(f"{fname:25s} | {ftype:4s} | {unc_str:>12s} | {ctrl_str:>15s} | {delta_str:>8s} | {pct_str:>10s}")

    print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(r2_uncontrolled, mediation_results, tick_results, output_dir):
    """Save analysis results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mediation results
    med_out = {
        "r2_uncontrolled": r2_uncontrolled,
        "mediator_sets": {}
    }
    for name, res in mediation_results.items():
        med_out["mediator_sets"][name] = {
            "r2_controlled": res["r2_controlled"],
            "pct_explained": res["pct_explained"],
            "mediators": res["mediators"],
        }
    with open(output_dir / "nvp_mediation.json", "w") as f:
        json.dump(med_out, f, indent=2)
    print(f"Saved {output_dir / 'nvp_mediation.json'}")

    # Tick control results
    tick_out = {}
    for fname, res in tick_results.items():
        tick_out[fname] = {
            "type": res["type"],
            "uncontrolled": res["uncontrolled"],
            "controlled": res["controlled"],
            "delta": res["delta"],
            "pct_via_tick": res["pct_via_tick"],
        }
    with open(output_dir / "tick_control.json", "w") as f:
        json.dump(tick_out, f, indent=2)
    print(f"Saved {output_dir / 'tick_control.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Causal mediation analysis for linear probe results.")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation data")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Keep every Nth step per trajectory (default: 10)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge alpha (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default="results/mediation_analysis",
                        help="Output directory")
    args = parser.parse_args()

    # Load data
    print("Loading activation data...")
    records, total = load_data(args.data_dir, subsample_rate=args.subsample)
    print(f"Total raw records: {total}, after subsampling: {len(records)}")

    print("Computing features...")
    activations, features, metadata = compute_features(records)
    print(f"Activations shape: {activations.shape}")
    print(f"Features shape:    {features.shape}")

    # Analysis 1: n_visible_players mediation
    print("\n" + "=" * 60)
    print("ANALYSIS 1: n_visible_players mediation")
    print("=" * 60)
    r2_unc, med_results = mediation_analysis_nvp(
        activations, features, metadata, alpha=args.alpha)

    # Analysis 2: Tick control
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Tick control for all features")
    print("=" * 60)
    tick_results = tick_control_analysis(
        activations, features, metadata, alpha=args.alpha)

    # Print summary tables
    print_table1(r2_unc, med_results)
    print_table2(tick_results)

    # Save
    save_results(r2_unc, med_results, tick_results, args.output_dir)
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
