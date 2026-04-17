"""Subspace activation patching: causal intervention along the tick probe direction.

Instead of swapping entire activation vectors, we decompose activations into
a tick-relevant component (along the learned probe direction) and an orthogonal
complement, then swap only the tick component. This isolates the causal
contribution of the tick representation to action selection.

Method: interchange intervention (Geiger et al., 2021)
  a_patched = a_perp_target + a_parallel_source
            = a_target + ((a_source . d) - (a_target . d)) * d

Controls:
  1. Random direction baseline (same swap but along random unit vectors)
  2. Orthogonal complement swap (swap everything EXCEPT the tick direction)
  3. Full activation swap (for comparison)
  4. Dose-response interpolation (t=0 to t=1.5)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "activation_data_10x/baseline_10M/activations.cache.npz")
MODEL_PATH = os.path.join(BASE_DIR, "policies/baseline_10M.pt")
PROBE_PATH = os.path.join(BASE_DIR, "results/linear_10x/probe_directions.npz")
RESULTS_DIR = os.path.join(BASE_DIR, "results/activation_patching")

TICK_IDX = 30
ROW_IDX = 26
COL_IDX = 27

EARLY_TICK_MAX = 10
MID_TICK_LO = 400
MID_TICK_HI = 600
LATE_TICK_MIN = 800  # lowered from 900 for more samples

MOVE_DIRS = ["North", "South", "East", "West", "Stay"]
ATTACK_STYLES = ["Melee", "Range", "Mage"]

# All embedding-free action heads
EMBEDDING_FREE_HEADS = {
    "move": (5, MOVE_DIRS),
    "attack_style": (3, ATTACK_STYLES),
    "gold_quantity": (99, None),
    "inventory_price": (99, None),
}

N_RANDOM_DIRS = 50
N_BOOTSTRAP = 1000
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_inner_policy(model):
    policy = model
    while hasattr(policy, "policy"):
        policy = policy.policy
    return policy


def kl_divergence(p_logits, q_logits):
    """KL(P || Q) from logits. Returns per-row KL."""
    p = F.softmax(p_logits, dim=-1)
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def get_probe_direction(probe_path, feature_name="tick"):
    """Load probe direction and transform back to original activation space."""
    probe = np.load(probe_path)
    coef = probe[f"{feature_name}_coef"]          # in scaled space
    scaler_scale = probe[f"{feature_name}_scaler_scale"]

    # Transform from scaled space to original space
    d = coef / scaler_scale
    d = d / np.linalg.norm(d)
    return d.astype(np.float32)


def project_along(activations, direction):
    """Project activations onto direction. Returns scalar projections."""
    return activations @ direction  # (N,)


def swap_component(target_act, source_act, direction):
    """Interchange intervention: replace target's component along direction with source's."""
    d = torch.from_numpy(direction).unsqueeze(0)  # (1, 256)
    target_proj = (target_act * d).sum(dim=-1, keepdim=True)  # (N, 1)
    source_proj = (source_act * d).sum(dim=-1, keepdim=True)  # (M, 1)
    return target_act + (source_proj - target_proj) * d


def swap_orthogonal_complement(target_act, source_act, direction):
    """Swap everything EXCEPT the component along direction."""
    d = torch.from_numpy(direction).unsqueeze(0)
    target_proj = (target_act * d).sum(dim=-1, keepdim=True)
    source_proj = (source_act * d).sum(dim=-1, keepdim=True)
    # Keep target's tick component, replace everything else with source
    return source_act + (target_proj - source_proj) * d


def interpolated_patch(target_act, source_act, direction, t):
    """Dose-response: interpolate the tick component by factor t (0=no change, 1=full swap)."""
    d = torch.from_numpy(direction).unsqueeze(0)
    target_proj = (target_act * d).sum(dim=-1, keepdim=True)
    source_proj = (source_act * d).sum(dim=-1, keepdim=True)
    return target_act + t * (source_proj - target_proj) * d


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    means = np.sort([rng.choice(values, size=len(values), replace=True).mean()
                     for _ in range(n_boot)])
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return values.mean(), lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)

    # ==== 1. Load data, model, probe direction ====
    print("=" * 70)
    print("1. LOADING DATA, MODEL, AND PROBE DIRECTION")
    print("=" * 70)

    cache = np.load(CACHE_PATH)
    activations = cache["activations"]
    features = cache["features"]
    print(f"  Activations: {activations.shape}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    inner = get_inner_policy(checkpoint)
    decoder = inner.action_decoder

    # Load all embedding-free heads
    heads = {}
    for name in EMBEDDING_FREE_HEADS:
        layer = decoder.layers[name]
        layer.eval()
        heads[name] = layer
        print(f"  Head '{name}': {layer}")

    tick_dir = get_probe_direction(PROBE_PATH, "tick")
    print(f"  Tick probe direction: norm={np.linalg.norm(tick_dir):.4f}")

    # ==== 2. Select tick groups ====
    print("\n" + "=" * 70)
    print("2. SELECTING TICK GROUPS")
    print("=" * 70)

    ticks = features[:, TICK_IDX]

    groups = {
        "early": ticks <= EARLY_TICK_MAX,
        "mid": (ticks >= MID_TICK_LO) & (ticks <= MID_TICK_HI),
        "late": ticks >= LATE_TICK_MIN,
    }

    for name, mask in groups.items():
        idx = np.where(mask)[0]
        t = ticks[mask]
        print(f"  {name:5s}: n={len(idx):6d}, tick=[{t.min():.0f}, {t.max():.0f}], "
              f"mean_row={features[mask, ROW_IDX].mean():.1f}, "
              f"mean_col={features[mask, COL_IDX].mean():.1f}")

    early_idx = np.where(groups["early"])[0]
    mid_idx = np.where(groups["mid"])[0]
    late_idx = np.where(groups["late"])[0]

    # ==== 3. Compute original logits for all groups ====
    print("\n" + "=" * 70)
    print("3. ORIGINAL ACTION PROBABILITIES")
    print("=" * 70)

    with torch.no_grad():
        early_act = torch.tensor(activations[early_idx], dtype=torch.float32)
        mid_act = torch.tensor(activations[mid_idx], dtype=torch.float32)
        late_act = torch.tensor(activations[late_idx], dtype=torch.float32)

    orig_probs = {}
    orig_logits = {}
    for head_name, layer in heads.items():
        with torch.no_grad():
            orig_logits[(head_name, "early")] = layer(early_act)
            orig_logits[(head_name, "mid")] = layer(mid_act)
            orig_logits[(head_name, "late")] = layer(late_act)
            for grp in ["early", "mid", "late"]:
                orig_probs[(head_name, grp)] = F.softmax(orig_logits[(head_name, grp)], dim=-1).numpy()

    # Print move and attack_style
    for head_name in ["move", "attack_style"]:
        n_acts, labels = EMBEDDING_FREE_HEADS[head_name]
        print(f"\n  {head_name} probabilities (mean):")
        print(f"    {'Action':<10} {'Early':>8} {'Mid':>8} {'Late':>8}")
        for i, label in enumerate(labels):
            e = orig_probs[(head_name, "early")][:, i].mean()
            m = orig_probs[(head_name, "mid")][:, i].mean()
            l = orig_probs[(head_name, "late")][:, i].mean()
            print(f"    {label:<10} {e:>8.4f} {m:>8.4f} {l:>8.4f}")

    # ==== 4. Tick-direction projection statistics ====
    print("\n" + "=" * 70)
    print("4. TICK PROJECTION STATISTICS")
    print("=" * 70)

    for grp_name, idx in [("early", early_idx), ("mid", mid_idx), ("late", late_idx)]:
        proj = activations[idx] @ tick_dir
        print(f"  {grp_name:5s}: proj mean={proj.mean():.3f}, std={proj.std():.3f}, "
              f"range=[{proj.min():.3f}, {proj.max():.3f}]")

    # ==== 5. Subspace patching: tick direction swap ====
    print("\n" + "=" * 70)
    print("5. TICK-DIRECTION PATCHING (into early-tick base)")
    print("=" * 70)

    # Pair source activations with early targets
    mid_paired = rng.choice(len(mid_idx), size=len(early_idx), replace=True)
    late_paired = rng.choice(len(late_idx), size=len(early_idx), replace=True)

    with torch.no_grad():
        # Tick-direction swap: replace early's tick component with mid/late's
        patched_mid_tick = swap_component(early_act, mid_act[mid_paired], tick_dir)
        patched_late_tick = swap_component(early_act, late_act[late_paired], tick_dir)

        # Full activation swap (for comparison)
        patched_mid_full = mid_act[mid_paired]
        patched_late_full = late_act[late_paired]

        # Orthogonal complement swap: swap everything EXCEPT tick direction
        patched_mid_orth = swap_orthogonal_complement(early_act, mid_act[mid_paired], tick_dir)
        patched_late_orth = swap_orthogonal_complement(early_act, late_act[late_paired], tick_dir)

    # Compute KL divergences for each condition and head
    conditions = {
        "tick_dir_mid→early": patched_mid_tick,
        "tick_dir_late→early": patched_late_tick,
        "full_swap_mid→early": patched_mid_full,
        "full_swap_late→early": patched_late_full,
        "orth_comp_mid→early": patched_mid_orth,
        "orth_comp_late→early": patched_late_orth,
    }

    kl_results = {}
    print(f"\n  {'Condition':<28s} {'move':>8s} {'atk_style':>10s} {'gold_qty':>10s} {'inv_price':>10s}")
    print("  " + "-" * 70)
    for cond_name, patched_act in conditions.items():
        kls = {}
        for head_name, layer in heads.items():
            with torch.no_grad():
                patched_logits = layer(patched_act)
                kl = kl_divergence(orig_logits[(head_name, "early")], patched_logits).numpy()
                kls[head_name] = kl
        kl_results[cond_name] = kls
        print(f"  {cond_name:<28s} "
              f"{kls['move'].mean():>8.4f} "
              f"{kls['attack_style'].mean():>10.4f} "
              f"{kls['gold_quantity'].mean():>10.4f} "
              f"{kls['inventory_price'].mean():>10.4f}")

    # ==== 6. Random direction baseline ====
    print("\n" + "=" * 70)
    print("6. RANDOM DIRECTION BASELINE")
    print("=" * 70)

    random_kls = {head_name: [] for head_name in heads}
    for i in range(N_RANDOM_DIRS):
        rand_dir = rng.randn(256).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir)

        with torch.no_grad():
            patched_rand = swap_component(early_act, late_act[late_paired], rand_dir)
            for head_name, layer in heads.items():
                patched_logits = layer(patched_rand)
                kl = kl_divergence(orig_logits[(head_name, "early")], patched_logits).numpy().mean()
                random_kls[head_name].append(kl)

    print(f"  Random direction KL (mean +/- std over {N_RANDOM_DIRS} directions):")
    print(f"  {'Head':<18s} {'Random':>12s} {'Tick dir':>12s} {'Ratio':>8s}")
    print("  " + "-" * 54)
    for head_name in heads:
        rand_mean = np.mean(random_kls[head_name])
        rand_std = np.std(random_kls[head_name])
        tick_kl = kl_results["tick_dir_late→early"][head_name].mean()
        ratio = tick_kl / max(rand_mean, 1e-8)
        print(f"  {head_name:<18s} {rand_mean:>8.4f}±{rand_std:.4f} {tick_kl:>12.4f} {ratio:>8.1f}x")

    # ==== 7. Dose-response curve ====
    print("\n" + "=" * 70)
    print("7. DOSE-RESPONSE CURVE")
    print("=" * 70)

    t_values = np.linspace(0, 1.5, 16)
    dose_kls = {head_name: [] for head_name in heads}
    dose_move_probs = []

    for t in t_values:
        with torch.no_grad():
            patched_t = interpolated_patch(early_act, late_act[late_paired], tick_dir, t)
            for head_name, layer in heads.items():
                patched_logits = layer(patched_t)
                kl = kl_divergence(orig_logits[(head_name, "early")], patched_logits).numpy().mean()
                dose_kls[head_name].append(kl)
                if head_name == "move":
                    dose_move_probs.append(F.softmax(patched_logits, dim=-1).numpy().mean(axis=0))

    dose_move_probs = np.array(dose_move_probs)  # (n_t, 5)

    print(f"  {'t':>5s} {'move_KL':>10s} {'atk_KL':>10s} ", end="")
    print(" ".join(f"{d:>7s}" for d in MOVE_DIRS))
    for i, t in enumerate(t_values):
        print(f"  {t:>5.2f} {dose_kls['move'][i]:>10.4f} {dose_kls['attack_style'][i]:>10.4f} ", end="")
        print(" ".join(f"{dose_move_probs[i, j]:>7.4f}" for j in range(5)))

    # ==== 8. Probability shifts from tick-direction patching ====
    print("\n" + "=" * 70)
    print("8. PROBABILITY SHIFTS (tick-direction patching into early)")
    print("=" * 70)

    for source_name, cond_name in [("mid", "tick_dir_mid→early"), ("late", "tick_dir_late→early")]:
        print(f"\n  {source_name}→early tick-direction swap:")
        for head_name in ["move", "attack_style"]:
            n_acts, labels = EMBEDDING_FREE_HEADS[head_name]
            if labels is None:
                continue
            with torch.no_grad():
                patched_act = conditions[cond_name]
                patched_probs = F.softmax(heads[head_name](patched_act), dim=-1).numpy()
            orig = orig_probs[(head_name, "early")]
            print(f"    {head_name}:")
            print(f"      {'Action':<10} {'Original':>10} {'Patched':>10} {'Delta':>10}")
            for i, label in enumerate(labels):
                o = orig[:, i].mean()
                p = patched_probs[:, i].mean()
                print(f"      {label:<10} {o:>10.4f} {p:>10.4f} {p-o:>+10.4f}")

    # ==== 9. Bootstrap CIs ====
    print("\n" + "=" * 70)
    print("9. BOOTSTRAP 95% CIs (move KL)")
    print("=" * 70)

    for cond_name in ["tick_dir_mid→early", "tick_dir_late→early",
                       "full_swap_mid→early", "full_swap_late→early",
                       "orth_comp_mid→early", "orth_comp_late→early"]:
        vals = kl_results[cond_name]["move"]
        mean, lo, hi = bootstrap_ci(vals)
        print(f"  {cond_name:<28s}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # ==== 10. Plots ====
    print("\n" + "=" * 70)
    print("10. SAVING PLOTS")
    print("=" * 70)

    # --- Plot 1: KL comparison across conditions ---
    fig, ax = plt.subplots(figsize=(12, 5))
    cond_names = ["tick_dir_late→early", "full_swap_late→early", "orth_comp_late→early"]
    cond_labels = ["Tick direction\nonly", "Full activation\nswap", "Orthogonal\ncomplement"]
    x = np.arange(len(cond_names))
    width = 0.18
    for i, (head_name, _) in enumerate(EMBEDDING_FREE_HEADS.items()):
        vals = [kl_results[c][head_name].mean() for c in cond_names]
        ax.bar(x + i * width, vals, width, label=head_name)
    # Add random baseline as horizontal line
    for i, (head_name, _) in enumerate(EMBEDDING_FREE_HEADS.items()):
        ax.axhline(np.mean(random_kls[head_name]), color=f"C{i}", ls="--", alpha=0.5, lw=1)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Mean KL Divergence")
    ax.set_title("Subspace Patching: KL(original || patched) by Condition\n(dashed = random direction baseline)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "subspace_kl_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Plot 2: Dose-response curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t_values, dose_kls["move"], "o-", label="move", markersize=4)
    ax1.plot(t_values, dose_kls["attack_style"], "s-", label="attack_style", markersize=4)
    ax1.axhline(np.mean(random_kls["move"]), color="C0", ls="--", alpha=0.5, label="random baseline (move)")
    ax1.axvline(1.0, color="gray", ls=":", alpha=0.5)
    ax1.set_xlabel("Interpolation factor t (0=no change, 1=full swap)")
    ax1.set_ylabel("Mean KL Divergence")
    ax1.set_title("Dose-Response: KL vs Intervention Strength")
    ax1.legend(fontsize=8)

    for i, d in enumerate(MOVE_DIRS):
        ax2.plot(t_values, dose_move_probs[:, i], "o-", label=d, markersize=4)
    ax2.axvline(1.0, color="gray", ls=":", alpha=0.5)
    ax2.set_xlabel("Interpolation factor t")
    ax2.set_ylabel("Mean probability")
    ax2.set_title("Move Direction Probabilities vs Intervention Strength")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "dose_response.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Plot 3: Move probabilities comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(MOVE_DIRS))
    w = 0.15

    for ax_idx, (source, cond) in enumerate([("mid", "tick_dir_mid→early"),
                                               ("late", "tick_dir_late→early")]):
        ax = axes[ax_idx]
        with torch.no_grad():
            patched_probs = F.softmax(heads["move"](conditions[cond]), dim=-1).numpy()

        ax.bar(x - 1.5*w, orig_probs[("move", "early")].mean(axis=0), w,
               label="Early original", color="#4C72B0")
        ax.bar(x - 0.5*w, orig_probs[("move", source)].mean(axis=0), w,
               label=f"{source.title()} original", color="#DD8452")
        ax.bar(x + 0.5*w, patched_probs.mean(axis=0), w,
               label=f"Tick-dir {source}→early", color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(MOVE_DIRS)
        ax.set_ylabel("Mean probability")
        ax.set_title(f"Move Probs: {source}→early tick-direction patch")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "move_probs_subspace.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
