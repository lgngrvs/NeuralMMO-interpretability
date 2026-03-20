"""Analyze whether cluster assignments follow a life-phase pattern across all agents.

For each agent trajectory, splits its lifetime into bins (e.g. 10 bins from
birth to death) and computes the fraction of timesteps in each cluster per bin.
If all agents follow C0→C1 (or vice versa), we'll see a clear phase transition
in the aggregate plot.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import orjson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Entity observation column indices
ENT_ID = 0
ENT_HEALTH = 12


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cluster assignments vs agent life phase")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--cluster-dir", "--cluster_dir", type=str, required=True)
    parser.add_argument("--n-bins", type=int, default=20,
                        help="Number of life-phase bins (default: 20)")
    parser.add_argument("--min-lifespan", type=int, default=50,
                        help="Minimum alive ticks to include an agent (default: 50)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cluster_dir = Path(args.cluster_dir)

    # Load models
    pca_path = cluster_dir / "pca_model.pkl"
    cluster_before_umap = pca_path.exists()
    if cluster_before_umap:
        print("Loading PCA + HDBSCAN models (cluster-before-umap mode)...", flush=True)
        pca = joblib.load(pca_path)
    else:
        print("Loading UMAP + HDBSCAN models...", flush=True)
        umap_reducer = joblib.load(cluster_dir / "umap_model.pkl")
    hdbscan_clusterer = joblib.load(cluster_dir / "hdbscan_model.pkl")
    from hdbscan import approximate_predict

    # Load data
    json_files = list(data_dir.rglob("activations.json"))
    if not json_files:
        print(f"No activations.json found under {data_dir}")
        sys.exit(1)

    json_path = json_files[0]
    file_size_mb = json_path.stat().st_size / 1024 / 1024
    print(f"Loading {json_path.name} ({file_size_mb:.0f} MB)...", flush=True)
    with open(json_path, "rb") as f:
        all_records = orjson.loads(f.read())
    print(f"  {len(all_records)} records loaded")

    # Group by trajectory, filter agent_id=0
    print("Grouping trajectories...", flush=True)
    trajectories = defaultdict(list)
    for r in all_records:
        if r["agent_id"] == 0:
            continue
        trajectories[(r["env_id"], r["agent_id"])].append(r)
    del all_records

    print(f"  {len(trajectories)} trajectories")

    # For each trajectory: filter alive, split into longest episode,
    # predict clusters, bin by life phase
    n_bins = args.n_bins
    unique_clusters = sorted(set(hdbscan_clusterer.labels_) - {-1})
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

    # Accumulate: for each bin, count of each cluster across all agents
    bin_cluster_counts = np.zeros((n_bins, n_clusters + 1), dtype=np.float64)  # +1 for noise
    n_agents_used = 0

    for (env_id, agent_id), records in trajectories.items():
        records.sort(key=lambda r: r["step"])

        # Filter alive ticks
        alive_records = []
        for r in records:
            entity = r["observation"]["Entity"]
            alive = False
            for row in entity:
                if row[ENT_ID] == agent_id and row[ENT_HEALTH] > 0:
                    alive = True
                    break
            if alive:
                alive_records.append(r)

        if len(alive_records) < args.min_lifespan:
            continue

        # Split into episodes, keep longest
        steps = np.array([r["step"] for r in alive_records])
        boundaries = np.where(np.diff(steps) > 1)[0] + 1
        ep_starts = [0] + boundaries.tolist()
        ep_ends = boundaries.tolist() + [len(alive_records)]
        ep_lengths = [e - s for s, e in zip(ep_starts, ep_ends)]
        longest = max(range(len(ep_lengths)), key=lambda i: ep_lengths[i])
        s, e = ep_starts[longest], ep_ends[longest]
        episode = alive_records[s:e]

        if len(episode) < args.min_lifespan:
            continue

        # Predict clusters
        activations = np.array([r["activation"] for r in episode], dtype=np.float32)
        if cluster_before_umap:
            cluster_input = pca.transform(activations)
        else:
            cluster_input = umap_reducer.transform(activations)
        labels, _ = approximate_predict(hdbscan_clusterer, cluster_input)

        # Bin by life phase (0 = birth, n_bins-1 = death)
        life_frac = np.linspace(0, 1, len(episode), endpoint=False)
        bin_indices = np.clip((life_frac * n_bins).astype(int), 0, n_bins - 1)

        for tick_idx, (b, lab) in enumerate(zip(bin_indices, labels)):
            if lab == -1:
                bin_cluster_counts[b, -1] += 1  # noise
            else:
                bin_cluster_counts[b, cluster_to_idx[lab]] += 1

        n_agents_used += 1

    print(f"  {n_agents_used} agents with lifespan >= {args.min_lifespan}")

    if n_agents_used == 0:
        print("No agents met the minimum lifespan threshold!")
        sys.exit(1)

    # Normalize each bin to fractions
    bin_totals = bin_cluster_counts.sum(axis=1, keepdims=True)
    bin_fracs = np.where(bin_totals > 0, bin_cluster_counts / bin_totals, 0)

    # --- Plot: stacked area chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.linspace(0, 100, n_bins)  # life phase as percentage

    cmap = plt.colormaps.get_cmap("tab20").resampled(max(n_clusters + 1, 1))
    labels_legend = [f"C{c}" for c in unique_clusters] + ["noise"]
    colors = [cmap(i) for i in range(n_clusters)] + ["lightgray"]

    ax.stackplot(x, *[bin_fracs[:, i] for i in range(bin_fracs.shape[1])],
                 labels=labels_legend, colors=colors, alpha=0.8)

    ax.set_xlabel("Life Phase (%)", fontsize=12)
    ax.set_ylabel("Fraction of Timesteps", fontsize=12)
    ax.set_title(f"Cluster Distribution vs Life Phase (n={n_agents_used} agents, "
                 f"min_lifespan={args.min_lifespan})\n{data_dir.name}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    output_path = cluster_dir / f"cluster_life_phase_{data_dir.name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")

    # Also print the raw numbers
    print(f"\nCluster fractions by life phase bin:")
    print(f"{'Phase%':>7s}", end="")
    for lab in labels_legend:
        print(f"  {lab:>6s}", end="")
    print()
    for i, pct in enumerate(x):
        print(f"{pct:6.0f}%", end="")
        for j in range(bin_fracs.shape[1]):
            print(f"  {bin_fracs[i, j]:6.3f}", end="")
        print()

    import shutil
    if shutil.which("imgcat"):
        import subprocess
        subprocess.run(["imgcat", str(output_path)], check=False)


if __name__ == "__main__":
    main()
