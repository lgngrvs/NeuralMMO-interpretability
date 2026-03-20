"""Visualize agent life statistics over their lifespan.

By default picks the agent at the 75th percentile lifespan.  With --num_agents N,
randomly samples N agents and saves all plots to an output directory.

Memory-efficient: uses ijson to stream-parse the 2GB+ JSON files, keeping only
one record in memory at a time during the scan phase.
"""

import argparse
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Entity observation column indices (from nmmo EntityState) — same as analyze_activations.py
ENT_ID = 0
ENT_NPC_TYPE = 1
ENT_ATTACKER_ID = 8
ENT_LATEST_COMBAT_TICK = 9
ENT_GOLD = 11
ENT_HEALTH = 12
ENT_FOOD = 13
ENT_WATER = 14
ENT_MELEE_LVL = 15
ENT_RANGE_LVL = 17
ENT_MAGE_LVL = 19

# Action indices
ACT_MOVE_DIR = 8
ACT_MOVE_NOOP = 4
ACT_ATTACK_TARGET = 1
ACT_ATTACK_NOOP = 100
ACT_BUY_ITEM = 2
ACT_BUY_NOOP = 384
ACT_SELL_ITEM = 9
ACT_SELL_NOOP = 12
ACT_GIVE_ITEM = 4
ACT_GIVE_NOOP = 12


def extract_features(record):
    """Extract the same features as analyze_activations.py from a single record."""
    obs = record["observation"]
    action = record["action"]
    entity = np.array(obs["Entity"], dtype=np.float32)
    inventory = np.array(obs["Inventory"], dtype=np.float32)
    current_tick = np.array(obs["CurrentTick"], dtype=np.float32).item()
    agent_id = record["agent_id"]

    visible_mask = entity[:, ENT_ID] != 0
    visible_ents = entity[visible_mask]
    npc_mask = visible_ents[:, ENT_NPC_TYPE] > 0
    player_mask = visible_ents[:, ENT_NPC_TYPE] == 0

    self_mask = entity[:, ENT_ID] == agent_id
    if self_mask.any():
        self_row = entity[self_mask][0]
        self_health = self_row[ENT_HEALTH]
        self_food = self_row[ENT_FOOD]
        self_water = self_row[ENT_WATER]
        self_gold = self_row[ENT_GOLD]
        max_combat = max(self_row[ENT_MELEE_LVL], self_row[ENT_RANGE_LVL],
                         self_row[ENT_MAGE_LVL])
        # latest_combat_tick defaults to 0, so require it to be positive before
        # using the "recent combat" heuristic — otherwise tick 0-9 always reads
        # as in-combat even when no fight has happened.
        in_combat = (self_row[ENT_ATTACKER_ID] != 0 or
                     (self_row[ENT_LATEST_COMBAT_TICK] > 0 and
                      (current_tick - self_row[ENT_LATEST_COMBAT_TICK]) < 10))
        n_players = max(0, player_mask.sum() - 1)
        alive = self_row[ENT_HEALTH] > 0
    else:
        self_health = self_food = self_water = self_gold = max_combat = 0
        in_combat = False
        n_players = player_mask.sum()
        alive = False

    n_inv = (np.abs(inventory).sum(axis=1) > 0).sum()

    is_moving = action[ACT_MOVE_DIR] != ACT_MOVE_NOOP
    is_attacking = action[ACT_ATTACK_TARGET] != ACT_ATTACK_NOOP
    is_trading = (action[ACT_BUY_ITEM] != ACT_BUY_NOOP or
                  action[ACT_SELL_ITEM] != ACT_SELL_NOOP or
                  action[ACT_GIVE_ITEM] != ACT_GIVE_NOOP)

    return {
        "tick": current_tick,  # episode-local tick (may wrap)
        "alive": alive,
        "n_visible_entities": visible_mask.sum(),
        "n_visible_npcs": npc_mask.sum(),
        "n_visible_players": float(n_players),
        "self_health": self_health,
        "self_food": self_food,
        "self_water": self_water,
        "self_gold": self_gold,
        "max_combat_level": max_combat,
        "in_combat": float(in_combat),
        "n_inventory_items": float(n_inv),
        "is_moving": float(is_moving),
        "is_attacking": float(is_attacking),
        "is_trading": float(is_trading),
    }


def split_into_lives(feature_series, steps, cluster_labels=None):
    """Split feature_series and steps into separate lives based on gap detection."""
    if len(steps) < 2:
        cl = [cluster_labels] if cluster_labels is not None else [None]
        return [feature_series], [steps], cl

    ticks = [f["tick"] for f in feature_series]
    split_indices = [0]
    for i in range(len(steps) - 1):
        step_diff = steps[i + 1] - steps[i]
        tick_reset = ticks[i + 1] < ticks[i] - 50
        if step_diff > 20 or tick_reset:
            split_indices.append(i + 1)
    split_indices.append(len(steps))

    lives_features = []
    lives_steps = []
    lives_clusters = []
    for i in range(len(split_indices) - 1):
        s, e = split_indices[i], split_indices[i + 1]
        lives_features.append(feature_series[s:e])
        lives_steps.append(steps[s:e])
        if cluster_labels is not None:
            lives_clusters.append(cluster_labels[s:e])
        else:
            lives_clusters.append(None)

    return lives_features, lives_steps, lives_clusters


# Binary features get fill_between + step; continuous features get line plots.
BINARY_FEATURES = {"in_combat", "is_moving", "is_attacking", "is_trading"}

PANELS = [
    {
        "title": "Vitals",
        "features": ["self_health", "self_food", "self_water"],
        "colors": ["#e74c3c", "#e67e22", "#3498db"],
    },
    {
        "title": "Combat",
        "features": ["in_combat", "is_attacking", "max_combat_level"],
        "colors": ["#e74c3c", "#c0392b", "#8e44ad"],
    },
    {
        "title": "Nearby Entities",
        "features": ["n_visible_entities", "n_visible_npcs", "n_visible_players"],
        "colors": ["#2c3e50", "#27ae60", "#2980b9"],
    },
    {
        "title": "Economy & Inventory",
        "features": ["self_gold", "n_inventory_items"],
        "colors": ["#f1c40f", "#e67e22"],
        "twin_features": ["is_trading"],
        "twin_colors": ["#1abc9c"],
    },
    {
        "title": "Actions",
        "features": ["is_moving", "is_attacking", "is_trading"],
        "colors": ["#3498db", "#c0392b", "#1abc9c"],
    },
]


def _plot_panel(ax, panel, x_axis, feature_series):
    """Draw one panel's features onto ax. Returns legend handles/labels."""
    for feat_name, color in zip(panel["features"], panel["colors"]):
        values = np.array([f[feat_name] for f in feature_series], dtype=float)
        if feat_name in BINARY_FEATURES:
            ax.fill_between(x_axis, 0, values, alpha=0.3, color=color,
                            label=feat_name, step="mid")
            ax.step(x_axis, values, color=color, alpha=0.7, linewidth=0.8, where="mid")
        else:
            ax.plot(x_axis, values, color=color, label=feat_name,
                    linewidth=1.2, alpha=0.85)

    # Twin axis for binary features sharing a panel with continuous ones
    if panel.get("twin_features"):
        ax2 = ax.twinx()
        for feat_name, color in zip(panel["twin_features"], panel["twin_colors"]):
            values = np.array([f[feat_name] for f in feature_series], dtype=float)
            ax2.fill_between(x_axis, 0, values, alpha=0.25, color=color,
                             label=feat_name, step="mid")
            ax2.step(x_axis, values, color=color, alpha=0.6, linewidth=0.8, where="mid")
        ax2.set_ylim(-0.05, 1.15)
        ax2.set_ylabel("binary", fontsize=7, color="#888888")
        ax2.tick_params(axis="y", labelsize=7, colors="#888888")
        # Merge legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7)
        return
    ax.legend(loc="upper right", fontsize=7)


def _fill_cluster_labels(cluster_labels):
    """Fill unmatched (-1) cluster labels using nearest matched neighbor."""
    filled = cluster_labels.copy()
    matched = np.where(filled >= 0)[0]
    if len(matched) == 0:
        return filled
    # For each unmatched index, find the nearest matched index
    unmatched = np.where(filled < 0)[0]
    if len(unmatched) == 0:
        return filled
    # Use searchsorted to find nearest matched neighbor
    insert_pos = np.searchsorted(matched, unmatched)
    for i, pos in enumerate(insert_pos):
        if pos == 0:
            filled[unmatched[i]] = filled[matched[0]]
        elif pos >= len(matched):
            filled[unmatched[i]] = filled[matched[-1]]
        else:
            # Pick whichever matched index is closer
            left = matched[pos - 1]
            right = matched[pos]
            if unmatched[i] - left <= right - unmatched[i]:
                filled[unmatched[i]] = filled[left]
            else:
                filled[unmatched[i]] = filled[right]
    return filled


def _plot_cluster_panel(ax, x_axis, cluster_labels):
    """Draw the cluster panel as a colored strip, filling gaps by nearest neighbor."""
    filled = _fill_cluster_labels(cluster_labels)
    unique_clusters = sorted(set(filled[filled >= 0]))

    if not unique_clusters:
        ax.text(0.5, 0.5, "No cluster data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#999999")
        ax.set_yticks([])
        return

    cmap = plt.cm.get_cmap("tab10", max(max(unique_clusters) + 1, 1))

    # Draw colored spans for contiguous runs of the same cluster
    from itertools import groupby
    positions = list(range(len(filled)))
    for cluster_id, group in groupby(zip(positions, filled), key=lambda t: t[1]):
        indices = [idx for idx, _ in group]
        start_idx, end_idx = indices[0], indices[-1]
        # Extend to halfway between neighboring points for clean edges
        x_start = x_axis[start_idx]
        x_end = x_axis[end_idx]
        if start_idx > 0:
            x_start = (x_axis[start_idx - 1] + x_axis[start_idx]) / 2
        if end_idx < len(x_axis) - 1:
            x_end = (x_axis[end_idx] + x_axis[end_idx + 1]) / 2
        color = cmap(int(cluster_id)) if cluster_id >= 0 else "#cccccc"
        ax.axvspan(x_start, x_end, color=color, alpha=0.5)

    # Add legend patches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cmap(ci), alpha=0.5, label=f"Cluster {ci}")
               for ci in unique_clusters]
    # Show original match rate
    n_matched = (cluster_labels >= 0).sum()
    pct = 100 * n_matched / len(cluster_labels)
    ax.set_ylabel(f"Cluster\n({pct:.0f}% matched)", fontsize=8, fontweight="bold")
    ax.set_yticks([])
    ax.legend(handles=handles, loc="upper right", fontsize=7)


def plot_agent(feature_series, steps, target_key, cluster_labels, data_name, output_path,
               lifespan_rank=None, total_agents=None):
    """Plot and save a single agent's life timeline, one column per life."""
    alive_lifespan = len(feature_series)

    lives_feats, lives_steps, lives_clusters = split_into_lives(
        feature_series, steps, cluster_labels)
    n_lives = len(lives_feats)
    if n_lives > 1:
        print(f"    {n_lives} lives detected")

    panels = list(PANELS)
    has_clusters = cluster_labels is not None
    if has_clusters:
        panels.append({"title": "Cluster", "type": "cluster"})

    n_rows = len(panels)
    n_cols = n_lives

    # Column widths proportional to life length
    life_lengths = [len(s) for s in lives_steps]
    width_ratios = [max(l, 1) for l in life_lengths]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(4 * n_cols, 14), 2.5 * n_rows),
        gridspec_kw={"width_ratios": width_ratios} if n_cols > 1 else {},
        squeeze=False,
    )

    for col, (lf, ls, lc) in enumerate(zip(lives_feats, lives_steps, lives_clusters)):
        x_axis = np.array(ls, dtype=float)
        # Use episode-local tick as x-axis (starts at 0 each life)
        ticks = np.array([f["tick"] for f in lf], dtype=float)

        for row, panel in enumerate(panels):
            ax = axes[row, col]

            if panel.get("type") == "cluster":
                if lc is not None:
                    _plot_cluster_panel(ax, ticks, lc)
            else:
                _plot_panel(ax, panel, ticks, lf)

            ax.grid(True, alpha=0.2)
            ax.set_xlim(ticks[0], ticks[-1])

            # Row labels on leftmost column only
            if col == 0:
                if not panel.get("type") == "cluster":
                    ax.set_ylabel(panel["title"], fontsize=9, fontweight="bold")
                # cluster panel sets its own ylabel in _plot_cluster_panel
            else:
                if not panel.get("type") == "cluster":
                    ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

            # Column titles on top row
            if row == 0:
                ax.set_title(f"Life {col + 1} ({len(ls)} ticks)", fontsize=10)

            # X-axis label on bottom row
            if row == n_rows - 1:
                ax.set_xlabel("Tick", fontsize=9)

        # Share y-limits across columns for each row
    for row in range(n_rows):
        if panels[row].get("type") == "cluster":
            continue
        all_ylims = [axes[row, c].get_ylim() for c in range(n_cols)]
        ymin = min(lo for lo, hi in all_ylims)
        ymax = max(hi for lo, hi in all_ylims)
        for c in range(n_cols):
            axes[row, c].set_ylim(ymin, ymax)

    rank_str = f"  (lifespan rank {lifespan_rank}/{total_agents})" if lifespan_rank else ""
    lives_str = f", {n_lives} lives" if n_lives > 1 else ""
    fig.suptitle(
        f"Agent Life Timeline ({alive_lifespan} alive ticks{lives_str}){rank_str}\n"
        f"env_id={target_key[0]}, agent_id={target_key[1]}  |  data: {data_name}",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize agent life timelines")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation data")
    parser.add_argument("--cluster_dir", type=str, default=None,
                        help="Directory with cluster_labels.npy from analyze_activations.py")
    parser.add_argument("--num_agents", type=int, default=1,
                        help="Number of agents to visualize (randomly sampled). "
                             "Default 1 picks the p75 lifespan agent.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for agent sampling (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path: PNG file (single agent) or directory (multiple agents)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    json_files = list(data_dir.rglob("activations.json"))
    if not json_files:
        print(f"No activations.json found under {args.data_dir}")
        sys.exit(1)

    json_path = json_files[0]

    # --- Phase 1: Stream to count records per trajectory ---
    print(f"Phase 1: Streaming {json_path.name} to find trajectories...", flush=True)
    trajectory_counts = defaultdict(int)
    n = 0
    with open(json_path, "rb") as f:
        for item in ijson.items(f, "item", use_float=True):
            trajectory_counts[(item["env_id"], item["agent_id"])] += 1
            n += 1
            if n % 5000 == 0:
                print(f"  {n} records scanned, {len(trajectory_counts)} trajectories...",
                      end="\r", flush=True)

    print(f"  {n} records scanned, {len(trajectory_counts)} trajectories found     ")

    sorted_trajs = sorted(trajectory_counts.items(), key=lambda x: x[1])
    total_agents = len(sorted_trajs)

    print(f"  Lifespan distribution: min={sorted_trajs[0][1]}, "
          f"median={sorted_trajs[total_agents//2][1]}, "
          f"p75={sorted_trajs[int(0.75 * (total_agents - 1))][1]}, "
          f"max={sorted_trajs[-1][1]}")

    # Select agents
    if args.num_agents == 1:
        p75_idx = int(0.75 * (total_agents - 1))
        selected = [sorted_trajs[p75_idx]]
        print(f"  Selected p75 agent: env_id={selected[0][0][0]}, "
              f"agent_id={selected[0][0][1]}, lifespan={selected[0][1]}")
    else:
        rng = random.Random(args.seed)
        k = min(args.num_agents, total_agents)
        selected = rng.sample(sorted_trajs, k)
        selected.sort(key=lambda x: x[1])  # sort by lifespan for output naming
        print(f"  Randomly sampled {k} agents (seed={args.seed})")

    # Build set of target keys for Phase 2
    target_keys = {key for key, _ in selected}
    # Compute lifespan rank (1-indexed) for each selected agent within the full population
    rank_lookup = {}
    for i, (key, _count) in enumerate(sorted_trajs):
        if key in target_keys:
            rank_lookup[key] = i + 1

    # --- Load cluster labels if provided ---
    label_lookup = None
    if args.cluster_dir:
        cluster_path = Path(args.cluster_dir) / "cluster_labels.npy"
        if cluster_path.exists():
            all_labels = np.load(cluster_path, allow_pickle=True)
            label_lookup = {}
            for row in all_labels:
                label_lookup[(int(row[0]), int(row[1]), int(row[2]))] = int(row[3])
            print(f"  Loaded {len(label_lookup)} cluster labels from {cluster_path}")
        else:
            print(f"  Warning: {cluster_path} not found, skipping cluster overlay")

    # --- Phase 2: Stream again, collect records for all selected agents ---
    print(f"\nPhase 2: Extracting records for {len(selected)} agent(s)...", flush=True)
    agent_records = defaultdict(list)
    with open(json_path, "rb") as f:
        for item in ijson.items(f, "item", use_float=True):
            key = (item["env_id"], item["agent_id"])
            if key in target_keys:
                agent_records[key].append(item)

    # --- Determine output directory ---
    if args.num_agents > 1:
        if args.output:
            output_dir = Path(args.output)
        elif args.cluster_dir:
            output_dir = Path("life_visualizations") / Path(args.cluster_dir).name
        else:
            output_dir = Path("life_visualizations") / data_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot each agent ---
    saved_paths = []
    for agent_key, _lifespan in selected:
        records = agent_records.get(agent_key, [])
        records.sort(key=lambda r: r["step"])

        feature_series = []
        steps = []
        for r in records:
            feats = extract_features(r)
            if feats["alive"]:
                feature_series.append(feats)
                steps.append(r["step"])

        if not feature_series:
            print(f"  Skipping agent env_id={agent_key[0]}, agent_id={agent_key[1]}: "
                  f"no alive ticks")
            continue

        # Cluster labels for this agent
        cluster_labels = None
        if label_lookup is not None:
            cluster_labels = np.array([
                label_lookup.get((agent_key[0], agent_key[1], s), -1) for s in steps
            ])
            n_matched = (cluster_labels >= 0).sum()
            print(f"  Agent env={agent_key[0]} id={agent_key[1]}: "
                  f"{len(feature_series)} alive ticks, "
                  f"{n_matched}/{len(steps)} cluster-matched")
        else:
            print(f"  Agent env={agent_key[0]} id={agent_key[1]}: "
                  f"{len(feature_series)} alive ticks")

        rank = rank_lookup[agent_key]

        if args.num_agents == 1:
            out_path = args.output or f"agent_life_p75_{data_dir.name}.png"
        else:
            out_path = str(output_dir / f"agent-{rank}-of-{total_agents}.png")

        plot_agent(feature_series, steps, agent_key, cluster_labels,
                   data_dir.name, out_path,
                   lifespan_rank=rank, total_agents=total_agents)
        saved_paths.append(out_path)
        print(f"  Saved: {out_path}")

    # Display plots inline via imgcat
    for img_path in saved_paths:
        try:
            result = subprocess.run(["imgcat", img_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  imgcat {img_path} failed (exit {result.returncode}): "
                      f"{result.stderr.strip()}")
        except FileNotFoundError:
            break

    print(f"\nDone. {len(saved_paths)} plot(s) saved.")


if __name__ == "__main__":
    main()
