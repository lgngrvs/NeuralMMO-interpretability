"""Visualize a single agent's life statistics over its lifespan.

Picks the agent at the 75th percentile lifespan and plots the same features
that analyze_activations.py computes, tick-by-tick.

Memory-efficient: uses ijson to stream-parse the 2GB+ JSON files, keeping only
one record in memory at a time during the scan phase.
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Visualize a single agent's life")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation data")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: agent_life_p75_<name>.png)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    json_files = list(data_dir.rglob("activations.json"))
    if not json_files:
        print(f"No activations.json found under {args.data_dir}")
        sys.exit(1)

    json_path = json_files[0]

    # --- Phase 1: Stream to count records per trajectory ---
    # ijson.items yields one parsed object at a time; memory stays flat.
    print(f"Phase 1: Streaming {json_path.name} to find p75 agent...", flush=True)
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
    p75_idx = int(0.75 * (len(sorted_trajs) - 1))
    target_key, target_lifespan = sorted_trajs[p75_idx]

    print(f"  Lifespan distribution: min={sorted_trajs[0][1]}, "
          f"median={sorted_trajs[len(sorted_trajs)//2][1]}, "
          f"p75={target_lifespan}, max={sorted_trajs[-1][1]}")
    print(f"  Selected agent: env_id={target_key[0]}, agent_id={target_key[1]}, "
          f"lifespan={target_lifespan} records")

    # --- Phase 2: Stream again, keep only target agent's records ---
    target_env, target_agent = target_key
    print(f"\nPhase 2: Extracting records for target agent...", flush=True)
    target_records = []
    with open(json_path, "rb") as f:
        for item in ijson.items(f, "item", use_float=True):
            if item["env_id"] == target_env and item["agent_id"] == target_agent:
                target_records.append(item)

    target_records.sort(key=lambda r: r["step"])
    print(f"  Loaded {len(target_records)} records for target agent")

    # Extract features (alive ticks only)
    feature_series = []
    steps = []
    for r in target_records:
        feats = extract_features(r)
        if feats["alive"]:
            feature_series.append(feats)
            steps.append(r["step"])

    if not feature_series:
        print("No alive ticks found for selected agent!")
        sys.exit(1)

    alive_lifespan = len(feature_series)
    # Use step (monotonic) as x-axis, not CurrentTick (wraps per episode)
    x_axis = np.array(steps)
    print(f"  {alive_lifespan} alive ticks to visualize")

    # --- Plot ---
    panels = [
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
            "features": ["self_gold", "n_inventory_items", "is_trading"],
            "colors": ["#f1c40f", "#e67e22", "#1abc9c"],
        },
        {
            "title": "Movement",
            "features": ["is_moving"],
            "colors": ["#3498db"],
        },
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(16, 3 * len(panels)),
                              sharex=True)

    for ax, panel in zip(axes, panels):
        for feat_name, color in zip(panel["features"], panel["colors"]):
            values = np.array([f[feat_name] for f in feature_series])
            if feat_name in ("in_combat", "is_moving", "is_attacking", "is_trading"):
                ax.fill_between(x_axis, 0, values, alpha=0.3, color=color,
                                label=feat_name, step="mid")
                ax.step(x_axis, values, color=color, alpha=0.7, linewidth=0.8, where="mid")
            else:
                ax.plot(x_axis, values, color=color, label=feat_name,
                        linewidth=1.2, alpha=0.85)

        ax.set_ylabel(panel["title"], fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=len(panel["features"]))
        ax.grid(True, alpha=0.2)
        ax.set_xlim(x_axis[0], x_axis[-1])

    axes[-1].set_xlabel("Step", fontsize=11)
    fig.suptitle(
        f"Agent Life Timeline (p75 lifespan = {alive_lifespan} alive ticks)\n"
        f"env_id={target_key[0]}, agent_id={target_key[1]}  |  data: {data_dir.name}",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    output_path = args.output or f"agent_life_p75_{data_dir.name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
