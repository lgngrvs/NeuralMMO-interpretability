"""Activation controls v2: combat-related sanity checks on NeuralMMO agent hidden states.

Tests:
  1. Health = 0 (from v1, for comparison)
  2. Health = 1 (barely alive vs healthy)
  3. latest_combat_tick = current tick (just fought)
  4. damage = 50
  5. freeze = 3 (max freeze)
  6. attacker_id = 0 (no attacker)
  7. Random baseline
"""

import torch
import torch.nn.functional as F
import numpy as np
import orjson
from pathlib import Path

# Entity column indices
COL_ID = 0
COL_NPC_TYPE = 1
COL_ROW = 2
COL_COL = 3
COL_DAMAGE = 4
COL_TIME_ALIVE = 5
COL_FREEZE = 6
COL_ITEM_LEVEL = 7
COL_ATTACKER_ID = 8
COL_LATEST_COMBAT_TICK = 9
COL_MESSAGE = 10
COL_GOLD = 11
COL_HEALTH = 12
COL_FOOD = 13
COL_WATER = 14


def load_model(policy_path):
    """Load policy checkpoint, return (inner_policy, lstm)."""
    model = torch.load(policy_path, map_location="cpu", weights_only=False)
    model.eval()
    recurrent_wrapper = model.policy
    inner = recurrent_wrapper.policy
    lstm = recurrent_wrapper.recurrent
    return inner, lstm


def load_data(path, n_samples=500):
    """Load activation records, filtering to live agents only."""
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    live = []
    for r in data:
        obs = r["observation"]
        tile = np.array(obs["Tile"])
        entity = np.array(obs["Entity"])
        if np.all(tile[:, 2] == 0):
            continue
        self_mask = entity[:, COL_ID] == r["agent_id"]
        if not self_mask.any():
            continue
        if entity[self_mask, COL_HEALTH][0] <= 0:
            continue
        live.append(r)
    print(f"Loaded {len(data)} records, {len(live)} live agents")
    if len(live) > n_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(live), n_samples, replace=False)
        live = [live[i] for i in sorted(indices)]
    return live


def obs_to_tensors(record):
    """Convert a single record's observation to tensors."""
    obs = record["observation"]
    tile = torch.tensor(obs["Tile"], dtype=torch.float32).unsqueeze(0)
    entity = torch.tensor(obs["Entity"], dtype=torch.float32).unsqueeze(0)
    inventory = torch.tensor(obs["Inventory"], dtype=torch.float32).unsqueeze(0)
    market = torch.tensor(obs["Market"], dtype=torch.float32).unsqueeze(0)
    agent_id = torch.tensor([record["agent_id"]], dtype=torch.float32)
    return tile, entity, inventory, market, agent_id


def encode(policy, lstm, tile, entity, inventory, market, agent_id):
    """Run observation through encoders + LSTM, returning hidden state."""
    tile_enc = policy.tile_encoder(tile)
    player_emb, my_agent = policy.player_encoder(entity, agent_id)
    item_emb = policy.item_encoder(inventory)
    inv_enc = policy.inventory_encoder(item_emb)
    market_emb = policy.item_encoder(market)
    market_enc = policy.market_encoder(market_emb)
    task = torch.zeros(1, 2048)
    task_enc = policy.task_encoder(task)
    obs = torch.cat([tile_enc, my_agent, inv_enc, market_enc, task_enc], dim=-1)
    hidden = policy.proj_fc(obs)
    hidden = hidden.unsqueeze(0)  # (T=1, B=1, H=256)
    hidden, _ = lstm(hidden)
    hidden = hidden.squeeze(0)  # (1, 256)
    return hidden


def compute_metrics(hidden_a, hidden_b):
    """Compute L2 distance and cosine similarity between hidden state pairs."""
    l2 = torch.norm(hidden_a - hidden_b, dim=-1).item()
    cos = F.cosine_similarity(hidden_a, hidden_b, dim=-1).item()
    return l2, cos


def run_entity_column_control(policy, lstm, records, column, value_fn, name):
    """Generic control: modify a self-entity column and measure hidden state change.

    value_fn: callable(record) -> scalar value to set for that column.
    """
    l2s, coss = [], []
    for rec in records:
        tile, entity, inv, market, aid = obs_to_tensors(rec)
        self_mask = entity[0, :, COL_ID] == aid[0]
        if not self_mask.any():
            continue
        with torch.no_grad():
            h_orig = encode(policy, lstm, tile, entity, inv, market, aid)
            entity_mod = entity.clone()
            entity_mod[0, self_mask, column] = value_fn(rec)
            h_mod = encode(policy, lstm, tile, entity_mod, inv, market, aid)
        l2, cos = compute_metrics(h_orig, h_mod)
        l2s.append(l2)
        coss.append(cos)
    return np.array(l2s), np.array(coss)


def run_random_baseline(policy, lstm, records):
    """Baseline: hidden state distances between random pairs of observations."""
    rng = np.random.default_rng(123)
    n = len(records)
    pairs = list(zip(rng.choice(n, 200, replace=False), rng.choice(n, 200, replace=False)))
    l2s, coss = [], []
    for i, j in pairs:
        if i == j:
            continue
        tile_i, ent_i, inv_i, mkt_i, aid_i = obs_to_tensors(records[i])
        tile_j, ent_j, inv_j, mkt_j, aid_j = obs_to_tensors(records[j])
        with torch.no_grad():
            h_i = encode(policy, lstm, tile_i, ent_i, inv_i, mkt_i, aid_i)
            h_j = encode(policy, lstm, tile_j, ent_j, inv_j, mkt_j, aid_j)
        l2, cos = compute_metrics(h_i, h_j)
        l2s.append(l2)
        coss.append(cos)
    return np.array(l2s), np.array(coss)


def main():
    policy_path = Path("policies/takeru_100M.pt")
    data_path = Path("takeru_100M/activations.json")

    print("Loading model...")
    policy, lstm = load_model(policy_path)
    print(f"Model loaded: {type(policy).__name__}")

    print("\nLoading data...")
    records = load_data(data_path, n_samples=500)

    # Define all controls
    controls = [
        ("Health = 0 (dead)", COL_HEALTH, lambda rec: 0),
        ("Health = 1 (barely alive)", COL_HEALTH, lambda rec: 1),
        ("latest_combat_tick = current tick", COL_LATEST_COMBAT_TICK,
         lambda rec: float(rec["observation"]["CurrentTick"][0])),
        ("damage = 50", COL_DAMAGE, lambda rec: 50),
        ("freeze = 3 (max)", COL_FREEZE, lambda rec: 3),
        ("attacker_id = 0 (no attacker)", COL_ATTACKER_ID, lambda rec: 0),
    ]

    results = {}

    for name, col, val_fn in controls:
        print(f"\nRunning: {name}...")
        l2s, coss = run_entity_column_control(policy, lstm, records, col, val_fn, name)
        results[name] = (l2s, coss)

    print("\nRunning: Random baseline...")
    rand_l2, rand_cos = run_random_baseline(policy, lstm, records)
    results["Random baseline"] = (rand_l2, rand_cos)

    # Print summary table
    baseline_mean_l2 = rand_l2.mean()

    print(f"\n{'=' * 90}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(f"  {'Control':<40s} {'N':>5s}  {'L2 mean':>8s} {'L2 std':>8s} "
          f"{'Cos mean':>8s} {'Cos std':>8s} {'% of rnd':>8s}")
    print(f"  {'-'*40} {'-'*5}  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name in list(dict.fromkeys(
        [c[0] for c in controls] + ["Random baseline"]
    )):
        l2s, coss = results[name]
        pct = f"{l2s.mean() / baseline_mean_l2:.1%}" if name != "Random baseline" else "100.0%"
        print(f"  {name:<40s} {len(l2s):>5d}  {l2s.mean():>8.4f} {l2s.std():>8.4f} "
              f"{coss.mean():>8.4f} {coss.std():>8.4f} {pct:>8s}")

    print(f"{'=' * 90}")

    # Detailed stats
    for name, (l2s, coss) in results.items():
        print(f"\n--- {name} ---")
        print(f"  L2:  mean={l2s.mean():.4f}  std={l2s.std():.4f}  "
              f"median={np.median(l2s):.4f}  min={l2s.min():.4f}  max={l2s.max():.4f}")
        print(f"  Cos: mean={coss.mean():.4f}  std={coss.std():.4f}  "
              f"median={np.median(coss):.4f}  min={coss.min():.4f}  max={coss.max():.4f}")


if __name__ == "__main__":
    main()
