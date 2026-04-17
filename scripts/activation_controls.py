"""Activation controls: sanity-check that model hidden states respond to observation changes.

Tests four manipulation conditions against a random-pair baseline:
  1. Tile terrain (original vs all-water)
  2. Self-entity health (original vs health=0)
  3. Inventory (original vs empty)
  4. Entity presence (action logit comparison: self-only vs self + adjacent agent)
"""

import torch
import torch.nn.functional as F
import numpy as np
import orjson
from pathlib import Path

# Entity column indices (from nmmo.entity.entity.EntityState)
COL_ID = 0
COL_NPC_TYPE = 1
COL_ROW = 2
COL_COL = 3
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
    print(f"  LSTM: {lstm}")
    return inner, lstm


def load_data(path, n_samples=500):
    """Load activation records, filtering to live agents only."""
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    # Filter to live agents: must have non-zero tiles and self-entity with health > 0
    live = []
    for r in data:
        obs = r["observation"]
        tile = np.array(obs["Tile"])
        entity = np.array(obs["Entity"])
        if np.all(tile[:, 2] == 0):
            continue  # dead agent (zeroed observation)
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
    """Run observation through encoders + LSTM, returning hidden state.

    Uses zero-initialized LSTM state (same as episode start).
    """
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
    # Pass through LSTM with zero state (T=1, B=1, H=256)
    hidden = hidden.unsqueeze(0)
    hidden, _ = lstm(hidden)
    hidden = hidden.squeeze(0)  # back to (1, 256)
    return hidden, player_emb, item_emb, market_emb


def compute_metrics(hidden_a, hidden_b):
    """Compute L2 distance and cosine similarity between hidden state pairs."""
    l2 = torch.norm(hidden_a - hidden_b, dim=-1).item()
    cos = F.cosine_similarity(hidden_a, hidden_b, dim=-1).item()
    return l2, cos


def run_tile_control(policy, lstm, records):
    """Control 1: Original tiles vs uniform grass terrain (material_id=2)."""
    l2s, coss = [], []
    for rec in records:
        tile, entity, inv, market, aid = obs_to_tensors(rec)
        with torch.no_grad():
            h_orig, _, _, _ = encode(policy, lstm, tile, entity, inv, market, aid)
            tile_mod = tile.clone()
            tile_mod[:, :, 2:] = 2  # all grass
            h_mod, _, _, _ = encode(policy, lstm, tile_mod, entity, inv, market, aid)
        l2, cos = compute_metrics(h_orig, h_mod)
        l2s.append(l2)
        coss.append(cos)
    return np.array(l2s), np.array(coss)


def run_health_control(policy, lstm, records):
    """Control 2: Original self-entity vs self-entity with health=0."""
    l2s, coss = [], []
    for rec in records:
        tile, entity, inv, market, aid = obs_to_tensors(rec)
        self_mask = entity[0, :, COL_ID] == aid[0]
        if not self_mask.any():
            continue
        with torch.no_grad():
            h_orig, _, _, _ = encode(policy, lstm, tile, entity, inv, market, aid)
            entity_mod = entity.clone()
            entity_mod[0, self_mask, COL_HEALTH] = 0
            h_mod, _, _, _ = encode(policy, lstm, tile, entity_mod, inv, market, aid)
        l2, cos = compute_metrics(h_orig, h_mod)
        l2s.append(l2)
        coss.append(cos)
    return np.array(l2s), np.array(coss)


def run_inventory_control(policy, lstm, records):
    """Control 3: Original inventory vs empty inventory."""
    l2s, coss = [], []
    for rec in records:
        tile, entity, inv, market, aid = obs_to_tensors(rec)
        with torch.no_grad():
            h_orig, _, _, _ = encode(policy, lstm, tile, entity, inv, market, aid)
            inv_mod = torch.zeros_like(inv)
            h_mod, _, _, _ = encode(policy, lstm, tile, entity, inv_mod, market, aid)
        l2, cos = compute_metrics(h_orig, h_mod)
        l2s.append(l2)
        coss.append(cos)
    return np.array(l2s), np.array(coss)


def run_entity_control(policy, lstm, records):
    """Control 4: Self-only entity list vs self + adjacent agent.

    Measures action logit differences since other entities don't affect hidden state.
    """
    attack_style_diffs = []
    move_diffs = []
    attack_target_max_diffs = []

    for rec in records:
        tile, entity, inv, market, aid = obs_to_tensors(rec)
        self_mask = entity[0, :, COL_ID] == aid[0]
        if not self_mask.any():
            continue
        self_idx = self_mask.nonzero(as_tuple=True)[0][0].item()
        self_row = entity[0, self_idx, COL_ROW].item()
        self_col = entity[0, self_idx, COL_COL].item()

        # Condition A: self only
        entity_a = torch.zeros_like(entity)
        entity_a[0, 0] = entity[0, self_idx]

        # Condition B: self + one adjacent agent
        entity_b = torch.zeros_like(entity)
        entity_b[0, 0] = entity[0, self_idx]
        entity_b[0, 1, COL_ID] = 999
        entity_b[0, 1, COL_NPC_TYPE] = 0
        entity_b[0, 1, COL_ROW] = self_row + 1
        entity_b[0, 1, COL_COL] = self_col
        entity_b[0, 1, COL_HEALTH] = 100
        entity_b[0, 1, COL_FOOD] = 100
        entity_b[0, 1, COL_WATER] = 100

        with torch.no_grad():
            h_a, pe_a, _, _ = encode(policy, lstm, tile, entity_a, inv, market, aid)
            h_b, pe_b, _, _ = encode(policy, lstm, tile, entity_b, inv, market, aid)

            # Simple action heads (no embedding lookup)
            as_a = policy.action_decoder.layers["attack_style"](h_a)
            as_b = policy.action_decoder.layers["attack_style"](h_b)
            move_a = policy.action_decoder.layers["move"](h_a)
            move_b = policy.action_decoder.layers["move"](h_b)

            # Attack target logits (embedding dot product)
            at_proj_a = policy.action_decoder.layers["attack_target"](h_a)
            at_logits_a = torch.matmul(pe_a, at_proj_a.unsqueeze(-1)).squeeze(-1)
            at_proj_b = policy.action_decoder.layers["attack_target"](h_b)
            at_logits_b = torch.matmul(pe_b, at_proj_b.unsqueeze(-1)).squeeze(-1)

        attack_style_diffs.append(torch.norm(as_a - as_b).item())
        move_diffs.append(torch.norm(move_a - move_b).item())
        min_len = min(at_logits_a.shape[1], at_logits_b.shape[1])
        at_diff = (at_logits_a[:, :min_len] - at_logits_b[:, :min_len]).abs().max().item()
        attack_target_max_diffs.append(at_diff)

    return (
        np.array(attack_style_diffs),
        np.array(move_diffs),
        np.array(attack_target_max_diffs),
    )


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
            h_i, _, _, _ = encode(policy, lstm, tile_i, ent_i, inv_i, mkt_i, aid_i)
            h_j, _, _, _ = encode(policy, lstm, tile_j, ent_j, inv_j, mkt_j, aid_j)
        l2, cos = compute_metrics(h_i, h_j)
        l2s.append(l2)
        coss.append(cos)
    return np.array(l2s), np.array(coss)


def print_stats(name, l2s, coss):
    """Print summary statistics for a control condition."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  N samples:      {len(l2s)}")
    print(f"  L2 distance:    mean={l2s.mean():.4f}  std={l2s.std():.4f}  "
          f"median={np.median(l2s):.4f}  min={l2s.min():.4f}  max={l2s.max():.4f}")
    print(f"  Cosine sim:     mean={coss.mean():.4f}  std={coss.std():.4f}  "
          f"median={np.median(coss):.4f}  min={coss.min():.4f}  max={coss.max():.4f}")


def main():
    policy_path = Path("policies/takeru_100M.pt")
    data_path = Path("takeru_100M/activations.json")

    print("Loading model...")
    policy, lstm = load_model(policy_path)
    print(f"Model loaded: {type(policy).__name__}")

    # Verify encoding matches saved activations
    print("\nVerification: comparing manual encode vs saved activation...")
    with open(data_path, "rb") as f:
        all_data = orjson.loads(f.read())
    rec = all_data[100]
    tile, entity, inv, market, aid = obs_to_tensors(rec)
    with torch.no_grad():
        h, _, _, _ = encode(policy, lstm, tile, entity, inv, market, aid)
    saved = np.array(rec["activation"])
    l2_check = np.linalg.norm(h.numpy().flatten() - saved)
    print(f"  Computed norm: {h.norm():.4f}, Saved norm: {np.linalg.norm(saved):.4f}")
    print(f"  L2 diff: {l2_check:.4f}")
    if l2_check < 1.0:
        print("  PASS: Manual encoding matches saved activations")
    else:
        print(f"  WARNING: Large mismatch (L2={l2_check:.2f}). Results may not reflect")
        print(f"  true model behavior (expected due to zero-init LSTM state vs real state).")
    del all_data

    print("\nLoading data...")
    records = load_data(data_path, n_samples=500)

    # Run all controls
    print("\nRunning Control 1: Tile Terrain...")
    tile_l2, tile_cos = run_tile_control(policy, lstm, records)
    print_stats("Control 1: Tile Terrain (original vs all-grass)", tile_l2, tile_cos)

    print("\nRunning Control 2: Self-Entity Health...")
    health_l2, health_cos = run_health_control(policy, lstm, records)
    print_stats("Control 2: Self-Entity Health (original vs health=0)", health_l2, health_cos)

    print("\nRunning Control 3: Inventory...")
    inv_l2, inv_cos = run_inventory_control(policy, lstm, records)
    print_stats("Control 3: Inventory (original vs empty)", inv_l2, inv_cos)

    print("\nRunning Random Baseline...")
    rand_l2, rand_cos = run_random_baseline(policy, lstm, records)
    print_stats("Baseline: Random Pairs", rand_l2, rand_cos)

    # Effect sizes
    print(f"\n{'=' * 60}")
    print(f"  Effect Sizes (control L2 / baseline L2)")
    print(f"{'=' * 60}")
    baseline_mean = rand_l2.mean()
    print(f"  Tile terrain:   {tile_l2.mean() / baseline_mean:.2%} of random baseline")
    print(f"  Self health:    {health_l2.mean() / baseline_mean:.2%} of random baseline")
    print(f"  Inventory:      {inv_l2.mean() / baseline_mean:.2%} of random baseline")

    # Control 4
    print("\nRunning Control 4: Entity Presence (action logits)...")
    as_diffs, mv_diffs, at_diffs = run_entity_control(policy, lstm, records)
    print(f"\n{'=' * 60}")
    print(f"  Control 4: Entity Presence (self-only vs self + adjacent)")
    print(f"{'=' * 60}")
    print(f"  N samples:              {len(as_diffs)}")
    print(f"  Attack style L2:        mean={as_diffs.mean():.4f}  std={as_diffs.std():.4f}")
    print(f"  Move logit L2:          mean={mv_diffs.mean():.4f}  std={mv_diffs.std():.4f}")
    print(f"  Attack target max diff: mean={at_diffs.mean():.4f}  std={at_diffs.std():.4f}")
    print(f"\n  Note: attack_style and move logits should be ~0 (entities don't")
    print(f"  affect hidden state). attack_target should show difference (uses")
    print(f"  entity embeddings for dot-product lookup).")


if __name__ == "__main__":
    main()
