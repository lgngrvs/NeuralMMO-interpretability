"""Cluster activation vectors and test for interpretable behavioral modes.

Loads per-timestep activation records (from extract_activations.py), subsamples
trajectories, runs UMAP + HDBSCAN, and checks whether clusters correspond to
distinct behavioral features with large effect sizes vs random baselines.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import struct
import sys
import threading
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Entity observation column indices (from nmmo EntityState)
ENT_ID = 0
ENT_NPC_TYPE = 1
ENT_ROW = 2
ENT_COL = 3
ENT_DAMAGE = 4
ENT_TIME_ALIVE = 5
ENT_FREEZE = 6
ENT_ITEM_LEVEL = 7
ENT_ATTACKER_ID = 8
ENT_LATEST_COMBAT_TICK = 9
ENT_GOLD = 11
ENT_HEALTH = 12
ENT_FOOD = 13
ENT_WATER = 14
ENT_MELEE_LVL = 15
ENT_RANGE_LVL = 17
ENT_MAGE_LVL = 19
ENT_FISHING_LVL = 21
ENT_HERBALISM_LVL = 23
ENT_PROSPECTING_LVL = 25
ENT_CARVING_LVL = 27
ENT_ALCHEMY_LVL = 29

# Tile observation column indices
TILE_ROW = 0
TILE_COL = 1
TILE_MATERIAL = 2

# Item/Inventory column indices
ITEM_ID = 0
ITEM_LEVEL = 3
ITEM_EQUIPPED = 14
ITEM_LISTED_PRICE = 15

# Tile material IDs
MAT_WATER = 1
MAT_FOILAGE = 4  # forest canopy
MAT_TREE = 9
MAT_OCEAN = 14

# Action indices in the flattened MultiDiscrete (alphabetical, Comm excluded)
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
ACT_USE_ITEM = 11
ACT_USE_NOOP = 12

SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@contextmanager
def spinner(message):
    """Show a spinning indicator while a block runs."""
    stop = threading.Event()

    def spin():
        i = 0
        while not stop.is_set():
            sys.stderr.write(f"\r{SPINNER_CHARS[i % len(SPINNER_CHARS)]} {message}")
            sys.stderr.flush()
            i += 1
            time.sleep(0.1)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        sys.stderr.write(f"\r✓ {message}\n")
        sys.stderr.flush()


FEATURE_NAMES = [
    # Nearby entities
    "n_visible_entities",
    "n_visible_npcs",
    "n_visible_players",
    "nearest_entity_dist",
    "nearest_player_dist",
    # Self vitals
    "self_health",
    "self_food",
    "self_water",
    "self_gold",
    # Combat
    "max_combat_level",
    "melee_level",
    "range_level",
    "mage_level",
    "in_combat",
    "self_damage",
    # Harvest/profession skills
    "fishing_level",
    "herbalism_level",
    "prospecting_level",
    "carving_level",
    "alchemy_level",
    "max_harvest_level",
    # Equipment & inventory
    "item_level",
    "n_inventory_items",
    "n_equipped_items",
    "n_listed_items",
    # Spatial
    "self_row",
    "self_col",
    # Terrain around agent
    "n_water_tiles",
    "n_forest_tiles",
    # Time
    "tick",
    "time_alive",
    # Actions
    "is_moving",
    "is_attacking",
    "is_trading",
    "is_using_item",
]


def _is_alive(record):
    """Check if the agent is alive (self row present with health > 0)."""
    agent_id = record["agent_id"]
    if agent_id == 0:  # padding/empty agent slot
        return False
    entity = record["observation"]["Entity"]
    for row in entity:
        if row[ENT_ID] == agent_id:
            return row[ENT_HEALTH] > 0
    return False


# ---------------------------------------------------------------------------
# Fast binary-cache loader
# ---------------------------------------------------------------------------
# On first load of a JSONL file, we parse it in parallel, extract features,
# filter dead agents, and save the results as .npz.  Subsequent loads just
# mmap the .npz -- loading 500k records in seconds instead of 30+ minutes.
# ---------------------------------------------------------------------------

def _extract_features_from_record(r):
    """Extract (activation, features_vec, step, env_id, agent_id) from one record.

    Returns None if the agent is dead.
    """
    agent_id = r["agent_id"]
    if agent_id == 0:
        return None

    obs = r["observation"]
    entity_raw = obs["Entity"]

    # Check alive inline -- avoid creating numpy array just for this
    self_row_raw = None
    for row in entity_raw:
        if row[ENT_ID] == agent_id:
            if row[ENT_HEALTH] <= 0:
                return None
            self_row_raw = row
            break
    if self_row_raw is None:
        return None

    action = r["action"]
    entity = np.array(entity_raw, dtype=np.float32)
    inventory = np.array(obs["Inventory"], dtype=np.float32)
    tile = np.array(obs["Tile"], dtype=np.float32)
    current_tick_raw = obs["CurrentTick"]
    current_tick = float(current_tick_raw[0]) if isinstance(current_tick_raw, (list, np.ndarray)) else float(current_tick_raw)

    # Visible entities: rows where id != 0
    visible_mask = entity[:, ENT_ID] != 0
    n_visible = visible_mask.sum()

    visible_ents = entity[visible_mask]
    npc_mask = visible_ents[:, ENT_NPC_TYPE] > 0
    player_mask = visible_ents[:, ENT_NPC_TYPE] == 0

    self_ent = entity[entity[:, ENT_ID] == agent_id][0]
    self_health = self_ent[ENT_HEALTH]
    self_food = self_ent[ENT_FOOD]
    self_water = self_ent[ENT_WATER]
    self_gold = self_ent[ENT_GOLD]
    melee_lvl = self_ent[ENT_MELEE_LVL]
    range_lvl = self_ent[ENT_RANGE_LVL]
    mage_lvl = self_ent[ENT_MAGE_LVL]
    max_combat = max(melee_lvl, range_lvl, mage_lvl)
    in_combat = (self_ent[ENT_ATTACKER_ID] != 0 or
                 (current_tick - self_ent[ENT_LATEST_COMBAT_TICK]) < 10)
    self_damage = self_ent[ENT_DAMAGE]
    self_r = self_ent[ENT_ROW]
    self_c = self_ent[ENT_COL]
    item_level = self_ent[ENT_ITEM_LEVEL]
    time_alive = self_ent[ENT_TIME_ALIVE]
    fishing_lvl = self_ent[ENT_FISHING_LVL]
    herbalism_lvl = self_ent[ENT_HERBALISM_LVL]
    prospecting_lvl = self_ent[ENT_PROSPECTING_LVL]
    carving_lvl = self_ent[ENT_CARVING_LVL]
    alchemy_lvl = self_ent[ENT_ALCHEMY_LVL]
    max_harvest = max(fishing_lvl, herbalism_lvl, prospecting_lvl,
                      carving_lvl, alchemy_lvl)
    n_players = player_mask.sum() - 1  # Don't count self

    # Nearest entity/player distances (Chebyshev)
    if n_visible > 1:
        others = visible_ents[visible_ents[:, ENT_ID] != agent_id]
        if len(others) > 0:
            dists = np.maximum(np.abs(others[:, ENT_ROW] - self_r),
                               np.abs(others[:, ENT_COL] - self_c))
            nearest_entity_dist = dists.min()
            player_others = others[others[:, ENT_NPC_TYPE] == 0]
            nearest_player_dist = (
                np.maximum(np.abs(player_others[:, ENT_ROW] - self_r),
                           np.abs(player_others[:, ENT_COL] - self_c)).min()
                if len(player_others) > 0 else 99.0
            )
        else:
            nearest_entity_dist = 99.0
            nearest_player_dist = 99.0
    else:
        nearest_entity_dist = 99.0
        nearest_player_dist = 99.0

    # Inventory features
    inv_mask = inventory[:, ITEM_ID] != 0
    n_inv = inv_mask.sum()
    n_equipped = (inventory[inv_mask, ITEM_EQUIPPED] > 0).sum() if n_inv > 0 else 0
    n_listed = (inventory[inv_mask, ITEM_LISTED_PRICE] > 0).sum() if n_inv > 0 else 0

    # Tile features
    tile_mats = tile[:, TILE_MATERIAL]
    n_water_tiles = ((tile_mats == MAT_WATER) | (tile_mats == MAT_OCEAN)).sum()
    n_forest_tiles = ((tile_mats == MAT_FOILAGE) | (tile_mats == MAT_TREE)).sum()

    # Action features
    is_moving = action[ACT_MOVE_DIR] != ACT_MOVE_NOOP
    is_attacking = action[ACT_ATTACK_TARGET] != ACT_ATTACK_NOOP
    is_trading = (action[ACT_BUY_ITEM] != ACT_BUY_NOOP or
                  action[ACT_SELL_ITEM] != ACT_SELL_NOOP or
                  action[ACT_GIVE_ITEM] != ACT_GIVE_NOOP)
    is_using_item = action[ACT_USE_ITEM] != ACT_USE_NOOP

    feat = np.array([
        n_visible, npc_mask.sum(), max(0, n_players),
        nearest_entity_dist, nearest_player_dist,
        self_health, self_food, self_water, self_gold,
        max_combat, melee_lvl, range_lvl, mage_lvl,
        float(in_combat), self_damage,
        fishing_lvl, herbalism_lvl, prospecting_lvl,
        carving_lvl, alchemy_lvl, max_harvest,
        item_level, n_inv, n_equipped, n_listed,
        self_r, self_c,
        n_water_tiles, n_forest_tiles,
        current_tick, time_alive,
        float(is_moving), float(is_attacking),
        float(is_trading), float(is_using_item),
    ], dtype=np.float32)

    return (
        np.array(r["activation"], dtype=np.float32),
        feat,
        r["step"],
        r["env_id"],
        agent_id,
    )


def _process_chunk(args):
    """Process a chunk of a JSONL file: parse JSON, filter dead, extract features.

    args: (file_path, start_offset, end_offset, chunk_id)
    Returns: (activations, features, steps, env_ids, agent_ids) as numpy arrays,
             or None if no alive records found.
    """
    import orjson

    file_path, start_offset, end_offset, chunk_id = args

    activations_list = []
    features_list = []
    steps = []
    env_ids = []
    agent_ids = []

    with open(file_path, "rb") as fh:
        fh.seek(start_offset)
        # If we're not at the start, skip partial first line
        if start_offset > 0:
            fh.readline()

        while True:
            pos = fh.tell()
            if pos >= end_offset:
                break
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                r = orjson.loads(line)
            except Exception:
                continue

            result = _extract_features_from_record(r)
            if result is None:
                continue

            act, feat, step, env_id, agent_id = result
            activations_list.append(act)
            features_list.append(feat)
            steps.append(step)
            env_ids.append(env_id)
            agent_ids.append(agent_id)

    if not activations_list:
        return None

    return (
        np.stack(activations_list),
        np.stack(features_list),
        np.array(steps, dtype=np.int64),
        np.array(env_ids, dtype=np.int64),
        np.array(agent_ids, dtype=np.int64),
    )


def _find_line_offsets(file_path, n_chunks):
    """Find byte offsets that split a file into roughly equal chunks at line boundaries."""
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // n_chunks
    offsets = [0]

    with open(file_path, "rb") as fh:
        for i in range(1, n_chunks):
            fh.seek(i * chunk_size)
            fh.readline()  # skip to next line boundary
            offsets.append(fh.tell())

    offsets.append(file_size)
    return offsets


def _cache_path_for_file(json_file):
    """Return the path for the binary cache of a given JSONL file."""
    return json_file.with_suffix(".cache.npz")


def _build_cache_parallel(json_file, n_workers=None):
    """Parse a JSONL file in parallel, extract features, and save as .npz cache.

    Returns (activations, features, steps, env_ids, agent_ids) numpy arrays
    for all alive records.
    """
    import orjson

    file_size = os.path.getsize(json_file)
    if n_workers is None:
        # Use many workers but cap at a reasonable number
        n_workers = min(multiprocessing.cpu_count(), 64)

    print(f"  Building cache with {n_workers} workers...", flush=True)

    if str(json_file).endswith(".jsonl"):
        offsets = _find_line_offsets(str(json_file), n_workers)
        chunk_args = [
            (str(json_file), offsets[i], offsets[i + 1], i)
            for i in range(len(offsets) - 1)
        ]

        with multiprocessing.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_chunk, chunk_args),
                total=len(chunk_args),
                desc="Processing chunks",
                unit="chunk",
            ))
    else:
        # .json file: load all at once, then process in parallel via chunks
        with open(json_file, "rb") as fh:
            all_records = orjson.loads(fh.read())
        # Process sequentially for .json (rare case)
        results = []
        acts, feats, ss, es, as_ = [], [], [], [], []
        for r in tqdm(all_records, desc="Processing records", unit="rec"):
            result = _extract_features_from_record(r)
            if result is not None:
                act, feat, step, env_id, agent_id = result
                acts.append(act)
                feats.append(feat)
                ss.append(step)
                es.append(env_id)
                as_.append(agent_id)
        if acts:
            results = [(np.stack(acts), np.stack(feats),
                        np.array(ss, dtype=np.int64),
                        np.array(es, dtype=np.int64),
                        np.array(as_, dtype=np.int64))]

    # Merge results from all chunks
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise ValueError(f"No alive records found in {json_file}")

    all_acts = np.concatenate([r[0] for r in valid_results])
    all_feats = np.concatenate([r[1] for r in valid_results])
    all_steps = np.concatenate([r[2] for r in valid_results])
    all_envs = np.concatenate([r[3] for r in valid_results])
    all_agents = np.concatenate([r[4] for r in valid_results])

    # Save cache
    cache_file = _cache_path_for_file(json_file)
    print(f"  Saving cache to {cache_file.name} ...", flush=True)
    np.savez(
        cache_file,
        activations=all_acts,
        features=all_feats,
        steps=all_steps,
        env_ids=all_envs,
        agent_ids=all_agents,
    )
    cache_size = os.path.getsize(cache_file)
    print(f"  Cache saved ({cache_size / 1024 / 1024:.0f} MB)", flush=True)

    return all_acts, all_feats, all_steps, all_envs, all_agents


def _load_or_build_cache(json_file):
    """Load from binary cache if available, otherwise build it.

    Returns (activations, features, steps, env_ids, agent_ids) numpy arrays.
    """
    cache_file = _cache_path_for_file(json_file)
    file_size = os.path.getsize(json_file)

    if cache_file.exists():
        cache_mtime = os.path.getmtime(cache_file)
        source_mtime = os.path.getmtime(json_file)
        if cache_mtime > source_mtime:
            print(f"  Loading from cache {cache_file.name}...", flush=True)
            data = np.load(cache_file)
            print(f"  Cache loaded: {len(data['activations'])} alive records", flush=True)
            return (
                data["activations"],
                data["features"],
                data["steps"],
                data["env_ids"],
                data["agent_ids"],
            )
        else:
            print(f"  Cache outdated, rebuilding...", flush=True)

    return _build_cache_parallel(json_file)


def load_data(path, subsample_rate=10):
    """Load activation records from a directory, subsampling per trajectory.

    Filters out dead agent timesteps before subsampling.
    Returns (subsampled_records, total_datapoints) where total_datapoints is the
    count before subsampling/filtering.

    When a binary cache (.cache.npz) exists and is newer than the source file,
    loading is near-instant.  Otherwise the cache is built in parallel on first
    load (takes a few minutes once, then cached for future runs).

    The returned records are lightweight dicts with pre-computed numpy arrays
    stored in a _precomputed_arrays attribute on the list for use by
    compute_features().
    """
    data_dir = Path(path)

    json_files = list(data_dir.rglob("activations.json"))
    jsonl_files = list(data_dir.rglob("activations.jsonl"))
    all_data_files = json_files + jsonl_files
    if not all_data_files:
        raise FileNotFoundError(f"No activations.json or activations.jsonl found under {path}")

    all_activations = []
    all_features = []
    all_steps = []
    all_env_ids = []
    all_agent_ids = []
    total_datapoints = 0

    for json_file in all_data_files:
        file_size = os.path.getsize(json_file)
        print(f"Loading {json_file.name} ({file_size / 1024 / 1024:.0f} MB)...",
              flush=True)

        acts, feats, steps, env_ids, agent_ids = _load_or_build_cache(json_file)

        # Count total records (alive records in cache; estimate total from file)
        n_alive = len(acts)
        # We don't know exact total without parsing, but we can estimate
        # For reporting purposes, use alive count as lower bound
        total_datapoints += n_alive  # Will be updated below if we can count

        # Try to get actual total line count from a sidecar or estimate
        # For JSONL, estimate from file size and average line size
        if json_file.suffix == ".jsonl":
            avg_line_size = file_size / max(n_alive, 1)
            estimated_total = int(file_size / avg_line_size) if avg_line_size > 0 else n_alive
            total_datapoints = total_datapoints - n_alive + estimated_total

        print(f"  {n_alive} alive records", flush=True)

        # Subsample per trajectory: group by (env_id, agent_id), sort by step, keep every Nth
        if subsample_rate and subsample_rate > 1:
            # Build trajectory groups using numpy operations
            traj_keys = env_ids.astype(np.int64) * 1_000_000 + agent_ids.astype(np.int64)
            unique_keys = np.unique(traj_keys)

            keep_mask = np.zeros(len(acts), dtype=bool)
            for key in unique_keys:
                mask = traj_keys == key
                indices = np.where(mask)[0]
                # Sort by step within trajectory
                sorted_order = np.argsort(steps[indices])
                sorted_indices = indices[sorted_order]
                # Keep every Nth
                keep_mask[sorted_indices[::subsample_rate]] = True

            acts = acts[keep_mask]
            feats = feats[keep_mask]
            steps = steps[keep_mask]
            env_ids = env_ids[keep_mask]
            agent_ids = agent_ids[keep_mask]
            print(f"  {keep_mask.sum()} records after subsampling (rate={subsample_rate})", flush=True)

        all_activations.append(acts)
        all_features.append(feats)
        all_steps.append(steps)
        all_env_ids.append(env_ids)
        all_agent_ids.append(agent_ids)

    # Concatenate across files
    final_acts = np.concatenate(all_activations)
    final_feats = np.concatenate(all_features)
    final_steps = np.concatenate(all_steps)
    final_env_ids = np.concatenate(all_env_ids)
    final_agent_ids = np.concatenate(all_agent_ids)

    n_final = len(final_acts)
    print(f"Loaded {total_datapoints} records, {n_final} after filtering + subsampling (rate={subsample_rate})")

    # Build lightweight record dicts for backward compatibility.
    # The actual numpy data is passed through a _precomputed attribute on the list.
    records = _PrecomputedRecordList(final_acts, final_feats, final_steps,
                                     final_env_ids, final_agent_ids)

    return records, total_datapoints


class _PrecomputedRecordList:
    """A list-like object that carries precomputed numpy arrays.

    Supports len() and iteration for backward compatibility, but the real data
    lives in the numpy arrays accessed by compute_features().
    """

    def __init__(self, activations, features, steps, env_ids, agent_ids):
        self._activations = activations
        self._features = features
        self._steps = steps
        self._env_ids = env_ids
        self._agent_ids = agent_ids
        self._n = len(activations)

    def __len__(self):
        return self._n

    def __iter__(self):
        """Yield lightweight dicts for backward compatibility."""
        for i in range(self._n):
            yield {
                "activation": self._activations[i],
                "step": int(self._steps[i]),
                "env_id": int(self._env_ids[i]),
                "agent_id": int(self._agent_ids[i]),
            }

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            new = _PrecomputedRecordList(
                self._activations[idx],
                self._features[idx],
                self._steps[idx],
                self._env_ids[idx],
                self._agent_ids[idx],
            )
            return new
        return {
            "activation": self._activations[idx],
            "step": int(self._steps[idx]),
            "env_id": int(self._env_ids[idx]),
            "agent_id": int(self._agent_ids[idx]),
        }


def compute_features(records):
    """Extract activations, interpretable features, and metadata from records.

    If records is a _PrecomputedRecordList (from the fast loader), this returns
    the precomputed arrays directly -- essentially a no-op.
    Falls back to the original per-record extraction for plain lists.
    """
    if isinstance(records, _PrecomputedRecordList):
        N = len(records)
        metadata = {
            "step": records._steps.astype(int),
            "env_id": records._env_ids.astype(int),
            "agent_id": records._agent_ids.astype(int),
        }
        return records._activations.copy(), records._features.copy(), metadata

    # Fallback: original slow path for plain list of dicts
    N = len(records)
    activations = np.zeros((N, len(records[0]["activation"])), dtype=np.float32)
    features = np.zeros((N, len(FEATURE_NAMES)), dtype=np.float32)
    metadata = {"step": np.zeros(N, dtype=int), "env_id": np.zeros(N, dtype=int),
                "agent_id": np.zeros(N, dtype=int)}

    for i, r in enumerate(tqdm(records, desc="Computing features", unit="rec")):
        activations[i] = r["activation"]
        metadata["step"][i] = r["step"]
        metadata["env_id"][i] = r["env_id"]
        metadata["agent_id"][i] = r["agent_id"]

        obs = r["observation"]
        action = r["action"]
        entity = np.array(obs["Entity"], dtype=np.float32)
        inventory = np.array(obs["Inventory"], dtype=np.float32)
        tile = np.array(obs["Tile"], dtype=np.float32)
        current_tick = np.array(obs["CurrentTick"], dtype=np.float32).item()
        agent_id = r["agent_id"]

        # Visible entities: rows where id != 0
        visible_mask = entity[:, ENT_ID] != 0
        n_visible = visible_mask.sum()

        # Split NPCs vs players among visible entities
        visible_ents = entity[visible_mask]
        npc_mask = visible_ents[:, ENT_NPC_TYPE] > 0
        player_mask = visible_ents[:, ENT_NPC_TYPE] == 0

        # Find self row
        self_mask = entity[:, ENT_ID] == agent_id
        if self_mask.any():
            self_ent = entity[self_mask][0]
            self_health = self_ent[ENT_HEALTH]
            self_food = self_ent[ENT_FOOD]
            self_water = self_ent[ENT_WATER]
            self_gold = self_ent[ENT_GOLD]
            melee_lvl = self_ent[ENT_MELEE_LVL]
            range_lvl = self_ent[ENT_RANGE_LVL]
            mage_lvl = self_ent[ENT_MAGE_LVL]
            max_combat = max(melee_lvl, range_lvl, mage_lvl)
            in_combat = (self_ent[ENT_ATTACKER_ID] != 0 or
                         (current_tick - self_ent[ENT_LATEST_COMBAT_TICK]) < 10)
            self_damage = self_ent[ENT_DAMAGE]
            self_r = self_ent[ENT_ROW]
            self_c = self_ent[ENT_COL]
            item_level = self_ent[ENT_ITEM_LEVEL]
            time_alive = self_ent[ENT_TIME_ALIVE]
            fishing_lvl = self_ent[ENT_FISHING_LVL]
            herbalism_lvl = self_ent[ENT_HERBALISM_LVL]
            prospecting_lvl = self_ent[ENT_PROSPECTING_LVL]
            carving_lvl = self_ent[ENT_CARVING_LVL]
            alchemy_lvl = self_ent[ENT_ALCHEMY_LVL]
            max_harvest = max(fishing_lvl, herbalism_lvl, prospecting_lvl,
                              carving_lvl, alchemy_lvl)
            # Don't count self as a visible player
            n_players = player_mask.sum() - 1

            # Nearest entity/player distances (Chebyshev)
            if n_visible > 1:  # >1 because self is visible
                others = visible_ents[visible_ents[:, ENT_ID] != agent_id]
                if len(others) > 0:
                    dists = np.maximum(np.abs(others[:, ENT_ROW] - self_r),
                                       np.abs(others[:, ENT_COL] - self_c))
                    nearest_entity_dist = dists.min()
                    player_others = others[others[:, ENT_NPC_TYPE] == 0]
                    nearest_player_dist = (
                        np.maximum(np.abs(player_others[:, ENT_ROW] - self_r),
                                   np.abs(player_others[:, ENT_COL] - self_c)).min()
                        if len(player_others) > 0 else 99.0
                    )
                else:
                    nearest_entity_dist = 99.0
                    nearest_player_dist = 99.0
            else:
                nearest_entity_dist = 99.0
                nearest_player_dist = 99.0
        else:
            self_health = self_food = self_water = self_gold = 0
            melee_lvl = range_lvl = mage_lvl = max_combat = 0
            in_combat = False
            self_damage = 0
            self_r = self_c = 0
            item_level = time_alive = 0
            fishing_lvl = herbalism_lvl = prospecting_lvl = 0
            carving_lvl = alchemy_lvl = max_harvest = 0
            n_players = player_mask.sum()
            nearest_entity_dist = 99.0
            nearest_player_dist = 99.0

        # Inventory features
        inv_mask = inventory[:, ITEM_ID] != 0
        n_inv = inv_mask.sum()
        n_equipped = (inventory[inv_mask, ITEM_EQUIPPED] > 0).sum() if n_inv > 0 else 0
        n_listed = (inventory[inv_mask, ITEM_LISTED_PRICE] > 0).sum() if n_inv > 0 else 0

        # Tile features: count terrain types in visible area
        tile_mats = tile[:, TILE_MATERIAL]
        n_water_tiles = ((tile_mats == MAT_WATER) | (tile_mats == MAT_OCEAN)).sum()
        n_forest_tiles = ((tile_mats == MAT_FOILAGE) | (tile_mats == MAT_TREE)).sum()

        # Action features
        is_moving = action[ACT_MOVE_DIR] != ACT_MOVE_NOOP
        is_attacking = action[ACT_ATTACK_TARGET] != ACT_ATTACK_NOOP
        is_trading = (action[ACT_BUY_ITEM] != ACT_BUY_NOOP or
                      action[ACT_SELL_ITEM] != ACT_SELL_NOOP or
                      action[ACT_GIVE_ITEM] != ACT_GIVE_NOOP)
        is_using_item = action[ACT_USE_ITEM] != ACT_USE_NOOP

        features[i] = [
            # Nearby entities
            n_visible, npc_mask.sum(), max(0, n_players),
            nearest_entity_dist, nearest_player_dist,
            # Self vitals
            self_health, self_food, self_water, self_gold,
            # Combat
            max_combat, melee_lvl, range_lvl, mage_lvl,
            float(in_combat), self_damage,
            # Harvest/profession skills
            fishing_lvl, herbalism_lvl, prospecting_lvl,
            carving_lvl, alchemy_lvl, max_harvest,
            # Equipment & inventory
            item_level, n_inv, n_equipped, n_listed,
            # Spatial
            self_r, self_c,
            # Terrain
            n_water_tiles, n_forest_tiles,
            # Time
            current_tick, time_alive,
            # Actions
            float(is_moving), float(is_attacking),
            float(is_trading), float(is_using_item),
        ]

    return activations, features, metadata


def run_umap(activations, n_neighbors=15, min_dist=0.1, n_components=2,
             random_state=42):
    """Reduce activations with UMAP.

    Returns (embedding, reducer) so the fitted UMAP model can be reused for
    projecting new points via reducer.transform().
    """
    import io
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=n_components, random_state=random_state,
                        verbose=True)
    # UMAP verbose=True writes progress bars to stderr (good) but also prints
    # epoch logs to stdout that arrive out of order. Suppress the stdout spam
    # and the n_jobs UserWarning.
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            embedding = reducer.fit_transform(activations)
    finally:
        sys.stdout = old_stdout
    return embedding, reducer


def run_hdbscan(embedding, min_cluster_size=15, min_samples=5):
    """Cluster UMAP embedding with HDBSCAN.

    Returns (labels, clusterer) so the fitted model can be reused for
    predicting cluster membership of new points via approximate_predict().
    """
    import hdbscan

    with spinner(f"Running HDBSCAN (min_cluster_size={min_cluster_size})"):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     min_samples=min_samples,
                                     prediction_data=True)
        labels = clusterer.fit_predict(embedding)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points "
          f"({n_noise / len(labels) * 100:.1f}%)")
    return labels, clusterer


def normalized_entropy(counts):
    """Compute normalized entropy (0-1) from an array of counts."""
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    ent = -np.sum(probs * np.log(probs))
    max_ent = np.log(len(counts))
    return ent / max_ent if max_ent > 0 else 0.0


def compute_cluster_stats(features, labels, metadata, n_baseline_samples=100,
                          rng_seed=42):
    """Compute per-cluster feature stats, Cohen's d vs random baseline, and confounders."""
    rng = np.random.RandomState(rng_seed)
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    N = len(features)
    n_features = features.shape[1]

    # Precompute step bins for entropy
    step_bins = np.digitize(metadata["step"],
                            np.linspace(metadata["step"].min(),
                                        metadata["step"].max() + 1, 21))

    stats = {}
    for label in tqdm(unique_labels, desc="Computing cluster stats", unit="cluster"):
        mask = labels == label
        K = mask.sum()
        cluster_features = features[mask]
        cluster_means = cluster_features.mean(axis=0)

        # Random contiguous window baseline
        baseline_means = np.zeros((n_baseline_samples, n_features))
        for b in range(n_baseline_samples):
            start = rng.randint(0, max(1, N - K))
            end = min(start + K, N)
            baseline_means[b] = features[start:end].mean(axis=0)

        baseline_mean_of_means = baseline_means.mean(axis=0)
        baseline_std = baseline_means.std(axis=0)

        # Cohen's d: (cluster_mean - baseline_mean) / pooled_sd
        # Use cluster std and baseline std for pooled estimate
        cluster_std = cluster_features.std(axis=0)
        pooled_sd = np.sqrt((cluster_std ** 2 + baseline_std ** 2) / 2)
        cohens_d = np.where(pooled_sd > 1e-8,
                            (cluster_means - baseline_mean_of_means) / pooled_sd,
                            0.0)

        # Confounder: step entropy
        cluster_step_bins = step_bins[mask]
        step_counts = np.bincount(cluster_step_bins, minlength=21)[1:]  # skip bin 0
        step_ent = normalized_entropy(step_counts)

        # Confounder: agent entropy
        cluster_agents = metadata["agent_id"][mask]
        agent_counts = np.bincount(cluster_agents)
        agent_ent = normalized_entropy(agent_counts)

        # Top features by |Cohen's d|
        top_idx = np.argsort(np.abs(cohens_d))[::-1]

        stats[label] = {
            "size": int(K),
            "means": cluster_means,
            "cohens_d": cohens_d,
            "top_features": top_idx,
            "step_entropy": step_ent,
            "agent_entropy": agent_ent,
        }

    return stats


def compute_rolling_features(features, metadata, window=5):
    """Compute per-trajectory rolling averages of features.

    Groups points by (env_id, agent_id), sorts by step within each group,
    and applies an edge-padded rolling mean. Window is in subsampled steps.
    Returns an array of the same shape as features with smoothed values.
    """
    rolling = np.copy(features)
    # Group indices by trajectory
    traj_indices = defaultdict(list)
    for i in range(len(features)):
        key = (int(metadata["env_id"][i]), int(metadata["agent_id"][i]))
        traj_indices[key].append(i)

    kernel = np.ones(window) / window
    for key, indices in traj_indices.items():
        # Sort by step within trajectory
        indices = sorted(indices, key=lambda i: metadata["step"][i])
        traj_feats = features[indices]  # (T, n_features)
        for f in range(traj_feats.shape[1]):
            vals = traj_feats[:, f]
            if len(vals) < window:
                # Too short for rolling, just keep raw
                continue
            padded = np.pad(vals, (window // 2, window - 1 - window // 2), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
            for j, idx in enumerate(indices):
                rolling[idx, f] = smoothed[j]

    return rolling


def generate_metric_umap(embedding, rolling_features, output_dir):
    """Generate a grid of UMAP scatter plots colored by rolling-average feature values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_features = len(FEATURE_NAMES)
    ncols = 2
    nrows = (n_features + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        vals = rolling_features[:, i]
        # Normalize to [0, 1] for alpha mapping
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            normed = (vals - vmin) / (vmax - vmin)
        else:
            normed = np.zeros_like(vals)
        alphas = 0.05 + 0.7 * normed  # range [0.05, 0.75]
        # Interpolate color from yellow (low) to deep orange-brown (high)
        colors = np.zeros((len(vals), 4))
        colors[:, 0] = 1.0 * (1 - normed) + 0.55 * normed   # R: 1.0 → 0.55
        colors[:, 1] = 0.9 * (1 - normed) + 0.25 * normed   # G: 0.9 → 0.25
        colors[:, 2] = 0.2 * (1 - normed) + 0.0 * normed    # B: 0.2 → 0.0
        colors[:, 3] = alphas
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   c=colors, s=1, rasterized=True)
        ax.set_title(FEATURE_NAMES[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # Opacity legend: a few reference alpha levels
        from matplotlib.lines import Line2D
        for level, label in [(0.0, f"{vmin:.1f}"), (0.5, f"{(vmin+vmax)/2:.1f}"), (1.0, f"{vmax:.1f}")]:
            a = 0.05 + 0.7 * level
            r = 1.0 * (1 - level) + 0.55 * level
            g = 0.9 * (1 - level) + 0.25 * level
            b = 0.2 * (1 - level) + 0.0 * level
            ax.plot([], [], 'o', color=(r, g, b, a), markersize=5, label=label)
        ax.legend(loc="upper right", fontsize=6, framealpha=0.5, handletextpad=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("UMAP Colored by Rolling-Average Feature Values", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "umap_metric_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def generate_feature_scatter(rolling_features, output_dir):
    """Generate a triangular scatter matrix of all feature pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_features = len(FEATURE_NAMES)
    # Lower triangle pairs
    pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    n_pairs = len(pairs)
    ncols = 7
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        ax.scatter(rolling_features[:, i], rolling_features[:, j],
                   s=0.5, alpha=0.08, c="#8B4000", rasterized=True)
        ax.set_xlabel(FEATURE_NAMES[i], fontsize=6)
        ax.set_ylabel(FEATURE_NAMES[j], fontsize=6)
        ax.tick_params(labelsize=5)

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Pair Scatter (Rolling Averages)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "feature_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def generate_pca_pairs(activations_reduced, output_dir):
    """Generate scatter plots of consecutive PCA direction pairs (1&2, 3&4, ...)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n_dims = activations_reduced.shape[1]
    n_pairs = n_dims // 2
    ncols = min(4, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(n_pairs):
        ax = axes[idx]
        d1, d2 = idx * 2, idx * 2 + 1
        # Compute point density for alpha via 2D histogram
        x, y = activations_reduced[:, d1], activations_reduced[:, d2]
        ax.scatter(x, y, s=0.5, alpha=0.08, c="#8B4000", rasterized=True)
        ax.set_xlabel(f"PC {d1 + 1}", fontsize=8)
        ax.set_ylabel(f"PC {d2 + 1}", fontsize=8)
        ax.tick_params(labelsize=6)

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("PCA Direction Pairs", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "pca_pairs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def generate_pca_feature_correlations(activations_reduced, rolling_features, pca,
                                      output_dir):
    """Compute and plot Pearson correlations between PCA component scores and features.

    Produces a heatmap showing which behavioral features vary along each PCA axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pcs = activations_reduced.shape[1]
    n_features = len(FEATURE_NAMES)

    # Pearson correlation matrix: (n_pcs, n_features)
    corr = np.zeros((n_pcs, n_features))
    for pc in range(n_pcs):
        for f in range(n_features):
            r = np.corrcoef(activations_reduced[:, pc], rolling_features[:, f])[0, 1]
            corr[pc, f] = r

    # Also compute R² per feature across all PCs (how much total PCA space captures each feature)
    # via multiple regression: R² = 1 - SS_res/SS_tot
    from numpy.linalg import lstsq
    r_squared = np.zeros(n_features)
    for f in range(n_features):
        y = rolling_features[:, f]
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot < 1e-10:
            continue
        # OLS: y = X @ beta
        coeffs, residuals, _, _ = lstsq(
            np.column_stack([activations_reduced, np.ones(len(y))]),
            y, rcond=None
        )
        y_pred = activations_reduced @ coeffs[:n_pcs] + coeffs[-1]
        ss_res = np.sum((y - y_pred) ** 2)
        r_squared[f] = 1 - ss_res / ss_tot

    # --- Heatmap ---
    # Show top 15 PCs max for readability
    n_show = min(n_pcs, 15)
    fig, (ax_heat, ax_r2) = plt.subplots(
        1, 2, figsize=(14, max(4, n_show * 0.45)),
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05}
    )

    # Explained variance labels
    evr = pca.explained_variance_ratio_
    pc_labels = [f"PC{i+1} ({evr[i]:.1%})" for i in range(n_show)]

    im = ax_heat.imshow(corr[:n_show], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax_heat.set_xticks(range(n_features))
    ax_heat.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks(range(n_show))
    ax_heat.set_yticklabels(pc_labels, fontsize=8)
    ax_heat.set_title("Pearson r: PC Score vs Rolling Feature", fontsize=11)

    # Annotate cells with |r| > 0.1
    for i in range(n_show):
        for j in range(n_features):
            val = corr[i, j]
            if abs(val) > 0.1:
                ax_heat.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(val) > 0.5 else "black")

    fig.colorbar(im, ax=ax_heat, label="Pearson r", shrink=0.8)

    # --- R² bar chart ---
    bars = ax_r2.barh(range(n_features), r_squared, color="#8B4000", alpha=0.7)
    ax_r2.set_yticks(range(n_features))
    ax_r2.set_yticklabels(FEATURE_NAMES, fontsize=7)
    ax_r2.set_xlabel("R² (all PCs)", fontsize=8)
    ax_r2.set_title("Total PCA\nExplained", fontsize=9)
    ax_r2.set_xlim(0, 1)
    ax_r2.invert_yaxis()
    for i, v in enumerate(r_squared):
        ax_r2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=6)

    fig.tight_layout()
    path = os.path.join(output_dir, "pca_feature_correlations.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")

    # Print a text summary of strong correlations
    print("\n  PCA-Feature Correlation Summary:")
    print(f"  {'Feature':25s} {'R²':>6s}  Top PC correlations")
    print(f"  {'-'*25} {'-'*6}  {'-'*40}")
    for f in range(n_features):
        top_pcs = np.argsort(np.abs(corr[:, f]))[::-1][:3]
        top_strs = [f"PC{pc+1}:{corr[pc,f]:+.2f}" for pc in top_pcs if abs(corr[pc, f]) > 0.05]
        print(f"  {FEATURE_NAMES[f]:25s} {r_squared[f]:6.3f}  {', '.join(top_strs)}")

    # --- Combined: scatter + binned profile for top 3 PCs, all features ---
    # Layout: each feature gets 2 rows (scatter, profile) x 3 cols (top PCs)
    n_bins = 20
    top_k = 3
    total_rows = n_features * 2
    fig2, axes2 = plt.subplots(total_rows, top_k,
                                figsize=(4.5 * top_k, 3.5 * n_features))

    for i in range(n_features):
        feat_vals = rolling_features[:, i]
        top_pcs_i = np.argsort(np.abs(corr[:, i]))[::-1][:top_k]
        row_scatter = i * 2
        row_profile = i * 2 + 1

        for rank, pc in enumerate(top_pcs_i):
            pc_vals = activations_reduced[:, pc]
            r_val = corr[pc, i]
            ax_scatter = axes2[row_scatter, rank]
            ax_profile = axes2[row_profile, rank]

            # --- Scatter row ---
            vmin, vmax = feat_vals.min(), feat_vals.max()
            if vmax > vmin:
                normed = (feat_vals - vmin) / (vmax - vmin)
            else:
                normed = np.zeros_like(feat_vals)
            alphas = 0.05 + 0.5 * normed
            sc_colors = np.zeros((len(feat_vals), 4))
            sc_colors[:, 0] = 1.0 * (1 - normed) + 0.55 * normed
            sc_colors[:, 1] = 0.9 * (1 - normed) + 0.25 * normed
            sc_colors[:, 2] = 0.2 * (1 - normed) + 0.0 * normed
            sc_colors[:, 3] = alphas

            ax_scatter.scatter(pc_vals, feat_vals, c=sc_colors, s=1, rasterized=True)
            ax_scatter.set_title(
                f"{FEATURE_NAMES[i]} vs PC{pc+1} (r={r_val:+.2f}, r²={r_val**2:.2f})",
                fontsize=8)
            ax_scatter.set_ylabel(FEATURE_NAMES[i], fontsize=7)
            ax_scatter.tick_params(labelsize=5)

            # --- Profile row ---
            bin_edges = np.percentile(pc_vals, np.linspace(0, 100, n_bins + 1))
            bin_centers = []
            bin_means = []
            bin_sems = []
            for b in range(n_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                if b == n_bins - 1:
                    mask = (pc_vals >= lo) & (pc_vals <= hi)
                else:
                    mask = (pc_vals >= lo) & (pc_vals < hi)
                if mask.sum() < 2:
                    continue
                bin_centers.append((lo + hi) / 2)
                vals_in_bin = feat_vals[mask]
                bin_means.append(vals_in_bin.mean())
                bin_sems.append(vals_in_bin.std() / np.sqrt(mask.sum()))

            bin_centers = np.array(bin_centers)
            bin_means = np.array(bin_means)
            bin_sems = np.array(bin_sems)

            ax_profile.plot(bin_centers, bin_means, color="#8B4000", linewidth=2)
            ax_profile.fill_between(bin_centers, bin_means - bin_sems,
                                    bin_means + bin_sems, color="#8B4000", alpha=0.2)
            ax_profile.set_xlabel(f"PC{pc+1} score", fontsize=7)
            ax_profile.set_ylabel(f"mean {FEATURE_NAMES[i]}", fontsize=7)
            ax_profile.tick_params(labelsize=5)

            # Sync axes
            xlim = ax_scatter.get_xlim()
            ax_profile.set_xlim(xlim)
            ylim = ax_scatter.get_ylim()
            ax_profile.set_ylim(ylim)

    fig2.suptitle("PCA Feature Profiles: Scatter (top) + Binned Mean ±SEM (bottom) per Feature",
                  fontsize=14, y=1.005)
    fig2.tight_layout()
    path2 = os.path.join(output_dir, "pca_feature_profiles.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Wrote {path2}")

    return corr, r_squared


def generate_umap_pairs(activations, n_neighbors, seed, output_dir, n_components=10):
    """Run a 10D UMAP on raw activations and plot consecutive dimension pairs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"Running {n_components}D UMAP for pairs plot...", flush=True)
    embedding_hd, _ = run_umap(activations, n_neighbors=n_neighbors,
                               n_components=n_components, random_state=seed)

    n_pairs = n_components // 2
    ncols = min(4, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(n_pairs):
        ax = axes[idx]
        d1, d2 = idx * 2, idx * 2 + 1
        ax.scatter(embedding_hd[:, d1], embedding_hd[:, d2],
                   s=0.5, alpha=0.08, c="#8B4000", rasterized=True)
        ax.set_xlabel(f"UMAP {d1 + 1}", fontsize=8)
        ax.set_ylabel(f"UMAP {d2 + 1}", fontsize=8)
        ax.tick_params(labelsize=6)

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"UMAP {n_components}D Direction Pairs", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "umap_pairs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")


def _write_summary(labels, stats, output_dir):
    """Write summary.txt with cluster stats."""
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("Activation Clustering Analysis\n")
        f.write("=" * 60 + "\n\n")

        n_clusters = len(stats)
        n_noise = (labels == -1).sum()
        f.write(f"Total points: {len(labels)}\n")
        f.write(f"Clusters: {n_clusters}\n")
        f.write(f"Noise points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)\n\n")

        for label in sorted(stats.keys()):
            s = stats[label]
            f.write(f"--- Cluster {label} (n={s['size']}) ---\n")
            f.write(f"  Step entropy:  {s['step_entropy']:.3f} "
                    f"{'(OK)' if s['step_entropy'] > 0.5 else '(LOW - temporal confound?)'}\n")
            f.write(f"  Agent entropy: {s['agent_entropy']:.3f} "
                    f"{'(OK)' if s['agent_entropy'] > 0.5 else '(LOW - agent confound?)'}\n")
            f.write(f"  Top features by |Cohen's d|:\n")
            for rank, idx in enumerate(s["top_features"][:5]):
                d = s["cohens_d"][idx]
                m = s["means"][idx]
                f.write(f"    {rank + 1}. {FEATURE_NAMES[idx]:25s}  "
                        f"mean={m:8.2f}  d={d:+.2f}\n")
            f.write("\n")
    print(f"  Wrote {summary_path}")


def _write_feature_heatmap(stats, output_dir):
    """Write cluster_features.png heatmap of Cohen's d."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not stats:
        print("No clusters found, skipping feature plot.")
        return

    cluster_ids = sorted(stats.keys())
    n_feats = len(FEATURE_NAMES)
    d_matrix = np.array([stats[c]["cohens_d"] for c in cluster_ids])

    fig, ax = plt.subplots(figsize=(max(12, n_feats * 0.8), max(4, len(cluster_ids) * 0.6)))
    im = ax.imshow(d_matrix, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels([f"C{c} (n={stats[c]['size']})" for c in cluster_ids], fontsize=8)
    ax.set_title("Cohen's d per Feature per Cluster")
    fig.colorbar(im, ax=ax, label="Cohen's d")

    for i in range(len(cluster_ids)):
        for j in range(n_feats):
            val = d_matrix[i, j]
            if abs(val) > 0.5:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 1.5 else "black")

    abs_output_dir = os.path.abspath(output_dir)
    ax.text(1.0, -0.02, abs_output_dir, ha="right", va="top",
            fontsize=5, color="gray", family="monospace", transform=ax.transAxes)
    feat_path = os.path.join(output_dir, "cluster_features.png")
    fig.savefig(feat_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {feat_path}")


def generate_outputs(embedding, labels, features, stats, output_dir):
    """Generate summary text, UMAP scatter, and feature heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    _write_summary(labels, stats, output_dir)

    # --- umap_scatter.png ---
    fig, ax = plt.subplots(figsize=(10, 8))
    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                   c="lightgray", s=1, alpha=0.3, label="noise")
    cluster_labels = sorted(set(labels) - {-1})
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(cluster_labels), 1))
    for i, label in enumerate(cluster_labels):
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap(i)], s=3, alpha=0.5, label=f"C{label}")
    ax.set_title("UMAP Embedding Colored by HDBSCAN Cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if len(cluster_labels) <= 20:
        ax.legend(markerscale=4, fontsize=8)
    abs_output_dir = os.path.abspath(output_dir)
    ax.text(1.0, -0.02, abs_output_dir, ha="right", va="top",
            fontsize=5, color="gray", family="monospace", transform=ax.transAxes)
    scatter_path = os.path.join(output_dir, "umap_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {scatter_path}")

    _write_feature_heatmap(stats, output_dir)



def generate_dendrogram_explorer(clusterer, embedding, output_dir, n_frames=75):
    """Generate interactive Plotly HTML for exploring HDBSCAN's condensed tree hierarchy.

    Precomputes cluster assignments at sampled lambda (1/distance) thresholds from
    the condensed tree and renders them as a Plotly slider over fixed UMAP coordinates.
    Low lambda = few broad clusters; high lambda = many fine-grained clusters + more noise.
    """
    import colorsys

    import plotly.graph_objects as go

    tree_df = clusterer.condensed_tree_.to_pandas()
    n_points = embedding.shape[0]

    # --- Build cluster hierarchy from condensed tree ---
    is_cluster_event = tree_df["child_size"] > 1
    cluster_tree = tree_df[is_cluster_event]
    point_tree = tree_df[~is_cluster_event]

    # parent -> [(child_cluster, birth_lambda)]
    cluster_children = defaultdict(list)
    cluster_birth = {}
    for _, row in cluster_tree.iterrows():
        parent, child = int(row["parent"]), int(row["child"])
        lam = row["lambda_val"]
        cluster_children[parent].append((child, lam))
        cluster_birth[child] = lam

    root = int(tree_df["parent"].min())
    cluster_birth[root] = 0.0

    # Point info: which cluster each point lives in and when it falls out
    point_home = {}
    point_lambda = {}
    for _, row in point_tree.iterrows():
        pid = int(row["child"])
        point_home[pid] = int(row["parent"])
        point_lambda[pid] = row["lambda_val"]

    # Cluster parent map (child -> parent) for ancestor walks
    cluster_parent_map = {}
    for _, row in cluster_tree.iterrows():
        cluster_parent_map[int(row["child"])] = int(row["parent"])

    def get_ancestors(cid):
        path = [cid]
        while cid in cluster_parent_map:
            cid = cluster_parent_map[cid]
            path.append(cid)
        return path

    ancestor_cache = {c: get_ancestors(c) for c in cluster_birth}

    # Pre-cache each point's ancestor chain (via its home cluster)
    point_ancestors = {}
    for pid in range(n_points):
        if pid in point_home:
            point_ancestors[pid] = ancestor_cache[point_home[pid]]

    # --- Sample lambda values across the condensed tree's range ---
    lam_min = tree_df["lambda_val"].min()
    lam_max = tree_df["lambda_val"].max()
    sampled_lambdas = np.linspace(lam_min, lam_max, n_frames)

    def get_active_clusters(lambda_cut):
        """Return the set of cluster IDs that are 'leaves' of the tree at this lambda."""
        active = set()

        def recurse(cid):
            born_children = [
                (c, l) for c, l in cluster_children.get(cid, []) if l <= lambda_cut
            ]
            if not born_children:
                active.add(cid)
            else:
                for c, _l in born_children:
                    recurse(c)

        recurse(root)
        return active

    # --- Precompute assignments at each sampled lambda ---
    with spinner("Precomputing dendrogram frames"):
        all_active_ids = set()
        frame_data = []

        for lam in sampled_lambdas:
            active = get_active_clusters(lam)
            all_active_ids.update(active)

            labels = np.full(n_points, -1, dtype=int)
            for pid in range(n_points):
                if pid not in point_ancestors:
                    continue
                if point_lambda[pid] <= lam:
                    continue  # point fell out → noise
                for anc in point_ancestors[pid]:
                    if anc in active:
                        labels[pid] = anc
                        break

            n_clusters = len(set(labels[labels >= 0]))
            n_noise = int((labels == -1).sum())
            frame_data.append((lam, labels, n_clusters, n_noise))

    # --- Assign stable colors to cluster IDs ---
    sorted_cids = sorted(all_active_ids)
    palette = {}
    for i, cid in enumerate(sorted_cids):
        hue = i / max(len(sorted_cids), 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.85)
        palette[cid] = (int(r * 255), int(g * 255), int(b * 255))

    def labels_to_colors(labels):
        colors = []
        for l in labels:
            if l == -1:
                colors.append("rgba(180,180,180,0.15)")
            else:
                r, g, b = palette.get(l, (128, 128, 128))
                colors.append(f"rgba({r},{g},{b},0.7)")
        return colors

    # --- Build Plotly figure with slider ---
    with spinner("Building Plotly HTML"):
        initial_colors = labels_to_colors(frame_data[0][1])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=embedding[:, 0].tolist(),
                y=embedding[:, 1].tolist(),
                mode="markers",
                marker=dict(color=initial_colors, size=3, line=dict(width=0)),
                hoverinfo="skip",
            )
        )

        steps = []
        for i, (lam, labels, nc, nn) in enumerate(frame_data):
            noise_pct = nn / n_points * 100
            colors = labels_to_colors(labels)
            steps.append(
                dict(
                    method="restyle",
                    args=[{"marker.color": [colors]}],
                    label=f"\u03bb={lam:.4f} \u2014 {nc} clusters, {noise_pct:.0f}% noise",
                )
            )

        fig.update_layout(
            sliders=[
                dict(
                    active=0,
                    steps=steps,
                    currentvalue=dict(visible=True, xanchor="center"),
                    pad=dict(t=40),
                    len=0.9,
                    x=0.05,
                )
            ],
            title=dict(
                text="Dendrogram Explorer \u2014 HDBSCAN Condensed Tree on UMAP",
                x=0.5,
            ),
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            width=950,
            height=700,
            template="plotly_white",
            showlegend=False,
        )

        html_path = os.path.join(output_dir, "dendrogram_explorer.html")
        fig.write_html(html_path, include_plotlyjs=True)

    print(f"  Wrote {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster activation vectors and analyze behavioral modes")
    parser.add_argument("data_dir", type=str,
                        help="Directory containing activation data (from extract_activations.py)")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Keep every Nth record per trajectory (default: 10, 1 = no subsampling)")
    parser.add_argument("--hdbscan-min-cluster", type=int, default=15,
                        help="HDBSCAN min_cluster_size (default: 15)")
    parser.add_argument("--hdbscan-min-samples", type=int, default=5,
                        help="HDBSCAN min_samples (default: 5)")
    parser.add_argument("--umap-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--cluster-before-umap", action="store_true",
                        help="Cluster in high-D activation space instead of 2D UMAP embedding. "
                             "Uses PCA to --pre-cluster-dims first, then HDBSCAN, then UMAP for viz only.")
    parser.add_argument("--pre-cluster-dims", type=int, default=30,
                        help="PCA dimensions before clustering when --cluster-before-umap (default: 30)")
    parser.add_argument("--metric-only", action="store_true",
                        help="Skip HDBSCAN clustering and only produce the metric UMAP plot.")
    parser.add_argument("--feature-scatter", action="store_true",
                        help="Generate scatter matrix of all feature pairs (rolling averages).")
    parser.add_argument("--umap-pairs", action="store_true",
                        help="Run a 10D UMAP on raw activations and plot consecutive dimension pairs.")
    parser.add_argument("--umap-pairs-dims", type=int, default=10,
                        help="Number of UMAP dimensions for --umap-pairs (default: 10)")
    parser.add_argument("--dendrogram-explorer", action="store_true",
                        help="Generate interactive Plotly HTML for scrubbing through HDBSCAN's "
                             "condensed tree hierarchy on the UMAP scatter. "
                             "Requires --cluster-before-umap.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-named under analysis_results/)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    records, total_datapoints = load_data(args.data_dir, subsample_rate=args.subsample)
    if not args.metric_only and len(records) < args.hdbscan_min_cluster:
        print(f"Only {len(records)} records after subsampling, need at least "
              f"{args.hdbscan_min_cluster}. Try --subsample 1.")
        return

    # Auto-name output dir: analysis_results/policy-N-subM-cK
    if args.output_dir is None:
        policy_name = data_dir.name
        dirname = (f"{policy_name}-{total_datapoints}"
                   f"-sub{args.subsample}-c{args.hdbscan_min_cluster}"
                   f"{f'-pca{args.pre_cluster_dims}' if args.cluster_before_umap else ''}")
        output_dir = os.path.join("analysis_results", dirname)
    else:
        output_dir = args.output_dir

    activations, features, metadata = compute_features(records)

    print(f"\nActivations: {activations.shape}, Features: {features.shape}", flush=True)

    # --- Run 2D UMAP (unless --metric-only) ---
    embedding = umap_reducer = None
    if not args.metric_only:
        print(f"Running UMAP on {activations.shape[0]} points...", flush=True)
        embedding, umap_reducer = run_umap(activations, n_neighbors=args.umap_neighbors,
                                           random_state=args.seed)

    # --- Run PCA if requested ---
    labels = hdbscan_clusterer = pca = activations_reduced = None
    if args.cluster_before_umap:
        from sklearn.decomposition import PCA

        pca_dims = min(args.pre_cluster_dims, activations.shape[1], activations.shape[0])
        print(f"Running PCA to {pca_dims} dims...", flush=True)
        pca = PCA(n_components=pca_dims, random_state=args.seed)
        activations_reduced = pca.fit_transform(activations)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA explains {explained:.1%} of variance")

    # --- Run HDBSCAN if not --metric-only ---
    if not args.metric_only:
        if args.cluster_before_umap:
            print(f"Running HDBSCAN on {pca_dims}-D PCA space...", flush=True)
            labels, hdbscan_clusterer = run_hdbscan(activations_reduced,
                                                    min_cluster_size=args.hdbscan_min_cluster,
                                                    min_samples=args.hdbscan_min_samples)
        else:
            labels, hdbscan_clusterer = run_hdbscan(embedding,
                                                    min_cluster_size=args.hdbscan_min_cluster,
                                                    min_samples=args.hdbscan_min_samples)

    # --- Compute rolling features ---
    print(f"\nComputing rolling feature averages...")
    rolling_features = compute_rolling_features(features, metadata)

    # --- Generate outputs ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nWriting outputs to {output_dir}/")
    imgs = []

    if labels is not None:
        stats = compute_cluster_stats(features, labels, metadata, rng_seed=args.seed)
        generate_outputs(embedding, labels, features, stats, output_dir)
        imgs += ["umap_scatter.png", "cluster_features.png"]

    if embedding is not None:
        generate_metric_umap(embedding, rolling_features, output_dir)
        imgs.append("umap_metric_scatter.png")

    if args.feature_scatter:
        generate_feature_scatter(rolling_features, output_dir)
        imgs.append("feature_scatter.png")

    if activations_reduced is not None:
        generate_pca_pairs(activations_reduced, output_dir)
        imgs.append("pca_pairs.png")
        generate_pca_feature_correlations(activations_reduced, rolling_features, pca,
                                          output_dir)
        imgs += ["pca_feature_correlations.png", "pca_feature_profiles.png"]

    if args.umap_pairs:
        generate_umap_pairs(activations, args.umap_neighbors, args.seed,
                            output_dir, n_components=args.umap_pairs_dims)
        imgs.append("umap_pairs.png")

    if args.dendrogram_explorer:
        if not args.cluster_before_umap:
            print("ERROR: --dendrogram-explorer requires --cluster-before-umap")
            sys.exit(1)
        generate_dendrogram_explorer(hdbscan_clusterer, embedding, output_dir)

    # --- Save models ---
    import joblib
    if umap_reducer is not None:
        joblib.dump(umap_reducer, os.path.join(output_dir, "umap_model.pkl"))
    if hdbscan_clusterer is not None:
        joblib.dump(hdbscan_clusterer, os.path.join(output_dir, "hdbscan_model.pkl"))
    if pca is not None:
        joblib.dump(pca, os.path.join(output_dir, "pca_model.pkl"))
    if labels is not None:
        label_records = np.column_stack([
            metadata["env_id"], metadata["agent_id"], metadata["step"], labels
        ])
        np.save(os.path.join(output_dir, "cluster_labels.npy"), label_records)
    saved = [f for f in ["umap_model.pkl", "hdbscan_model.pkl", "pca_model.pkl", "cluster_labels.npy"]
             if os.path.exists(os.path.join(output_dir, f))]
    if saved:
        print(f"  Wrote {', '.join(saved)}")

    # --- Display plots inline ---
    import subprocess
    for img in imgs:
        img_path = os.path.join(output_dir, img)
        try:
            result = subprocess.run(["imgcat", img_path], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  imgcat {img} failed (exit {result.returncode}): {result.stderr.strip()}")
        except FileNotFoundError:
            break

    print(f"\nDone.")


if __name__ == "__main__":
    main()
