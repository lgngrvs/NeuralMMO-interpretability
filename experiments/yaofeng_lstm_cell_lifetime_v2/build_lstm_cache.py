#!/usr/bin/env python
"""Build an LSTM-cell-layer cache from a multi-layer activation JSONL.

The existing `_build_cache_parallel` in scripts/analyze_activations.py pulls
the top-level `record["activation"]` (which is the first extracted layer, i.e.
action_decoder). For the v2 LSTM experiment we need the `lstm_cell` vector
from the same JSONL instead — same (env_id, agent_id, step) coverage, but
different 256-d vector per record.

Output: <data_dir>/lstm_cell.cache.npz with the same schema as the
analyze_activations.py cache:
    activations, features, steps, env_ids, agent_ids

The feature vector is derived from each record's observation + action exactly
as `_extract_features_from_record` does, so clustering + Cohen's d can reuse
the same downstream code.
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from analyze_activations import (  # noqa: E402
    _extract_features_from_record,
    _find_line_offsets,
)


LAYER = "lstm_cell"


def _process_chunk_lstm(args):
    """Parse a chunk of JSONL and pull the lstm_cell activation from each record.

    Mirrors `analyze_activations._process_chunk` but overrides the activation
    source.
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

            # Pull the right activation (lstm_cell) before feature extraction.
            layer_acts = r.get("activations", {})
            if LAYER not in layer_acts:
                continue
            r_override = dict(r)
            r_override["activation"] = layer_acts[LAYER]

            result = _extract_features_from_record(r_override)
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


def build_cache(jsonl_path, cache_path, n_workers=None):
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 64)

    print(f"Building LSTM cache with {n_workers} workers...")
    print(f"  source: {jsonl_path}")
    print(f"  target: {cache_path}")

    offsets = _find_line_offsets(str(jsonl_path), n_workers)
    chunk_args = [
        (str(jsonl_path), offsets[i], offsets[i + 1], i)
        for i in range(len(offsets) - 1)
    ]

    with multiprocessing.Pool(n_workers) as pool:
        results = list(pool.imap(_process_chunk_lstm, chunk_args))

    valid = [r for r in results if r is not None]
    if not valid:
        raise RuntimeError("No alive lstm_cell records found in JSONL")

    all_acts = np.concatenate([r[0] for r in valid])
    all_feats = np.concatenate([r[1] for r in valid])
    all_steps = np.concatenate([r[2] for r in valid])
    all_envs = np.concatenate([r[3] for r in valid])
    all_agents = np.concatenate([r[4] for r in valid])

    np.savez(
        cache_path,
        activations=all_acts,
        features=all_feats,
        steps=all_steps,
        env_ids=all_envs,
        agent_ids=all_agents,
    )
    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(
        f"  Wrote {len(all_acts)} alive records "
        f"({all_acts.shape[1]}-d) to cache ({size_mb:.0f} MB)"
    )
    return all_acts, all_feats, all_steps, all_envs, all_agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing the multilayer activations.jsonl "
        "(e.g. activation_data/yaofeng_200M_multilayer)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output cache .npz path. Default: <data_dir>/lstm_cell.cache.npz",
    )
    parser.add_argument("-n", "--n-workers", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache already exists.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    jsonl_files = list(data_dir.rglob("activations.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No activations.jsonl under {data_dir}")
    jsonl_path = jsonl_files[0]

    cache_path = Path(args.output) if args.output else jsonl_path.parent / "lstm_cell.cache.npz"

    if cache_path.exists() and not args.force:
        print(f"Cache already exists: {cache_path}. Use --force to rebuild.")
        return

    build_cache(jsonl_path, cache_path, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
