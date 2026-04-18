#!/usr/bin/env python
"""Pre-build the caches for batch_8 and batch_9 in parallel.

Kicks off two cache builds at once. Each uses up to 64 CPU workers internally
(as hard-coded in analyze_activations._build_cache_parallel), so we limit to
~60 workers each to stay under the 160-CPU budget.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))


def build_one(json_path, n_workers):
    import numpy as np

    from analyze_activations import _build_cache_parallel, _cache_path_for_file

    json_file = Path(json_path)
    cache_file = _cache_path_for_file(json_file)

    # If cache is newer than source, skip
    if cache_file.exists():
        import os
        if os.path.getmtime(cache_file) > os.path.getmtime(json_file):
            print(f"[skip] {json_file} already has a current cache", flush=True)
            data = np.load(cache_file)
            return str(json_file), len(data["activations"])

    print(f"[build] {json_file} with {n_workers} workers", flush=True)
    t = time.time()
    acts, *_ = _build_cache_parallel(json_file, n_workers=n_workers)
    dt = time.time() - t
    print(f"[done] {json_file}: {len(acts)} alive records in {dt/60:.1f} min", flush=True)
    return str(json_file), len(acts)


def main():
    jobs = [
        ("/workspace/NeuralMMO-interpretability/activation_data/takeru_200M_batch_8/takeru_200M/activations.jsonl", 64),
        ("/workspace/NeuralMMO-interpretability/activation_data/takeru_200M_batch_9/takeru_200M/activations.jsonl", 64),
    ]
    # Build batch_9 first (smaller at 8.6G), then batch_8 (20G). Actually, run both
    # in parallel since each sub-parallelizes. With 64+64=128 workers, fits in budget.
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(build_one, p, n) for p, n in jobs]
        for fut in futures:
            print(fut.result(), flush=True)


if __name__ == "__main__":
    main()
