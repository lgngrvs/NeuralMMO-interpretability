#!/usr/bin/env python
"""Parallel activation extraction across multiple GPUs.

Shards activation extraction across N GPU processes, each running
scripts/extract_activations.py as a subprocess with a unique seed
and CUDA_VISIBLE_DEVICES assignment. After all shards finish, merges
their JSONL outputs into a single file with remapped env_ids to avoid
collisions.

Usage:
    uv run python scripts/extract_activations_parallel.py pve \
        -p yaofeng --layers action_decoder,lstm_cell \
        --num-shards 8 --gpus 0,1,2,3,4,5,6,7 \
        --base-seed 1000 --num-episode 40 \
        --streaming --flush-interval 5000 -o activation_data
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, "extract_activations.py")

# Offset added to env_id per shard to avoid collisions during merge.
ENV_ID_SHARD_OFFSET = 10_000_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel multi-GPU activation extraction wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required positional arg (evaluation mode)
    parser.add_argument("mode", type=str, choices=["pve", "pvp"], help="Evaluation mode")
    parser.add_argument(
        "-p", "--policies", type=str, required=True,
        help="Comma-separated list of policy names (without .pt)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="activation_data",
        help="Final output directory for merged data",
    )

    # Parallelization args
    parser.add_argument(
        "--num-shards", type=int, required=True,
        help="Number of parallel extraction processes",
    )
    parser.add_argument(
        "--gpus", type=str, required=True,
        help="Comma-separated GPU indices (e.g. 0,1,2,3). Must match --num-shards.",
    )
    parser.add_argument(
        "--base-seed", type=int, default=1000,
        help="Base seed; shard i uses base_seed + i (default: 1000)",
    )

    # Episode count (TOTAL across all shards)
    parser.add_argument(
        "-n", "--num-episode", type=int, default=4,
        help="TOTAL number of episodes (divided among shards)",
    )

    # Pass-through args for extract_activations.py
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layers to extract (passed through)")
    parser.add_argument("--layer", type=str, default=None,
                        help="Single layer to extract (passed through)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming JSONL output (passed through)")
    parser.add_argument("--flush-interval", type=int, default=5000,
                        help="Flush interval for streaming (passed through)")
    parser.add_argument("--policies-dir", type=str, default=None,
                        help="Directory containing policy .pt files (passed through)")
    parser.add_argument("-t", "--task-file", type=str, default=None,
                        help="Path to task file (passed through)")

    # Merge control
    parser.add_argument(
        "--keep-shards", action="store_true",
        help="Keep per-shard temporary directories after merge (default: delete)",
    )
    parser.add_argument(
        "--scratch-dir", type=str, default=None,
        help="Parent directory for shard temp dirs (default: system temp)",
    )

    return parser.parse_args()


def build_shard_cmd(args, shard_idx, shard_output_dir, episodes_per_shard):
    """Build the command list for a single shard subprocess."""
    seed = args.base_seed + shard_idx

    cmd = [
        sys.executable, EXTRACT_SCRIPT,
        args.mode,
        "-p", args.policies,
        "-o", shard_output_dir,
        "-s", str(seed),
        "-n", str(episodes_per_shard),
    ]

    # Pass through optional flags
    if args.layers is not None:
        cmd.extend(["--layers", args.layers])
    if args.layer is not None:
        cmd.extend(["--layer", args.layer])
    if args.streaming:
        cmd.append("--streaming")
    if args.flush_interval != 5000:
        cmd.extend(["--flush-interval", str(args.flush_interval)])
    if args.policies_dir is not None:
        cmd.extend(["--policies-dir", args.policies_dir])
    if args.task_file is not None:
        cmd.extend(["-t", args.task_file])

    return cmd


def determine_output_subdirs(args):
    """Determine what subdirectory names the extraction script will create.

    Returns (subdir_suffix, multilayer) to help find shard outputs during merge.
    """
    if args.layers is not None:
        layers_list = [s.strip() for s in args.layers.split(",") if s.strip()]
    elif args.layer is not None:
        layers_list = [args.layer]
    else:
        layers_list = ["action_decoder"]

    multilayer = len(layers_list) > 1
    subdir_suffix = "_multilayer" if multilayer else ""
    return subdir_suffix, multilayer


def find_shard_jsonl_files(shard_dir, policies, subdir_suffix):
    """Find all JSONL files produced by a shard, keyed by policy name."""
    results = {}
    for policy in policies:
        subdir_name = f"{policy}{subdir_suffix}"
        jsonl_path = os.path.join(shard_dir, subdir_name, "activations.jsonl")
        if os.path.exists(jsonl_path):
            results[policy] = jsonl_path
        else:
            # Check for JSON (non-streaming) output
            json_path = os.path.join(shard_dir, subdir_name, "activations.json")
            if os.path.exists(json_path):
                results[policy] = json_path
    return results


def merge_shard_outputs(shard_dirs, policies, subdir_suffix, output_dir, num_shards):
    """Merge per-shard JSONL files into the final output directory.

    Processes line-by-line to avoid loading into RAM. Remaps env_id to
    env_id + shard_idx * ENV_ID_SHARD_OFFSET to avoid collisions.
    """
    import orjson

    for policy in policies:
        out_subdir = f"{policy}{subdir_suffix}"
        out_policy_dir = os.path.join(output_dir, out_subdir)
        os.makedirs(out_policy_dir, exist_ok=True)
        out_path = os.path.join(out_policy_dir, "activations.jsonl")

        total_records = 0
        with open(out_path, "wb") as out_fh:
            for shard_idx, shard_dir in enumerate(shard_dirs):
                shard_files = find_shard_jsonl_files(shard_dir, [policy], subdir_suffix)
                if policy not in shard_files:
                    print(f"  Warning: no output for policy '{policy}' in shard {shard_idx}")
                    continue

                shard_path = shard_files[policy]
                is_json_array = shard_path.endswith(".json")

                if is_json_array:
                    # Handle non-streaming JSON array output
                    import json
                    with open(shard_path, "r") as fh:
                        records = json.load(fh)
                    for record in records:
                        record["env_id"] = record["env_id"] + shard_idx * ENV_ID_SHARD_OFFSET
                        out_fh.write(orjson.dumps(record) + b"\n")
                        total_records += 1
                else:
                    # Stream JSONL line-by-line
                    with open(shard_path, "rb") as in_fh:
                        for line in in_fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = orjson.loads(line)
                            except Exception:
                                continue
                            record["env_id"] = (
                                record["env_id"] + shard_idx * ENV_ID_SHARD_OFFSET
                            )
                            out_fh.write(orjson.dumps(record) + b"\n")
                            total_records += 1

        print(f"  Merged {total_records} records for '{policy}' -> {out_path}")


def main():
    args = parse_args()

    gpu_list = [g.strip() for g in args.gpus.split(",")]
    if len(gpu_list) != args.num_shards:
        print(
            f"Error: --num-shards={args.num_shards} but --gpus has "
            f"{len(gpu_list)} entries. They must match.",
            file=sys.stderr,
        )
        sys.exit(1)

    policies = [p.strip() for p in args.policies.split(",")]
    subdir_suffix, multilayer = determine_output_subdirs(args)
    episodes_per_shard = math.ceil(args.num_episode / args.num_shards)

    # Create scratch directory for shard outputs
    scratch_parent = args.scratch_dir or tempfile.gettempdir()
    scratch_dir = tempfile.mkdtemp(prefix="parallel_extract_", dir=scratch_parent)

    shard_dirs = []
    for i in range(args.num_shards):
        shard_dir = os.path.join(scratch_dir, f"shard_{i}")
        os.makedirs(shard_dir, exist_ok=True)
        shard_dirs.append(shard_dir)

    print("=" * 80)
    print("PARALLEL ACTIVATION EXTRACTION")
    print(f"  Shards:           {args.num_shards}")
    print(f"  GPUs:             {gpu_list}")
    print(f"  Total episodes:   {args.num_episode}")
    print(f"  Episodes/shard:   {episodes_per_shard}")
    print(f"  Base seed:        {args.base_seed}")
    print(f"  Policies:         {policies}")
    print(f"  Layers:           {args.layers or args.layer or 'action_decoder'}")
    print(f"  Streaming:        {args.streaming}")
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Scratch dir:      {scratch_dir}")
    print(f"  Keep shards:      {args.keep_shards}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Launch all shards
    # ------------------------------------------------------------------
    processes = []
    log_files = []
    t0 = time.time()

    for i in range(args.num_shards):
        cmd = build_shard_cmd(args, i, shard_dirs[i], episodes_per_shard)

        log_path = os.path.join(scratch_dir, f"shard_{i}.log")
        log_fh = open(log_path, "w")
        log_files.append((log_path, log_fh))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list[i]
        # Limit thread count to avoid contention
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("OPENBLAS_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")

        print(f"  Launching shard {i}: GPU={gpu_list[i]}, seed={args.base_seed + i}, "
              f"episodes={episodes_per_shard}")

        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=REPO_ROOT,
        )
        processes.append(proc)

    # ------------------------------------------------------------------
    # Wait for all shards (let all finish even if some fail)
    # ------------------------------------------------------------------
    print(f"\nWaiting for {args.num_shards} shards to complete...")

    exit_codes = []
    for i, proc in enumerate(processes):
        proc.wait()
        exit_codes.append(proc.returncode)
        log_files[i][1].close()

        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        elapsed_shard = time.time() - t0
        print(f"  Shard {i}: {status}  ({elapsed_shard:.0f}s elapsed)")

    t_shards = time.time() - t0

    # Check for failures
    failed = [i for i, rc in enumerate(exit_codes) if rc != 0]
    if failed:
        print(f"\n{'='*80}")
        print(f"ERROR: {len(failed)} shard(s) failed: {failed}")
        for i in failed:
            log_path = log_files[i][0]
            print(f"\n--- Tail of shard {i} log ({log_path}) ---")
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-30:]:
                        print(f"  {line}", end="")
            except Exception as e:
                print(f"  Could not read log: {e}")
        print(f"\n{'='*80}")

        if not args.keep_shards:
            print(f"Keeping scratch dir for debugging: {scratch_dir}")
        sys.exit(1)

    print(f"\nAll {args.num_shards} shards completed in {t_shards:.0f}s")

    # ------------------------------------------------------------------
    # Merge shard outputs
    # ------------------------------------------------------------------
    print(f"\nMerging shard outputs...")
    t_merge_start = time.time()

    os.makedirs(args.output_dir, exist_ok=True)
    merge_shard_outputs(shard_dirs, policies, subdir_suffix, args.output_dir, args.num_shards)

    t_merge = time.time() - t_merge_start
    print(f"Merge completed in {t_merge:.1f}s")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if args.keep_shards:
        print(f"\nShard directories preserved at: {scratch_dir}")
    else:
        shutil.rmtree(scratch_dir)
        print(f"\nCleaned up scratch directory: {scratch_dir}")

    total_elapsed = time.time() - t0
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
