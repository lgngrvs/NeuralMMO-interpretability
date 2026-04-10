"""Extract activations from policy models during evaluation.

Registers forward hooks on each policy's action decoder to capture the hidden
state (input to action heads) every timestep. Saves per-step records with
agent IDs, env IDs, activations, observations, and actions to JSON files.

Each record is tagged with the policy name that produced it.
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
import gc

import nmmo
import nmmo.core.config as nc
import nmmo.core.game_api

import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.frameworks.cleanrl
import pufferlib.policy_pool as pp
import pufferlib.vectorization
import torch
from pufferlib.emulation import unpack_batched_obs

import agent_zoo.neurips23_start_kit as default_learner
from evaluate import EvalRunner, EVAL_TASK_FILE
from reinforcement_learning import clean_pufferl
from train import get_init_args

POLICIES_DIR = "policies"

# Policy checkpoints were saved as full model pickles (not state_dicts),
# so we need weights_only=False for torch.load. Patch the policy store
# to use this setting since it doesn't expose the kwarg.
import pufferlib.policy_store

_original_get_policy = pufferlib.policy_store.PolicyStore.get_policy


def _patched_get_policy(self, name):
    path = os.path.join(self.path, name + ".pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.load(path, map_location=torch.device(device), weights_only=False)


pufferlib.policy_store.PolicyStore.get_policy = _patched_get_policy

SMOKE_NUM_AGENTS = 2
SMOKE_HORIZON = 10
SMOKE_NUM_ENVS = 1
SMOKE_BATCH_SIZE = SMOKE_NUM_AGENTS * SMOKE_HORIZON


class SmokeTestConfig(
    nc.Medium,
    nc.Terrain,
    nc.Resource,
    nc.Combat,
    nc.NPC,
    nc.Progression,
    nc.Item,
    nc.Equipment,
    nc.Profession,
    nc.Exchange,
):
    def __init__(self, task_file):
        super().__init__()
        self.set("GAME_PACKS", [(nmmo.core.game_api.AgentTraining, 1)])
        self.set("CURRICULUM_FILE_PATH", task_file)
        self.set("TASK_EMBED_DIM", 2048)
        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("PLAYER_N", SMOKE_NUM_AGENTS)
        self.set("HORIZON", SMOKE_HORIZON)
        self.set("PLAYER_DEATH_FOG", None)
        self.set("NPC_N", 0)
        self.set("RESOURCE_RESILIENT_POPULATION", 0)
        self.set("COMBAT_SPAWN_IMMUNITY", 20)
        self.set("TERRAIN_FLIP_SEED", True)
        self.set("MAP_CENTER", 128)
        self.set("MAP_FORCE_GENERATION", False)
        self.set("MAP_GENERATE_PREVIEWS", False)
        self.set("MAP_N", 1)
        self.set("PATH_MAPS", "maps/pve_eval/")


def make_agent_creator_for_device(device):
    """Create an agent creator that targets a specific device."""
    policy_args = get_init_args(default_learner.Policy.__init__)
    recurrent_args = get_init_args(default_learner.Recurrent.__init__)

    def agent_creator(env, args=None):
        policy = default_learner.Policy(env, **policy_args)
        policy = default_learner.Recurrent(env, policy, **recurrent_args)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
        return policy.to(device)

    return agent_creator


def setup_smoke_test(policy_store_dir, task_file, seed):
    """Set up a minimal evaluation for smoke testing."""
    num_policies = len(pp.get_policy_names(policy_store_dir))
    assert num_policies > 0, "No policies found"
    pool_kernel = pp.create_kernel(SMOKE_NUM_AGENTS, num_policies, shuffle_with_seed=seed)

    def env_creator(*args, **kwargs):
        env = nmmo.Env(SmokeTestConfig(task_file))
        env = default_learner.RewardWrapper(env, eval_mode=True, early_stop_agent_num=0)
        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    config = pufferlib.namespace(
        device="cpu",
        num_envs=SMOKE_NUM_ENVS,
        batch_size=SMOKE_BATCH_SIZE,
        envs_per_batch=SMOKE_NUM_ENVS,
        envs_per_worker=1,
        env_pool=False,
        torch_deterministic=True,
        total_timesteps=100_000_000,
        learning_rate=1e-4,
        compile=False,
        verbose=True,
        seed=seed,
        data_dir=policy_store_dir,
        pool_kernel=pool_kernel,
    )

    return clean_pufferl.create(
        config=config,
        agent_creator=make_agent_creator_for_device("cpu"),
        env_creator=env_creator,
        vectorization=pufferlib.vectorization.Serial,
        eval_mode=True,
        eval_model_path=policy_store_dir,
        policy_selector=pp.AllPolicySelector(seed),
    )


def get_inner_policy(agent):
    """Navigate wrapper layers to get the base policy with action_decoder.

    Structure: RecurrentPolicy -> .policy (RecurrentWrapper) -> .policy (Baseline/etc)
    """
    policy = agent
    while hasattr(policy, "policy"):
        policy = policy.policy
    return policy


def register_hooks(data):
    """Register activation capture hooks on all policies in the pool.

    Returns (activation_buffers, hook_handles) where activation_buffers is a
    dict mapping policy_id -> list that the hooks append to, and hook_handles
    is a list of all handles for later removal.
    """
    pool = data.policy_pool
    activation_buffers = {}
    hook_handles = []

    for policy_id in pool.policy_ids:
        if policy_id == pp.LEARNER_POLICY_ID:
            agent = pool.learner_policy
            name = "learner"
        else:
            agent = pool.current_policies[policy_id]["policy"]
            name = pool.current_policies[policy_id]["name"]

        inner = get_inner_policy(agent)
        buf = []
        activation_buffers[policy_id] = {"name": name, "buffer": buf}

        def make_hook(buffer):
            def hook(module, args, output):
                buffer.append(args[0].detach().cpu())
            return hook

        handle = inner.action_decoder.register_forward_hook(make_hook(buf))
        hook_handles.append(handle)

    return activation_buffers, hook_handles


def build_policy_index_map(data):
    """Build a mapping from batch index -> (policy_id, policy_name).

    Uses the policy pool's sample_idxs which map policy_id -> list of batch indices.
    """
    pool = data.policy_pool
    index_to_policy = {}
    for policy_id in pool.policy_ids:
        if policy_id == pp.LEARNER_POLICY_ID:
            name = "learner"
        else:
            name = pool.current_policies[policy_id]["name"]
        for idx in pool.sample_idxs[policy_id]:
            index_to_policy[idx] = {"policy_id": policy_id, "policy_name": name}
    return index_to_policy


def unpack_observations(flat_obs, unflatten_context):
    """Unpack flat observations into structured fields.

    Returns the full env_outputs dict with tensors on CPU.
    """
    return unpack_batched_obs(flat_obs, unflatten_context)


def extract_agent_ids(env_outputs):
    """Extract agent IDs from unpacked observations."""
    return env_outputs["AgentId"][:, 0].cpu().numpy().astype(int)


# Raw observation fields to save (skip Task embedding and ActionTargets)
OBS_FIELDS = ["Tile", "Entity", "Inventory", "Market", "CurrentTick"]


def build_observation_record(env_outputs, idx):
    """Build a dict of raw observation arrays for a single agent."""
    obs = {}
    for field in OBS_FIELDS:
        if field not in env_outputs:
            continue
        val = env_outputs[field][idx]
        if hasattr(val, "cpu"):
            val = val.cpu()
        obs[field] = val.numpy().tolist()
    return obs


def evaluate_and_extract(data, output_dir, num_eval_episode, max_steps=None,
                         streaming=False, flush_interval=5000):
    """Run evaluation while extracting activations, observations, and actions.

    Args:
        max_steps: If set, stop after this many env steps regardless of episodes completed.
                   Useful for smoke testing.
        streaming: If True, write records to JSONL incrementally to avoid OOM.
        flush_interval: How often to flush records to disk (only used if streaming=True).
    """
    config = data.config
    inner_policy = get_inner_policy(data.agent)
    unflatten_context = inner_policy.unflatten_context

    activation_buffers, hook_handles = register_hooks(data)
    index_to_policy = build_policy_index_map(data)

    # Build a mapping from batch index -> position within that policy's subset.
    # PolicyPool.forwards calls each policy on sample_idxs[policy_id], so the
    # hook captures activations in that subset order.
    pool = data.policy_pool
    index_to_subset_pos = {}
    for policy_id in pool.policy_ids:
        for pos, idx in enumerate(pool.sample_idxs[policy_id]):
            index_to_subset_pos[idx] = (policy_id, pos)

    os.makedirs(output_dir, exist_ok=True)

    data.policy_pool.mask[:] = 1

    # Streaming writer or in-memory records
    writer = None
    records = None
    if streaming:
        writer = StreamingRecordWriter(output_dir, flush_interval=flush_interval)
    else:
        records = {}

    cnt_episode = 0
    global_step = 0

    def should_stop():
        if max_steps is not None:
            return global_step >= max_steps
        return cnt_episode >= num_eval_episode

    while not should_stop():
        ptr = step = 0

        while True:
            step += 1
            if ptr == config.batch_size + 1:
                break
            if max_steps is not None and global_step >= max_steps:
                break

            o, r, d, t, i, env_id, mask = data.pool.recv()

            i = data.policy_pool.update_scores(i, "return")
            for ii, ee in zip(i["learner"], env_id):
                ii["env_id"] = ee

            with torch.no_grad():
                o_tensor = torch.as_tensor(o)
                d_tensor = torch.as_tensor(d).float().to(data.device).view(-1)

                next_lstm_state = data.next_lstm_state
                if next_lstm_state is not None:
                    next_lstm_state = (
                        next_lstm_state[0][:, env_id],
                        next_lstm_state[1][:, env_id],
                    )

                # Clear buffers before forward pass
                for info in activation_buffers.values():
                    info["buffer"].clear()

                actions, logprob, value, next_lstm_state = data.policy_pool.forwards(
                    o_tensor.to(data.device), next_lstm_state
                )

                if next_lstm_state is not None:
                    h, c = next_lstm_state
                    data.next_lstm_state[0][:, env_id] = h
                    data.next_lstm_state[1][:, env_id] = c

            actions_np = actions.cpu().numpy()
            env_outputs = unpack_observations(o_tensor.to(data.device), unflatten_context)
            agent_ids = extract_agent_ids(env_outputs)

            # Reconstruct per-policy activation tensors
            policy_activations = {}
            for policy_id, info in activation_buffers.items():
                if info["buffer"]:
                    policy_activations[policy_id] = (
                        torch.cat(info["buffer"], dim=0).numpy()
                    )

            # Record data for each alive agent
            for idx in range(len(env_id)):
                if not mask[idx]:
                    continue

                policy_id, subset_pos = index_to_subset_pos[idx]
                policy_name = index_to_policy[idx]["policy_name"]

                activation = None
                if policy_id in policy_activations:
                    activation = policy_activations[policy_id][subset_pos].tolist()

                if activation is None:
                    continue

                record = {
                    "step": global_step,
                    "env_id": int(env_id[idx]),
                    "agent_id": int(agent_ids[idx]),
                    "activation": activation,
                    "observation": build_observation_record(env_outputs, idx),
                    "action": actions_np[idx].tolist(),
                }

                if streaming:
                    writer.append(policy_name, record)
                else:
                    if policy_name not in records:
                        records[policy_name] = []
                    records[policy_name].append(record)

            # Count completed episodes
            for policy_name, policy_i in i.items():
                for agent_i in policy_i:
                    if isinstance(agent_i, dict) and agent_i.get("episode_done", False):
                        cnt_episode += 1

            learner_mask = torch.Tensor(mask * data.policy_pool.mask)
            indices = torch.where(learner_mask)[0][: config.batch_size - ptr + 1].numpy()
            ptr += len(indices)

            data.pool.send(actions_np)
            global_step += 1

        data.sort_keys = []
        if streaming:
            total_records = writer.total_records()
        else:
            total_records = sum(len(v) for v in records.values())
        print(f"Step {global_step}, {cnt_episode} episodes, {total_records} records collected.")

    for handle in hook_handles:
        handle.remove()

    if streaming:
        writer.close()
        writer.summary()
        return None  # data already on disk
    return records


class StreamingRecordWriter:
    """Write records to JSONL files incrementally to avoid OOM."""

    def __init__(self, output_dir, flush_interval=5000):
        self.output_dir = output_dir
        self.flush_interval = flush_interval
        self.buffers = {}  # policy_name -> list of records
        self.file_handles = {}  # policy_name -> open file handle
        self.counts = {}  # policy_name -> total records written

    def append(self, policy_name, record):
        if policy_name not in self.buffers:
            self.buffers[policy_name] = []
            self.counts[policy_name] = 0
        self.buffers[policy_name].append(record)
        if len(self.buffers[policy_name]) >= self.flush_interval:
            self.flush(policy_name)

    def _get_handle(self, policy_name):
        if policy_name not in self.file_handles:
            policy_dir = os.path.join(self.output_dir, policy_name)
            os.makedirs(policy_dir, exist_ok=True)
            filepath = os.path.join(policy_dir, "activations.jsonl")
            self.file_handles[policy_name] = open(filepath, "a")
        return self.file_handles[policy_name]

    def flush(self, policy_name):
        buf = self.buffers.get(policy_name, [])
        if not buf:
            return
        fh = self._get_handle(policy_name)
        for record in buf:
            fh.write(json.dumps(record) + "\n")
        fh.flush()
        self.counts[policy_name] = self.counts.get(policy_name, 0) + len(buf)
        self.buffers[policy_name] = []
        gc.collect()

    def flush_all(self):
        for policy_name in list(self.buffers.keys()):
            self.flush(policy_name)

    def close(self):
        self.flush_all()
        for fh in self.file_handles.values():
            fh.close()
        self.file_handles.clear()

    def total_records(self):
        return sum(self.counts.values()) + sum(len(b) for b in self.buffers.values())

    def summary(self):
        for name, count in self.counts.items():
            count += len(self.buffers.get(name, []))
            filepath = os.path.join(self.output_dir, name, "activations.jsonl")
            print(f"Saved {count} records to {filepath}")


def save_records(all_records, output_dir):
    """Save extracted records to JSON files, one per policy (legacy format)."""
    for policy_name, records in all_records.items():
        policy_dir = os.path.join(output_dir, policy_name)
        os.makedirs(policy_dir, exist_ok=True)
        filepath = os.path.join(policy_dir, "activations.json")
        with open(filepath, "w") as f:
            json.dump(records, f)
        print(f"Saved {len(records)} records to {filepath}")


def list_available_policies(policies_dir=POLICIES_DIR):
    """List all .pt policy files in the policies directory."""
    names = []
    for f in sorted(os.listdir(policies_dir)):
        if f.endswith(".pt"):
            names.append(f[:-3])
    return names


def prepare_policy_dir(selected_policies, policies_dir=POLICIES_DIR):
    """Create a temp directory with symlinks to only the selected policy files.

    The policy pool discovers policies by listing .pt files in a directory,
    so we symlink only the ones we want into a temp dir.
    Returns the path to the temp directory (caller must clean up).
    """
    tmp_dir = tempfile.mkdtemp(prefix="nmmo_extract_")
    for name in selected_policies:
        src = os.path.join(os.path.abspath(policies_dir), f"{name}.pt")
        if not os.path.exists(src):
            shutil.rmtree(tmp_dir)
            raise FileNotFoundError(f"Policy not found: {src}")
        os.symlink(src, os.path.join(tmp_dir, f"{name}.pt"))
    return tmp_dir


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Extract activations during evaluation")
    parser.add_argument("mode", type=str, nargs="?", choices=["pve", "pvp"], help="Evaluation mode")
    parser.add_argument(
        "-p",
        "--policies",
        type=str,
        required=False,
        help="Comma-separated list of policy names (without .pt). "
        "Use --list to see available policies.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="activation_data",
        help="Output directory for extracted data",
    )
    parser.add_argument(
        "--policies-dir",
        type=str,
        default=POLICIES_DIR,
        help="Directory containing policy .pt files",
    )
    parser.add_argument(
        "-t", "--task-file", type=str, default=EVAL_TASK_FILE, help="Path to the task file"
    )
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-n",
        "--num-episode",
        type=int,
        default=4,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available policies and exit"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal smoke test (2 agents, 1 env, 10 steps) to verify the pipeline works",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--streaming", action="store_true",
        help="Stream records to JSONL incrementally (avoids OOM for large runs)")
    parser.add_argument(
        "--flush-interval", type=int, default=5000,
        help="Records to buffer before flushing to disk (default: 5000, only with --streaming)")
    args = parser.parse_args()

    if args.list:
        for name in list_available_policies(args.policies_dir):
            print(name)
        return

    if not args.policies:
        parser.error("--policies is required (use --list to see available policies)")

    policies = [p.strip() for p in args.policies.split(",")]

    if args.smoke_test:
        policy_store_dir = prepare_policy_dir(policies, args.policies_dir)
        try:
            print(f"Smoke test: {SMOKE_NUM_AGENTS} agents, {SMOKE_NUM_ENVS} env, "
                  f"{SMOKE_HORIZON} steps")
            pufferl_data = setup_smoke_test(policy_store_dir, args.task_file, args.seed)

            records = evaluate_and_extract(
                pufferl_data, args.output_dir, num_eval_episode=0, max_steps=SMOKE_HORIZON
            )
            save_records(records, args.output_dir)

            total = sum(len(v) for v in records.values())
            print(f"Smoke test passed: {total} records extracted across "
                  f"{len(records)} policies.")
            clean_pufferl.close(pufferl_data)
        finally:
            shutil.rmtree(policy_store_dir)
        return

    if not args.mode:
        parser.error("mode is required (pve or pvp)")

    mode = args.mode
    policy_store_dir = prepare_policy_dir(policies, args.policies_dir)
    try:
        # Always use debug=True to force Serial vectorization.
        # Multiprocessing can't pickle the env creator closure, and Serial
        # is fine for extraction since we're not optimizing for throughput.
        runner = EvalRunner(policy_store_dir, debug=True)
        pufferl_data = runner.setup_evaluator(mode, args.task_file, args.seed)

        records = evaluate_and_extract(
            pufferl_data, args.output_dir, args.num_episode,
            streaming=args.streaming, flush_interval=args.flush_interval)
        if records is not None:
            save_records(records, args.output_dir)
        clean_pufferl.close(pufferl_data)
    finally:
        shutil.rmtree(policy_store_dir)


if __name__ == "__main__":
    main()
