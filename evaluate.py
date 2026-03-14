import os
import json
import random
import logging
import argparse
from collections import defaultdict

import nmmo
import nmmo.core.config as nc

import pufferlib
import pufferlib.policy_pool as pp

from reinforcement_learning import clean_pufferl
import agent_zoo.neurips23_start_kit as default_learner

from train import get_init_args

NUM_AGENTS = 128
EVAL_TASK_FILE = "neurips23_evaluation/heldout_task_with_embedding.pkl"
NUM_PVE_EVAL_EPISODE = 32
NUM_PVP_EVAL_EPISODE = 200  # TODO: cannot do more due to memory leak


def get_eval_config(debug=False):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": device,
        "num_envs": 6 if not debug else 1,
        "batch_size": 2**15 if not debug else 2**12,
    }


class EvalConfig(
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
    """NMMO config for NeurIPS 2023 competition evaluation.
    Hardcoded to keep the eval config independent from the training config.
    """

    def __init__(self, task_file, mode):
        super().__init__()
        self.set("GAME_PACKS", [(nmmo.core.game_api.AgentTraining, 1)])
        self.set("CURRICULUM_FILE_PATH", task_file)
        self.set("TASK_EMBED_DIM", 2048)  # must match the task file

        # Eval constants
        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("PLAYER_N", NUM_AGENTS)
        self.set("HORIZON", 1024)
        self.set("PLAYER_DEATH_FOG", None)
        self.set("NPC_N", 256)
        self.set("RESOURCE_RESILIENT_POPULATION", 0)
        self.set("COMBAT_SPAWN_IMMUNITY", 20)

        # Map related
        self.set("TERRAIN_FLIP_SEED", True)
        self.set("MAP_CENTER", 128)
        self.set("MAP_FORCE_GENERATION", False)
        self.set("MAP_GENERATE_PREVIEWS", True)
        if mode not in ["pve", "pvp"]:
            raise ValueError(f"Invalid eval_mode: {mode}")
        if mode == "pve":
            self.set("MAP_N", 4)
            self.set("PATH_MAPS", "maps/pve_eval/")
        else:
            self.set("MAP_N", 256)
            self.set("PATH_MAPS", "maps/pvp_eval/")


def make_env_creator(task_file, mode):
    def env_creator(*args, **kwargs):  # dummy args
        env = nmmo.Env(EvalConfig(task_file, mode))
        # Reward wrapper is for the learner, which is not used in evaluation
        env = default_learner.RewardWrapper(
            env,
            **{
                "eval_mode": True,
                "early_stop_agent_num": 0,
            },
        )
        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    return env_creator


def make_agent_creator():
    # NOTE: Assuming all policies are recurrent, which may not be true
    policy_args = get_init_args(default_learner.Policy.__init__)
    recurrent_args = get_init_args(default_learner.Recurrent.__init__)

    def agent_creator(env, args=None):
        policy = default_learner.Policy(env, **policy_args)
        policy = default_learner.Recurrent(env, policy, **recurrent_args)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
        return policy.to(get_eval_config()["device"])

    return agent_creator


class EvalRunner:
    def __init__(self, policy_store_dir, debug=False):
        self.policy_store_dir = policy_store_dir
        self._debug = debug

    def set_debug(self, debug):
        self._debug = debug

    def setup_evaluator(self, mode, task_file, seed):
        policies = pp.get_policy_names(self.policy_store_dir)
        assert len(policies) > 0, "No policies found in eval_model_path"
        if mode == "pve":
            assert len(policies) == 1, "PvE mode requires only one policy"
        logging.info(f"Policies to evaluate: {policies}")

        # pool_kernel determines policy-agent mapping
        pool_kernel = pp.create_kernel(NUM_AGENTS, len(policies), shuffle_with_seed=seed)

        config = self.get_pufferl_config(self._debug)
        config.seed = seed
        config.data_dir = self.policy_store_dir
        config.pool_kernel = pool_kernel

        vectorization = (
            pufferlib.vectorization.Serial
            if self._debug
            else pufferlib.vectorization.Multiprocessing
        )

        return clean_pufferl.create(
            config=config,
            agent_creator=make_agent_creator(),
            env_creator=make_env_creator(task_file, mode),
            vectorization=vectorization,
            eval_mode=True,
            eval_model_path=self.policy_store_dir,
            policy_selector=pp.AllPolicySelector(seed),
        )

    @staticmethod
    def get_pufferl_config(debug=False):
        config = get_eval_config(debug)
        # add required configs
        config["torch_deterministic"] = True
        config["total_timesteps"] = 100_000_000  # arbitrarily large, but will end much earlier
        config["envs_per_batch"] = config["num_envs"]
        config["envs_per_worker"] = 1
        config["env_pool"] = False
        config["learning_rate"] = 1e-4
        config["compile"] = False
        config["verbose"] = True  # not debug
        return pufferlib.namespace(**config)

    def perform_eval(self, mode, task_file, seed, num_eval_episode, save_file_prefix):
        pufferl_data = self.setup_evaluator(mode, task_file, seed)
        # this is a hack
        pufferl_data.policy_pool.mask[:] = 1  # policy_pool.mask is valid for training only

        eval_results = {}
        cnt_episode = 0
        while cnt_episode < num_eval_episode:
            _, infos = clean_pufferl.evaluate(pufferl_data)

            for pol, vals in infos.items():
                cnt_episode += sum(infos[pol]["episode_done"])
                if pol not in eval_results:
                    eval_results[pol] = defaultdict(list)
                for k, v in vals.items():
                    if k == "length":
                        eval_results[pol][k] += v  # length is a plain list
                    if k.startswith("curriculum"):
                        eval_results[pol][k] += [vv[0] for vv in v]

            pufferl_data.sort_keys = []  # TODO: check if this solves memory leak

            print(f"\nSeed: {seed}, evaluated {cnt_episode} episodes.\n")

        file_name = f"{save_file_prefix}_{seed}.json"
        self._save_results(eval_results, file_name)
        clean_pufferl.close(pufferl_data)
        return eval_results, file_name

    def _save_results(self, results, file_name):
        with open(os.path.join(self.policy_store_dir, file_name), "w") as f:
            json.dump(results, f)

    def run(
        self, mode, task_file=EVAL_TASK_FILE, seed=None, num_episode=None, save_file_prefix=None
    ):
        assert mode in ["pve", "pvp"], f"Invalid mode: {mode}"
        if mode == "pve":
            num_episode = num_episode or NUM_PVE_EVAL_EPISODE
            save_file_prefix = save_file_prefix or "eval_pve"
        else:
            num_episode = num_episode or NUM_PVP_EVAL_EPISODE
            save_file_prefix = save_file_prefix or "eval_pvp"

        if self._debug:
            num_episode = 4

        if seed is None:
            seed = random.randint(10000000, 99999999)

        logging.info(f"Evaluating {self.policy_store_dir} in the {mode} mode with seed: {seed}")
        logging.info(f"Using the task file: {task_file}")

        _, file_name = self.perform_eval(mode, task_file, seed, num_episode, save_file_prefix)

        print(f"Saved the result file to: {file_name}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate a policy store")
    parser.add_argument("policy_store_dir", type=str, help="Path to the policy directory")
    parser.add_argument("mode", type=str, choices=["pve", "pvp"], help="Evaluation mode")
    parser.add_argument(
        "-t", "--task-file", type=str, default=EVAL_TASK_FILE, help="Path to the task file"
    )
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-n", "--num-episode", type=int, default=None, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "-r", "--repeat", type=int, default=1, help="Number of times to repeat the evaluation"
    )
    parser.add_argument(
        "--save-file-prefix", type=str, default=None, help="Prefix for the save file"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    runner = EvalRunner(args.policy_store_dir, args.debug)
    for i in range(args.repeat):
        if i > 0:
            args.seed = None  # this will sample new seed
        runner.run(args.mode, args.task_file, args.seed, args.num_episode, args.save_file_prefix)
