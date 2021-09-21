import sys
import os
import numpy as np
from pathlib import Path
import socket
import setproctitle
import torch
from config import get_config
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
import formation_gym
from offpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            print(all_args.env_name)
            if all_args.env_name == "formation":
                env = formation_gym.make_env(all_args.scenario_name, benchmark = False, num_agents = all_args.num_agents)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "formation":
                env = formation_gym.make_env(all_args.scenario_name, benchmark = False, num_agents = all_args.num_agents)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of agents")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + "/results") / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name

    # create env
    env = make_train_env(all_args)
    # env = formation_gym.make_env(all_args.scenario_name, False, all_args.num_agents)
    num_agents = all_args.num_agents

    # create policies and mapping fn
    if all_args.share_policy:
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qmix", "vdn"]:
        from offpolicy.runner.rnn.mpe_runner import MPERunner as Runner
        assert all_args.n_rollout_threads == 1, (
            "only support 1 env in recurrent version.")
        eval_env = env
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from offpolicy.runner.mlp.mpe_runner import MPERunner as Runner
        eval_env = make_eval_env(all_args)
    else:
        raise NotImplementedError

    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "use_same_share_obs": all_args.use_same_share_obs,
              "run_dir": run_dir
              }

    runner = Runner(config=config, render = True)
    runner.eval()

    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

if __name__ == "__main__":
    main(sys.argv[1:])
