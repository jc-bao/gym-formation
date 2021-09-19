from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
import formation_gym


if __name__ == '__main__':
    # get the params
    args = get_args()
    env = formation_gym.make_env(args.scenario_name, benchmark = False, num_agents = args.num_agents)
    args.n_agents = args.num_agents
    args.n_players = 0
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
    action_shape = []
    for content in env.action_space:
        action_shape.append(content.n)
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    args.high_action = 1
    args.low_action = -1
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate(True)
        print('Average returns is', returns)
    else:
        runner.run()
