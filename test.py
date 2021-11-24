import argparse
import numpy as np

import formation_gym

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='formation_hd_env', help='Path of the scenario Python script.')
    parser.add_argument('-n', '--num-agents', type=int, default=3, help='Number of agents')
    parser.add_argument('-r', '--random', action='store_true', help='If use random policy.')
    parser.add_argument('--num-layer', type=int, default = 1, help = 'use hierachy policy to control')
    args = parser.parse_args()

    env = formation_gym.make_env(args.scenario, benchmark=False, num_agents = args.num_agents**args.num_layer)
    obs_n = env.reset()
    total_num_agents = args.num_agents**args.num_layer
    while True:
        # random policy
        if args.random: 
            act_n = [space.sample() for space in env.action_space]
        # demo policy
        else:
            act_n = formation_gym.get_action_BFS(formation_gym.ezpolicy, obs_n, args.num_agents)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        if np.all(done_n):
            obs_n = env.reset()
        # render all agent views
        env.render()