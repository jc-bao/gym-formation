import argparse
import numpy as np

import formation_gym
from formation_gym.policy import InteractivePolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='formation_hd_env', help='Path of the scenario Python script.')
    parser.add_argument('-n', '--num-agents', type=int, default=3, help='Number of agents')
    parser.add_argument('-d', '--demo', action='store_true', help='If show the demo.')
    parser.add_argument('-r', '--random', action='store_true', help='If use random policy.')
    args = parser.parse_args()

    env = formation_gym.make_env(args.scenario, benchmark=False, num_agents = args.num_agents)
    env.render()
    obs_n = env.reset()
    cnt = 0
    while True:
        # query for action from each agent's policy
        act_n = []
        # demo policy
        if args.demo:
            done, act_n = formation_gym.ezpolicy(env.world)
            if not done: cnt+=1
            else:
                print('total steps: ', cnt)
        # random policy
        elif args.random: 
            act_n = [space.sample() for space in env.action_space]
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()