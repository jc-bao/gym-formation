import argparse
import numpy as np

import formation_gym
from formation_gym.policy import InteractivePolicy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='basic_formation_env', help='Path of the scenario Python script.')
    parser.add_argument('-n', '--num-agents', type=int, default=3, help='Number of agents')
    parser.add_argument('-d', '--demo', action='store_true', help='If show the demo.')
    parser.add_argument('-r', '--random', action='store_true', help='If use random policy.')
    args = parser.parse_args()

    env = formation_gym.make_env(args.scenario, True, args.num_agents)
    env.render()
    policies = [InteractivePolicy(env,i) for i in range(env.num_agents)]
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
                # break
        # keyboard policy
        else:
            for i, policy in enumerate(policies):
                act_n.append(policy.action(obs_n[i]))
        # random policy
        if args.random: act_n = np.random.uniform(-1, 1, size=np.array(act_n).shape)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        print('reward:',reward_n)
        # render all agent views
        env.render()