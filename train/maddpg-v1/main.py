from runner import Runner
from common.arguments import get_args
import numpy as np
import formation_gym


if __name__ == '__main__':
    # get the params
    args = get_args()
    env = formation_gym.make_env(args.scenario_name, benchmark = False, num_agents = args.num_agents)
    args.n_agents = args.num_agents
    args.n_players = 0
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
    action_shape = [content.shape[0] for content in env.action_space]
    args.action_shape = action_shape[:args.n_agents]
    args.high_action = 1
    args.low_action = -1
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate(True)
        print('Average returns is', returns)
    else:
        runner.run()
