import torch
import os
import numpy as np
from pathlib import Path
from gym.spaces import Box, Discrete, Tuple

from utils import get_config, get_cent_act_dim, get_dim_from_space, make_train_env
from runner import Runner

if __name__ == "__main__":
    config = get_config()
    # torch setup
    if config['cuda'] and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(config['n_training_threads'])
        if config['cuda_deterministic']:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            print("choose to use cpu...")
            device = torch.device("cpu")
            torch.set_num_threads(config['n_training_threads'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    # dir setup
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + '/' + config['save_path']+ '/'+config['scenario_name']+'_' + config['algorithm_name']+'_'+str(config['experiment_index']))
    if not run_dir.exists(): 
        os.makedirs(str(run_dir))
    if not os.path.exists(run_dir/'logs'):
        os.makedirs(run_dir/'logs')
    if not os.path.exists(run_dir/'models'):
        os.makedirs(run_dir/'models')
    else: 
        config['restore'] = True
    config['fullpath'] = str(run_dir)
    config['model_path'] = str(run_dir) + '/models'
    config['log_path'] = str(run_dir) + '/logs'
    # env setup
    env = make_train_env(config)
    eval_env = make_train_env(config)
    # algorithm setup
    if config['share_policy']:
        config['policy_info'] = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }
        def policy_mapping_fn(id): return 'policy_0'
    else:
        config['policy_info'] = {
            'policy_' + str(agent_id): {
                "cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[agent_id],
                'obs_dim': get_dim_from_space(env.observation_space[agent_id]),
                "share_obs_space": env.share_observation_space[agent_id],
                "act_space": env.action_space[agent_id],
                "act_dim": get_dim_from_space(env.action_space[agent_id]),
                "output_dim": sum(get_dim_from_space(env.action_space[agent_id])) if isinstance((get_dim_from_space(env.action_space[agent_id]), np.ndarray)) else get_dim_from_space(env.action_space[agent_id]),
            }
            for agent_id in range(config['num_agents'])
        }
        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)
    # Q: why do we need this one
    config['policy_mapping_fn']=policy_mapping_fn
    # more parameters
    config['env'] = env 
    config['eval_env'] = eval_env
    config['discrete'] = isinstance(config['policy_info']["act_space"], Discrete) or "MultiDiscrete" in (config['policy_info']["act_space"].__class__.__name__)
    config['multidiscrete'] = ("MultiDiscrete" in config['policy_info']["act_space"].__class__.__name__)
    config['tpdv'] = dict(dtype=torch.float32, device=config['device'])
    # train
    total_steps = 0
    runner = Runner(config)
    while total_steps < config['env_steps']:
        total_steps = runner.run()
    # close
    env.close()
    eval_env.close()