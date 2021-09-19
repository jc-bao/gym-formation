import torch
import os
import numpy as np
from pathlib import Path

from utils import get_config, get_cent_act_dim, get_dim_from_space,DummyVecEnv, SubprocVecEnv
import formation_gym
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
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + '/' + config['save_path']+ '_'+config['scenario_name']+'_' + config['algorithm_name']+'_'+config['experiment_index'])
    if not run_dir.exists(): 
        os.makedirs(str(run_dir))
    else:
        if not os.path.exists(run_dir/'logs'):
            os.makedirs('logs')
        if not os.path.exists(run_dir/'models'):
            os.makedirs('models')
        else: 
            config['restore'] = True
    # env setup
    if config['n_rollout_threads'] == 1:
        env = DummyVecEnv(formation_gym.make_env(scenario_name = config['scenario_name'], benchmark = False, num_agents = config['num_agents']))
        eval_env = DummyVecEnv(formation_gym.make_env(scenario_name = config['scenario_name'], benchmark = False, num_agents = config['num_agents']))
    else:
        env = SubprocVecEnv([
            formation_gym.make_env(scenario_name = config['scenario_name'], benchmark = False, num_agents = config['num_agents']) for i in range(config['n_rollout_threads'])
            ])
        eval_env = SubprocVecEnv([
            formation_gym.make_env(scenario_name = config['scenario_name'], benchmark = False, num_agents = config['num_agents']) for i in range(config['n_rollout_threads'])
            ])
    
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
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(config['num_agents'])
        }
        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)
    # Q: why do we need this one
    config['policy_mapping_fn']=policy_mapping_fn
    # train
    total_steps = 0
    runner = Runner(config)
    while total_steps < config['env_steps']:
        total_steps = runner.run()
    # close
    env.close()
    eval_env.close()


        
