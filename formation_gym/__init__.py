import imp
from .environment import MultiAgentEnv
import os.path as osp
import numpy as np

def make_env(scenario_name='basic_formation_env', benchmark=False, num_agents = 3):
    # load scenario from script
    pathname = osp.join(osp.dirname(__file__), 'envs/'+scenario_name+'.py')
    scenario = imp.load_source('', pathname).Scenario() 
    # create world
    world = scenario.make_world(num_agents, num_agents) # use same number of agent and landmarks
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, shared_viewer = True)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, shared_viewer = True)
    return env

def ezpolicy(obs_n):
    num_agents = len(obs_n)
    act_n = []
    for i,obs in enumerate(obs_n):
        # get info from observation
        print(obs)
        p_vel = obs[:2]
        other_pos = obs[2:2*num_agents]
        ideal_shape = obs[2*num_agents+4:4*num_agents+4]
        ideal_shape = np.reshape(ideal_shape, (-1, 2))
        ideal_vel = obs[-1]
        # calculate relative formation
        current_shape = np.insert(other_pos, i*2, [0,0])
        current_shape = np.reshape(current_shape, (-1,2))
        current_shape -= np.mean(current_shape, axis = 0)
        # get action
        act = np.clip(ideal_shape[i] - current_shape[i], -1, 1)
        act_n.append(act)
    return act_n