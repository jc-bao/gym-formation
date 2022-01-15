import imp
from .environment import MultiAgentEnv
import os.path as osp
import numpy as np

def make_env(scenario_name='basic_formation_env', benchmark=False, num_agents = 3, reward_type = 'sparse'):
    # load scenario from script
    pathname = osp.join(osp.dirname(__file__), 'envs/'+scenario_name+'.py')
    scenario = imp.load_source('', pathname).Scenario() 
    # create world
    world = scenario.make_world(num_agents) # use same number of agent and landmarks
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, shared_viewer = True)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, shared_viewer = True, reward_type = reward_type)
    return env

def ezpolicy(obs):
    num_agents = len(obs)/6
    assert num_agents.is_integer(), num_agents
    num_agents = int(num_agents)
    # get info from observation
    p_vel = obs[:2]
    other_pos = obs[2:2*num_agents]
    ideal_shape = obs[4*num_agents-2:6*num_agents-2]
    ideal_shape = np.reshape(ideal_shape, (-1, 2))
    ideal_shape = ideal_shape - np.mean(ideal_shape, axis = 0)
    ideal_vel = obs[-2:]
    # calculate relative formation
    current_shape = np.append(other_pos, [0,0])
    current_shape = np.reshape(current_shape, (-1,2))
    current_shape -= np.mean(current_shape, axis = 0)
    # get action
    sort_mark_idx = np.argsort([np.linalg.norm(current_shape[-1] - mark) for mark in ideal_shape]) # distance to different landmarks
    for idx in sort_mark_idx:
        closest_agent_idx = np.argmin([np.linalg.norm(agent - ideal_shape[idx]) for agent in current_shape])
        if closest_agent_idx == (num_agents - 1) or idx == sort_mark_idx[-1]: # this agent is the closet agent
            act = np.clip(0.5*(ideal_shape[idx] - current_shape[-1]), -1, 1)
            break
    # add ideal velocity control to action
    done = np.linalg.norm(ideal_shape - current_shape) < 0.01
    if done: 
        act += ideal_vel
    else:
        act += ideal_vel * 0.3
    return act

def get_action_BFS(policy, obs, num_agents_per_layer):
    '''
    :param policy: agent policy function
    :param obs: total observation
    :param num_agents_per_layer: number of agents per group
    '''
    num_layer = np.log(len(obs))/ np.log(num_agents_per_layer)
    assert num_layer.is_integer(), 'Observation shape error!'
    queue = [obs]
    act = []
    while queue:
        current_layer_obs = queue.pop(0)
        current_layer_num_agents = len(current_layer_obs)
        next_layer_num_agents = int(len(current_layer_obs)/num_agents_per_layer)
        for i in range(num_agents_per_layer):
            leader_obs = current_layer_obs[i*next_layer_num_agents]
            # get current layer leader observation
            p_vel = leader_obs[:2]
            # get observation of others by inference center
            current_shape = np.insert(leader_obs[2:2*current_layer_num_agents], 2*i*next_layer_num_agents, [0,0]).reshape((-1, 2))
            layer_current_shape = np.array([np.mean(current_shape[next_layer_num_agents*k:next_layer_num_agents*(k+1)], axis = 0) for k in range(num_agents_per_layer)])
            layer_current_shape -= layer_current_shape[i]
            layer_current_shape = np.delete(layer_current_shape, i, 0).flatten()
            # get ideal formation
            ideal_shape = np.reshape(leader_obs[4*current_layer_num_agents-2:6*current_layer_num_agents-2], (-1, 2))
            layer_target_shape = np.array([np.mean(ideal_shape[next_layer_num_agents*(k):next_layer_num_agents*(k+1)], axis = 0) for k in range(num_agents_per_layer)]).flatten()
            # get ideal velocity
            layer_target_vel = leader_obs[-2:]
            obs_input = np.concatenate((p_vel, layer_current_shape, [0]*2*(num_agents_per_layer-1), layer_target_shape, layer_target_vel))
            current_layer = np.log(current_layer_num_agents)/ np.log(num_agents_per_layer)
            next_layer_target_vel = policy(obs_input) * (current_layer)
            # next layer observation
            if next_layer_num_agents == 1:
                # END case: reach the last layer and append the action
                act.append(next_layer_target_vel)
            else:
                next_layer_obs = []
                for j in range(i*next_layer_num_agents, (i+1)*next_layer_num_agents):
                    # remove redundent observation
                    obs_n = current_layer_obs[j]
                    p_vel = obs_n[:2]
                    others_pos = obs_n[2:2*current_layer_num_agents]
                    others_pos = others_pos[2*i*next_layer_num_agents:2*(i+1)*next_layer_num_agents-2]
                    comm = [0]*2*(next_layer_num_agents-1)
                    shape = obs_n[4*current_layer_num_agents-2:6*current_layer_num_agents-2]
                    shape = shape[2*i*next_layer_num_agents:2*(i+1)*next_layer_num_agents]
                    tar_vel = next_layer_target_vel
                    obs_n = np.concatenate((p_vel, others_pos, comm, shape, tar_vel))
                    next_layer_obs.append(obs_n)
                queue.append(next_layer_obs)
    return act