import imp
from multiagent.environment import MultiAgentEnv
import os.path as osp
import numpy as np

def make_env(scenario_name='basic_formation_env', benchmark=False):
    # load scenario from script
    pathname = osp.join(osp.dirname(__file__), 'envs/'+scenario_name+'.py')
    scenario = imp.load_source('', pathname).Scenario() 
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, shared_viewer = False)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, shared_viewer = True)
    return env

def ezpolicy(world):
    eps = 0.1
    act_n = []
    u = np.zeros(5) # 5-d because of no-move action
    for i in range(len(world.agents)):
        move1 = True
        move2 = True
        if -(world.agents[i].state.p_pos[0] - world.landmarks[i].state.p_pos[0])>eps: 
            u[1] += 0.5
        elif (world.agents[i].state.p_pos[0] - world.landmarks[i].state.p_pos[0])>eps: 
            u[2] += 0.5
        else: move1 = False
        if -(world.agents[i].state.p_pos[1] - world.landmarks[i].state.p_pos[1])>eps: 
            u[3] += 0.5
        elif (world.agents[i].state.p_pos[1] - world.landmarks[i].state.p_pos[1])>eps: 
            u[4] += 0.5
        else: move2 = False
        u[0] += 1.0
        act_n.append(np.concatenate([u, np.zeros(world.dim_c)]))
    done = not (move1 and move2)
    return done, act_n