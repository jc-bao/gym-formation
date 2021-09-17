import imp
from multiagent.environment import MultiAgentEnv
import os.path as osp

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