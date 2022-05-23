import numpy as np
from scipy.spatial.distance import directed_hausdorff

from formation_gym.scenario import BaseScenario
from formation_gym.core import World, Agent, Landmark

'''
use Hausdorff distance as reward function
refer to https://www.wikiwand.com/en/Hausdorff_distance#/Applications
'''

class Scenario(BaseScenario):
    def make_world(self, num_agents = 3, num_landmarks = 3, episode_length = 25, reward_type = 'fix'):
        self.reward_type = reward_type
        self.num_agents = num_agents
        # world properties
        world = World()
        world.world_length = episode_length
        world.dim_c = 2 # communication channel
        world.collaborative = True
        # agent properties
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # landmark properties
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmarks %d' % i
            landmark.collide = False 
            landmark.movable = False
            landmark.size = 0.02
        # initial conditions
        self.reset_world(world)
        return world
    
    def observation(self, agent, world):
        # agent pos & communication
        entity_pos = [entity.state.p_pos for entity in world.landmarks]
        other_pos = []
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel]+[agent.state.p_pos]+entity_pos + other_pos + comm)

    def reward(self, agent, world):
        if self.reward_type == 'hd':
            rew = 0
            u = np.array([a.state.p_pos for a in world.agents])
            v = np.array([l.state.p_pos for l in world.landmarks])
            # delta = np.mean(u, 0) - np.mean(v, 0)
            u -= np.mean(u, 0)
            v -= np.mean(v, 0)
            rew = -max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
            if agent.collide:
                for a in world.agents:
                    if self.is_collision(a, agent):
                        rew -= 1
        elif self.reward_type == 'min_dist':
            rew = 0
            u = np.array([a.state.p_pos for a in world.agents])
            v = np.array([l.state.p_pos for l in world.landmarks])
            u -= np.mean(u, 0)
            v -= np.mean(v, 0)
            for l in v:
                dists = [np.linalg.norm(a - l) for a in u]
                rew -= min(dists)
        elif self.reward_type == 'fix':
            rew = 0
            u = np.array([a.state.p_pos for a in world.agents])
            v = np.array([l.state.p_pos for l in world.landmarks])
            u -= np.mean(u, 0)
            v -= np.mean(v, 0)
            rew -= np.linalg.norm(u - v)
        return rew

    def done(self, agent, world):
        u = np.array([a.state.p_pos for a in world.agents])
        v = np.array([l.state.p_pos for l in world.landmarks])
        # delta = np.mean(u, 0) - np.mean(v, 0)
        u -= np.mean(u, 0)
        v -= np.mean(v, 0)
        dist = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        return dist < 0.05*self.num_agents

    def reset_world(self, world):
        # agent
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # landmark
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # get data to debug
        rew = self.reward(agent, world)
        collisions = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    collisions += 1
        min_dists = 0
        occupied_landmarks = 0
        for l in world.landmarks:
            dists = [np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]
            min_dists += min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        return {
            'reward': rew, 
            'collisions': collisions, 
            'min_dists': min_dists, 
            'occupied_landmarks': occupied_landmarks
        }

    def is_collision(self, agent1, agent2):
        dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        return dist < (agent1.size + agent2.size)/2