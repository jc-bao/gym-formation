import numpy as np
from scipy.spatial.distance import directed_hausdorff

from formation_gym.scenario import BaseScenario
from formation_gym.core import World, Agent, Landmark

'''
use Hausdorff distance as reward function
refer to https://www.wikiwand.com/en/Hausdorff_distance#/Applications

partial observation environment
'''

class Scenario(BaseScenario):
    def make_world(self, num_agents = 4, num_landmarks = 4, num_obs = 2, world_length = 25):
        self.num_obs = num_obs
        self.num_agents = num_agents
        # world properties
        world = World()
        world.world_length = world_length
        world.dim_c = 2 # communication channel
        world.collaborative = True
        # agent properties
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.04
        # landmark properties
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmarks %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02
        # initial conditions
        self.reset_world(world)
        return world
    
    def observation(self, agent, world):
        # landmark pos
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos)
        # agent pos & communication
        other_pos = []
        comm = []
        cnt = 0
        # way3: watch for 2 guys
        # get agent ID
        agent_id = int(agent.name.split()[-1])
        idx = [i % self.num_agents for i in range(agent_id, agent_id + self.num_obs)]
        for i in idx:
            other_pos.append(world.agents[i].state.p_pos - agent.state.p_pos)
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
        # make the furthest point to zero
        # way1: make the far observation to zero
        # others_dist = np.linalg.norm(other_pos, axis = 1)
        # idx = np.argpartition(others_dist, self.num_obs)
        # for i in idx[self.num_obs:]:
        #     other_pos[i] = np.zeros(world.dim_p)
        # way2: remove the far obs
        # other_pos = other_pos[idx[:self.num_obs]]
        return np.concatenate([agent.state.p_vel]+[agent.state.p_pos]+entity_pos + other_pos + comm)

    def reward(self, agent, world):
        rew = 0
        u = [a.state.p_pos for a in world.agents]
        v = [l.state.p_pos for l in world.landmarks]
        delta = np.mean(u, 0) - np.mean(v, 0)
        u = u - np.mean(u, 0)
        v = v - np.mean(v, 0)
        rew = -max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        # change landmark pos and color
        # for i in range(len(world.landmarks)):
            # world.landmarks[i].state.p_pos += delta
            # dist = min([np.linalg.norm(a.state.p_pos - world.landmarks[i].state.p_pos) for a in world.agents])
            # if dist <= 0.2: world.landmarks[i].color = np.array([0, 0.6, 0])
        self.set_bound(world)
        if agent.collide:
            for a in world.agents:
                if  agent!=a and self.is_collision(a, agent):
                    rew -= 1
        return rew

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
        return dist < (agent1.size + agent2.size)

    def set_bound(self, world):
        for agent in world.agents:
            agent.state.p_pos = np.clip(agent.state.p_pos, [-2, -2], [2, 2])

