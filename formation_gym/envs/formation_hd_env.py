import numpy as np
from scipy.spatial.distance import directed_hausdorff

from formation_gym.scenario import BaseScenario
from formation_gym.core import World, Agent, Landmark

'''
use Hausdorff distance as reward function
refer to https://www.wikiwand.com/en/Hausdorff_distance#/Applications
'''

class Scenario(BaseScenario):
    def make_world(self, num_agents = 3, num_landmarks = 3, episode_length = 25):
        # world properties
        world = World()
        world.world_length = episode_length
        world.dim_c = 2 # communication channel
        world.collaborative = True
        # agent properties
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.08
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
        # change landmark pos for visualization (Note: not necessary for training)
        u = [a.state.p_pos for a in world.agents]
        v = [l.state.p_pos for l in world.landmarks]
        delta = np.mean(u, 0) - np.mean(v, 0)
        for i in range(len(world.landmarks)):
            world.landmarks[i].state.p_pos += delta # synchronize the center of landmarks and agents
        # agent pos & communication
        other_pos = np.array([])
        comm = np.array([])
        for other in world.agents:
            if other is agent: continue
            comm = np.append(comm, other.state.c)
            other_pos = np.append(other_pos, other.state.p_pos - agent.state.p_pos)
        return np.concatenate((agent.state.p_vel, other_pos, comm, self.ideal_shape.flatten(), self.ideal_vel))

    def reward(self, agent, world):
        # part1: formation reward: define by hausdorff distance
        rew = 0
        agent_shape = [a.state.p_pos for a in world.agents]
        agent_shape = agent_shape - np.mean(agent_shape, 0)
        rew = -max(directed_hausdorff(agent_shape, self.ideal_shape)[0], directed_hausdorff(self.ideal_shape, agent_shape)[0])
        # part2: velocity reward: define by overall velocity difference
        mean_vel = np.mean([a.state.p_vel for a in world.agents], axis = 0)
        rew -= np.linalg.norm(self.ideal_vel - mean_vel)
        # part3: collision
        if agent.collide:
            for a in world.agents:
                if agent!=a and self.is_collision(a, agent):
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
        self.ideal_shape = []
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
            pos = np.random.uniform(-1, +1, world.dim_p)
            self.ideal_shape.append(pos)
            landmark.state.p_pos = pos
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.ideal_shape = self.ideal_shape - np.mean(self.ideal_shape, 0)
        # ideal velocity
        self.ideal_vel = np.random.uniform(-1, +1, world.dim_p)

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