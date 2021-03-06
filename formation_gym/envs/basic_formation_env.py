import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark

class Scenario(BaseScenario):
    def make_world(self, num_agents = 3, num_landmarks = 3):
        # world properties
        world = World()
        world.dim_c = 2 # communication channel
        world.collaborative = True
        # agent properties
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1 
        # landmark properties
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmarks %d' % i
            landmark.collide = False 
            landmark.movable = False
        # initial conditions
        self.reset_world(world)
        return world
    
    def observation(self, agent, world):
        # landmark pos
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # agent pos & communication
        other_pos = []
        comm = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel]+[agent.state.p_pos]+entity_pos + other_pos + comm)

    def reward(self, agent, world):
        rew = 0
        for l in world.landmarks:
            dists = [np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
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

