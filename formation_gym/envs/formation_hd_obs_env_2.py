import numpy as np
from scipy.spatial.distance import directed_hausdorff

from formation_gym.scenario import BaseScenario
from formation_gym.core import World, Agent, Landmark, Wall

'''
use Hausdorff distance as reward function
refer to https://www.wikiwand.com/en/Hausdorff_distance#/Applications
add obstables into consideration
'''

class Scenario(BaseScenario):
    def make_world(self, num_agents = 4, num_landmarks = 4, num_obstacles = 3, world_length = 50):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_obstacles = num_obstacles
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
            agent.size = 0.1
        # landmark and obstacles properties
        world.landmarks = [Landmark() for i in range(num_landmarks + num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            # setup landmarks
            if i < num_landmarks:
                landmark.name = 'landmarks %d' % i
                landmark.collide = False 
                landmark.movable = False
                landmark.size = 0.02
            # setup obstacles
            else: 
                landmark.name = 'obstacles %d' % (i - num_landmarks)
                landmark.collide = True 
                landmark.movable = True
                landmark.size = 0.15
        # setup walls 
        world.walls = []
        world.walls.append(Wall(orient='H',axis_pos=2.6,endpoints=(-2.2, 2.2),width=0.2,hard=True))
        world.walls.append(Wall(orient='H',axis_pos=-2.6,endpoints=(-2.2, 2.2),width=0.2,hard=True))
        world.walls.append(Wall(orient='V',axis_pos=2.2,endpoints=(-2.6, 2.6),width=0.2,hard=True))
        world.walls.append(Wall(orient='V',axis_pos=-2.2,endpoints=(-2.6, 2.6),width=0.2,hard=True))
        # initial conditions
        self.reset_world(world)
        return world
    
    def observation(self, agent, world):
        # landmark pos
        entity_pos = []
        for entity in world.landmarks[:self.num_landmarks]:
            entity_pos.append(entity.state.p_pos)
        for entity in world.landmarks[self.num_landmarks:]:
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
        u = [a.state.p_pos for a in world.agents]
        v = [l.state.p_pos for l in world.landmarks[:self.num_landmarks]]
        delta = np.mean(u, 0) - np.mean(v, 0)
        u = u - np.mean(u, 0)
        v = v - np.mean(v, 0)
        rew = -max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        # set boundary
        # self.set_bound(world)
        # change landmark pos and color
        for i, landmark in enumerate(world.landmarks):
            if i < self.num_landmarks:
                delta = [0, 0]
                landmark.state.p_pos += delta
            else:
                landmark.state.p_vel = np.array([0, -1])
            # dist = min([np.linalg.norm(a.state.p_pos - world.landmarks[i].state.p_pos) for a in world.agents])
            # if dist <= 0.2: world.landmarks[i].color = np.array([0, 0.6, 0])
        if agent.collide:
            for a in world.agents:
                if agent!=a and self.is_collision(a, agent):
                    rew -= 2
            for l in world.landmarks[self.num_landmarks:]:
                if self.is_collision(l, agent):
                    rew -= 2
        return rew

    def reset_world(self, world):
        # agent
        for agent in world.agents:
            agent.color = np.array([0.65, 0.65, 0.85])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # landmark
        for i, landmark in enumerate(world.landmarks):
            step = np.linspace(-1.8, 1.8, self.num_obstacles+1)
            # setup landmarks
            if i <self.num_landmarks:
                landmark.color = np.array([0, 0.6, 0])
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            # setup obstacles
            else: 
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.state.p_pos = np.random.uniform([step[i-self.num_landmarks], 2.0], [step[i+1-self.num_landmarks], 2.5])
                landmark.state.p_vel = np.array([0, -1])
            

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
            agent.state.p_pos = np.clip(agent.state.p_pos, [-2, -10], [2, 10])
        for landmark in world.landmarks[self.num_landmarks:]:
            landmark.state.p_pos = np.clip(landmark.state.p_pos, [-2, -10], [2, 10])