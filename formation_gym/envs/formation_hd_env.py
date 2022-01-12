import numpy as np
from scipy.spatial.distance import directed_hausdorff

from formation_gym.scenario import BaseScenario
from formation_gym.core import World, Agent, Landmark

'''
use Hausdorff distance as reward function
refer to https://www.wikiwand.com/en/Hausdorff_distance#/Applications
'''

class Scenario(BaseScenario):
    def make_world(self, num_agents = 3, episode_length = 100):
        # world properties
        world = World()
        world.world_length = episode_length
        world.dim_c = 2 # communication channel
        world.collaborative = True
        self.num_agents = num_agents
        # agent properties
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # landmark properties
        world.landmarks = [Landmark() for i in range(num_agents)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmarks %d' % i
            landmark.collide = False 
            landmark.movable = False
            landmark.size = 0.05
        # initial conditions
        self.reset_world(world)
        return world
    
    def observation(self, agent, world):
        # change landmark pos for visualization (Note: not necessary for training)
        u = [a.state.p_pos for a in world.agents]
        v = [l.state.p_pos for l in world.landmarks]
        delta = np.mean(u,0) - np.mean(v,0)
        for l in world.landmarks:
            l.state.p_pos += delta
        other_pos = np.array([])
        for other in world.agents:
            if other is agent: continue
            other_pos = np.append(other_pos, other.state.p_pos - agent.state.p_pos)
        obs = {
            'observation': np.append(agent.state.p_vel, other_pos),
            'achieved_goal': np.concatenate(u-np.mean(u,0)),
            'desired_goal': self.ideal_shape.flatten(),
        }
        return obs

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
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np.random.uniform(-1.5, +1.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # landmark: use can use `generate_shape` to generate target shape
        # self.ideal_shape = self.generate_shape(3).reshape(-1,2)
        self.ideal_shape = []
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            while True:
                pos = np.random.uniform(-1.5, +1.5, world.dim_p)
                shape = np.array(self.ideal_shape)
                if len(self.ideal_shape) == 0 or (np.linalg.norm(shape-pos, axis=-1) > 0.3).all():
                    break
            self.ideal_shape.append(pos)
            landmark.state.p_pos = self.ideal_shape[i]
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

    def generate_shape(self, layer, layer_shapes = None):
        # this is default shape
        layer_shapes = layer_shapes or np.array([
            [[0, -1], [0.5, 0], [0, 1]],
            [[0, 1.6], [-1, 0], [1, 0]],
            [[1.5, 0], [0, 0], [-1.5, 0]],
            [[0, 0.6], [1, 0], [-1, 0]],
        ])
        num_layers = layer_shapes.shape[0]
        assert layer < num_layers, 'Layer shape is not enough!'
        num_agents_per_layer = layer_shapes.shape[1]
        if layer == 0:
            return layer_shapes[0]
        else:
            old_shape = self.generate_shape(layer-1)
            shape = np.array([(layer_shapes[layer][i] + old_shape * 0.45) for i in range(num_agents_per_layer)])
        return shape

if __name__ == '__main__':
    s = Scenario()