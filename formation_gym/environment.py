import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from pprint import pprint

# update bounds to center around agent
cam_range = 2

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.GoalEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, post_step_callback=None,
                 shared_viewer=True, discrete_action=False):

        self.world = world
        self.world_length = self.world.world_length
        self._max_episode_steps = 100
        self.current_step = 0
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.num_agents = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # mimic vector env in order to render
        self.num_envs = 1

        self.post_step_callback = post_step_callback

        # environment parameters
        # self.discrete_action_space = True -> use number to control
        self.discrete_action_space = discrete_action

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(
            world, 'discrete_action') else False
        # in this env, force_discrete_action == False��because world do not have discrete_action

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        #self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = spaces.Box( \
            low=-self.agents[0].u_range, high=+self.agents[0].u_range, \
                shape=(world.dim_p*self.num_agents,), dtype=np.float32)
        obs_dim = len(observation_callback(self.agents[0], self.world))
        self.observation_space = spaces.Box( \
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.num_agents
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    def step(self, action_n):
        self.current_step += 1
        self.agents = self.world.policy_agents
        # set action for each agent
        action_n = np.array(action_n).reshape(self.num_agents, -1)
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # advance world state
        self.world.step()  # core.step()
        obs = self._get_obs()
        rew = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])
        info = {'is_success': rew==0}
        return obs, rew, self._get_done(), info

    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        self.agents = self.world.policy_agents

        return self._get_obs()

    # get observation for a all agent
    def _get_obs(self):
        if self.observation_callback is None:
            return np.zeros(0)
        obs = np.zeros(0)
        for agent in self.agents:
            agent_obs = self.observation_callback(agent, self.world)
            obs = np.append(obs, agent_obs['observation'])
        ag = agent_obs['achieved_goal']
        g = agent_obs['desired_goal']
        obs = {
            'observation': obs,
            'achieved_goal': ag, 
            'desired_goal': g
        }
        return obs

    def _get_done(self):
        return self.current_step == self._max_episode_steps

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        ideal_shape = desired_goal.reshape(self.num_agents, 1, self.world.dim_p)
        agent_shape = achieved_goal.reshape(1, self.num_agents, self.world.dim_p)
        # from matplotlib import pyplot
        # # for pos in ideal_shape:
        # pyplot.scatter(ideal_shape[..., 0], ideal_shape[...,1])
        # pyplot.scatter(agent_shape[..., 0], agent_shape[..., 1])
        # pyplot.show()
        # exit()
        aa = np.repeat(ideal_shape,self.num_agents,axis = 1)
        bb = np.repeat(agent_shape,self.num_agents,axis = 0)
        done = (np.abs(aa-bb) < 0.05).all(axis=-1).any(axis=-1).all()
        return float(done) - 1

    # set env action for a particular agent
    def _set_action(self, action, agent, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        action = [action]
        if agent.movable:
            # physical action
            if self.discrete_action_input: # use number
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
                d = self.world.dim_p
            else: # use one-hot
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                    d = 5
                else:
                    if self.force_discrete_action:
                        p = np.argmax(action[0][0:self.world.dim_p])
                        action[0][:] = 0.0
                        action[0][p] = 1.0
                    agent.action.u = action[0][0:self.world.dim_p]
                    d = self.world.dim_p
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

            action = action[1:]

        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]

            action = action[1:]

        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='rgb_array', close=False):
        if close:
            # close any existic renderers
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []

        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' +
                                agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)

            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from . import rendering
                self.viewers[i] = rendering.Viewer(300, 300)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from . import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            self.comm_geoms = []

            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()

                entity_comm_geoms = []

                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)

                else:
                    geom.set_color(*entity.color)
                    if entity.channel is not None:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)
            for wall in self.world.walls:
                corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                           (wall.axis_pos - 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
                if wall.orient == 'H':
                    corners = tuple(c[::-1] for c in corners)
                geom = rendering.make_polygon(corners)
                if wall.hard:
                    geom.set_color(*wall.color)
                else:
                    geom.set_color(*wall.color, alpha=0.5)
                self.render_geoms.append(geom)

            # add geoms to viewer
            # for viewer in self.viewers:
            #     viewer.geoms = []
            #     for geom in self.render_geoms:
            #         viewer.add_geom(geom)

            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from . import rendering

            if self.shared_viewer:
                u = [a.state.p_pos for a in self.agents]
                pos = np.mean(u, 0)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(
                pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                if 'agent' in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.state.c[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)
                else:
                    self.render_geoms[e].set_color(*entity.color)
                    if entity.channel is not None:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.channel[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)

            # render to display or array
            results.append(self.viewers[i].render(
                return_rgb_array=mode == 'rgb_array'))

        return results[0]

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx
