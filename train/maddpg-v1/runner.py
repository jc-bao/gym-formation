from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = f'{self.args.save_dir}/{self.args.scenario_name}'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        np.random.seed(4)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            actions.extend([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0] for _ in range(self.args.n_agents, self.args.n_players))

            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel(f'episode * {str(self.args.evaluate_rate / self.episode_limit)}')
                plt.ylabel('average returns')
                plt.savefig(f'{self.save_path}/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.noise - 0.0000005)
            np.save(f'{self.save_path}/returns', returns)

    def evaluate(self, rnd = False):
        returns = []
        steps = []
        collides = []
        for _ in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            c_num = 0
            for j in range(self.args.evaluate_episode_len):
                if rnd: self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                actions.extend([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0] for _ in range(self.args.n_agents, self.args.n_players))

                s_next, r, done, info = self.env.step(actions)
                for i in range(self.args.n_agents):
                    c_num += (info[i]['collisions']-1)
                c_num/=2
                rewards += r[0][0] if isinstance(r[0], list) else r[0]
                s = s_next
                if np.all(done): 
                    break
            steps.append(j)
            returns.append(rewards)
            collides.append(c_num)
            print('Returns is', rewards, 'Final Reward:', r[0], 'Steps:', j, 'Collisions:', c_num)
        print('Average returns is', np.mean(returns), 'Average steps is', np.mean(steps), 'pm', np.std(steps), 'Average collides is', np.mean(collides), 'pm', np.std(collides))
        return sum(returns) / self.args.evaluate_episodes