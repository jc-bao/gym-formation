import os
from tensorboardX import SummaryWriter
import torch
import numpy as np
import time

from maddpg import MADDPG
from maddpg import MADDPGPolicy
from utils import DecayThenFlatSchedule
from replaybuffer import MlpReplayBuffer, PrioritizedMlpReplayBuffer

class Runner(object):

    def __init__(self, config):
        self.config = config
        # state varibles
        self.total_steps = 0
        self.total_episodes = 0
        self.total_update = 0
        self.last_train_T = 0
        self.last_save_T = 0
        self.last_eval_T = 0
        self.last_log_T = 0
        self.last_hard_update_T = 0
        self.train_iters = self.config['env_steps'] / self.config['train_interval']
        # training param
        self.policy_ids = sorted(config['policy_info'].keys())
        self.agent_ids = [range(self.config['num_agents'])]

        # save path
        self.fullpath = os.path.dirname(os.path.abspath(__file__)) + '/' + config['save_path']+ '_'+config['scenario_name']+'_' + config['algorithm_name']+'_'+config['experiment_index']
        self.writter = SummaryWriter(self.fullpath+'/logs')

        # network and functions
        self.trainer = MADDPG(config)
        self.policies = {p_id: MADDPGPolicy(config, p_id) for p_id in self.policy_ids}
        # Q: Why use policy mapping function?
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if config['policy_mapping_fn'](agent_id) == policy_id]) for policy_id in self.policies.keys()}
        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}
        if self.config['restore']: 
            self.restore()
        self.collecter = self.shared_collect_rollout if self.config['share_policy'] else self.seperated_collect_rollout

        # replay buffer
        self.beta_anneal = DecayThenFlatSchedule(self.config['per_beta_start'], 1.0, self.train_iters, decay="linear") # to refine replay buffer
        if self.config['use_per']:
            self.buffer = PrioritizedMlpReplayBuffer(self.config['per_alpha'], self.config['policy_info'], self.policy_agents, self.config['buffer_size'], self.config['use_same_share_obs'], self.config['use_avail_acts'], self.config['use_reward_normalization'])
        else:
            self.buffer = MlpReplayBuffer(self.config['policy_info'], self.policy_agents, self.config['buffer_size'], self.config['use_same_share_obs'], self.config['use_avail_acts'], self.config['use_reward_normalization'])
        
        # fill replay buffer with random actions
        self.finish_first_train_reset = False
        num_warmup_episodes = max((int(self.config['batch_size']//self.config['episode_length']) + 1, self.config['num_random_episodes']))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    def run(self):
        # collect data
        self.trainer.prep_rollout()
        self.collecter(explore=True, training_episode=True, warmup=False)
        # save
        if (self.total_steps - self.last_save_T) / self.config['save_interval'] >= 1:
            self.save()
            self.last_save_T = self.total_steps
        # log
        if ((self.total_steps - self.last_log_T) / self.config['log_interval']) >= 1:
            self.log()
            self.last_log_T = self.total_steps
        # eval
        if self.config['use_eval'] and ((self.total_steps - self.last_eval_T) / self.config['eval_interval']) >= 1:
            self.eval()
            self.last_eval_T = self.total_steps
        return self.total_steps

    @torch.no_grad()
    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()
        eval_infos = {}
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.config['num_eval_episodes']):
            env_info = self.collecter( explore=False, training_episode=False, warmup=False)
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")
    
    def batch_train(self):
        """Do a gradient update for all policies."""
        self.trainer.prep_training()
        self.train_infos = []
        update_actor = True
        for p_id in self.policy_ids:
            if self.config['use_per']:
                beta = self.beta_anneal.eval(self.total_update)
                sample = self.buffer.sample(self.config['batch_size'], beta, p_id)
            else:
                sample = self.buffer.sample(self.config['batch_size'])

            update = self.trainer.shared_train_policy_on_batch if self.config['use_same_share_obs'] else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update(p_id, sample)
            update_actor = train_info['update_actor']

            if self.config['use_per']:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_infos.append(train_info)
        raise NotImplementedError
    
    def save(self):
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.config['model_path'] + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(), critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.config['model_path'] + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(), actor_save_path + '/actor.pt')

    def restore(self):
        for pid in self.policy_ids:
            path = self.config['model_path'] + '/' + str(pid)
            print("load the pretrained model from {}".format(path))
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    @torch.no_grad()
    def warmup(self, num_warmup_episodes):
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range(int(num_warmup_episodes // self.config['num_envs']) + 1):
            env_info = self.collecter(explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(env_info['average_step_rewards'])
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average step rewards: {}".format(warmup_reward))

    def log(self):
        """See parent class."""
        end = time.time()
        print("\n total num timesteps {}/{}={}%, FPS {}.\n"
              .format(self.total_steps,
                      self.config['num_env_steps'],
                      self.total_steps/self.config['num_env_steps'] *100,
                      int(self.total_steps / (end - self.start))))
        # for p_id, train_info in zip(self.policy_ids, self.train_infos):
        #     self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        self.env_infos = {}
        self.env_infos['average_episode_rewards'] = []

    def log_env(self, env_info, suffix=None):
        for k, v in env_info.items():
            if len(v) > 0:
                v = np.mean(v)
                suffix_k = k if suffix is None else suffix + k 
                print(suffix_k + " is " + str(v))
                self.writter.add_scalars(suffix_k, {suffix_k: v}, self.total_steps)

    def log_train(self, policy_id, train_info):
        for k, v in train_info.items():
            policy_k = str(policy_id) + '/' + k
            self.writter.add_scalars(policy_k, {policy_k: v}, self.total_steps)

    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. All agents share a single policy. Do training steps when appropriate
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        # parameter setup
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.config['env'] if explore else self.config['eval_env']
        n_rollout_threads = self.config['num_envs'] if explore else self.config['num_eval_envs']

        if not explore: # no need to record
            obs = env.reset()
            share_obs = obs.reshape(n_rollout_threads, -1)
        else:
            if self.finish_first_train_reset: # orinary
                obs = self.obs
                share_obs = self.share_obs
            else: # reset in the first train
                obs = env.reset()
                share_obs = obs.reshape(n_rollout_threads, -1)
                self.finish_first_train_reset = True

        # init
        episode_rewards = []
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        valid_transition = {}
        step_avail_acts = {}
        step_next_avail_acts = {}
        # start collect
        for step in range(self.config['episode_length']):
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, _ = policy.get_actions(obs_batch, t_env=self.total_steps, explore=explore)
            # detach action to step env
            if not isinstance(acts_batch, np.ndarray):
                acts_batch = acts_batch.cpu().detach().numpy()
            env_acts = np.split(acts_batch, n_rollout_threads)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)
            # reset when done
            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs = env.reset()

            if not explore and np.all(dones_env):
                average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
                env_info['average_episode_rewards'] = average_episode_rewards
                return env_info

            next_share_obs = next_obs.reshape(n_rollout_threads, -1)
            # store varibles
            step_obs[p_id] = obs
            step_share_obs[p_id] = share_obs
            step_acts[p_id] = env_acts
            step_rewards[p_id] = rewards
            step_next_obs[p_id] = next_obs
            step_next_share_obs[p_id] = next_share_obs
            step_dones[p_id] = np.zeros_like(dones)
            step_dones_env[p_id] = dones_env
            valid_transition[p_id] = np.ones_like(dones)
            step_avail_acts[p_id] = None
            step_next_avail_acts[p_id] = None

            obs = next_obs
            share_obs = next_share_obs

            if explore:
                self.obs = obs
                self.share_obs = share_obs
                # push all episodes collected in this rollout step to the buffer
                self.buffer.insert(n_rollout_threads,
                                   step_obs,
                                   step_share_obs,
                                   step_acts,
                                   step_rewards,
                                   step_next_obs,
                                   step_next_share_obs,
                                   step_dones,
                                   step_dones_env,
                                   valid_transition,
                                   step_avail_acts,
                                   step_next_avail_acts)

            # training process records and train if needed
            if training_episode:
                self.total_steps += n_rollout_threads
                if (self.last_train_T == 0 or ((self.total_steps - self.last_train_T) / self.config['train_interval']) >= 1):
                    self.batch_train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_steps
            
        average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards

        return env_info

    def seperated_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        env = self.env if explore else self.eval_env
        n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        if not explore:
            obs = env.reset()
            share_obs = []
            for o in obs:
                share_obs.append(list(chain(*o)))
            share_obs = np.array(share_obs)
        else:
            if self.finish_first_train_reset:
                obs = self.obs
                share_obs = self.share_obs
            else:
                obs = env.reset()
                share_obs = []
                for o in obs:
                    share_obs.append(list(chain(*o)))
                share_obs = np.array(share_obs)
                self.finish_first_train_reset = True

        agent_obs = []
        for agent_id in range(self.num_agents):
            env_obs = []
            for o in obs:
                env_obs.append(o[agent_id])
            env_obs = np.array(env_obs)
            agent_obs.append(env_obs)

        # [agents, parallel envs, dim]
        episode_rewards = []
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        valid_transition = {}
        step_avail_acts = {}
        step_next_avail_acts = {}

        acts = []
        for p_id in self.policy_ids:
            if is_multidiscrete(self.policy_info[p_id]['act_space']):
                self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
            else:
                self.sum_act_dim = self.policy_act_dim[p_id]
            temp_act = np.zeros((n_rollout_threads, self.sum_act_dim))
            acts.append(temp_act)

        for step in range(self.episode_length):
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    act, _ = policy.get_actions(agent_obs[agent_id],
                                                t_env=self.total_env_steps,
                                                explore=explore)

                if not isinstance(act, np.ndarray):
                    act = act.cpu().detach().numpy()
                acts[agent_id] = act

            env_acts = []
            for i in range(n_rollout_threads):
                env_act = []
                for agent_id in range(self.num_agents):
                    env_act.append(acts[agent_id][i])
                env_acts.append(env_act)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)

            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs = env.reset()

            if not explore and np.all(dones_env):
                average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
                env_info['average_episode_rewards'] = average_episode_rewards
                return env_info

            next_share_obs = []
            for no in next_obs:
                next_share_obs.append(list(chain(*no)))
            next_share_obs = np.array(next_share_obs)

            next_agent_obs = []
            for agent_id in range(self.num_agents):
                next_env_obs = []
                for no in next_obs:
                    next_env_obs.append(no[agent_id])
                next_env_obs = np.array(next_env_obs)
                next_agent_obs.append(next_env_obs)

            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                step_obs[p_id] = np.expand_dims(agent_obs[agent_id], axis=1)
                step_share_obs[p_id] = share_obs
                step_acts[p_id] = np.expand_dims(acts[agent_id], axis=1)
                step_rewards[p_id] = np.expand_dims(rewards[:, agent_id], axis=1)
                step_next_obs[p_id] = np.expand_dims(next_agent_obs[agent_id], axis=1)
                step_next_share_obs[p_id] = next_share_obs
                step_dones[p_id] = np.zeros_like(np.expand_dims(dones[:, agent_id], axis=1))
                step_dones_env[p_id] = dones_env
                valid_transition[p_id] = np.ones_like(np.expand_dims(dones[:, agent_id], axis=1))
                step_avail_acts[p_id] = None
                step_next_avail_acts[p_id] = None

            obs = next_obs
            agent_obs = next_agent_obs
            share_obs = next_share_obs

            if explore:
                self.obs = obs
                self.share_obs = share_obs
                self.buffer.insert(n_rollout_threads,
                                   step_obs,
                                   step_share_obs,
                                   step_acts,
                                   step_rewards,
                                   step_next_obs,
                                   step_next_share_obs,
                                   step_dones,
                                   step_dones_env,
                                   valid_transition,
                                   step_avail_acts,
                                   step_next_avail_acts)

            # train
            if training_episode:
                self.total_env_steps += n_rollout_threads
                if (self.last_train_T == 0 or ((self.total_env_steps - self.last_train_T) / self.train_interval) >= 1):
                    self.batch_train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps

        average_episode_rewards = np.mean(np.sum(episode_rewards, axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards

        return env_info