import os
from tensorboardX import SummaryWriter

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
        self.policies = {p_id: MADDPGPolicy(config) for p_id in self.policy_ids}
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

        raise NotImplementedError

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
        for p_id in self.policy_ids:
            if self.config['use_per']:
                beta = self.beta_anneal.eval(self.total_update)
                sample = self.buffer.sample(self.config['batch_size'], beta, p_id)
            else:
                sample = self.buffer.sample(self.config['batch_size'])

            update = self.trainer.shared_train_policy_on_batch if self.use_same_share_obs else self.trainer.cent_train_policy_on_batch
            
            train_info, new_priorities, idxes = update(p_id, sample)
            update_actor = train_info['update_actor']

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)
        raise NotImplementedError
    
    def save(self):
        raise NotImplementedError

    def restore(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def log_clear(self):
        raise NotImplementedError

    def log_env(self, env_info, suffix=None):
        '''
        suffix is env identifier
        '''
        raise NotImplementedError

    def log_train(self, policy_id, train_info):
        raise NotImplementedError

    def shared_collect_rollout(self):
        
        raise NotImplementedError

    def seperated_collect_rollout(self):
        raise NotImplementedError