import numpy as np

class MlpReplayBuffer(object):
    def __init__(self, policy_info, policy_agents, buffer_size, use_same_share_obs, use_avail_acts,
                 use_reward_normalization=False):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def insert(self, num_insert_steps, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env, valid_transition,
               avail_acts, next_avail_acts):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

class PrioritizedMlpReplayBuffer(MlpReplayBuffer):
    def __init__(self, alpha, policy_info, policy_agents, buffer_size, use_same_share_obs, use_avail_acts,
                 use_reward_normalization=False):
        raise NotImplementedError

    def insert(self, num_insert_steps, obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env,
               valid_transition, avail_acts=None, next_avail_acts=None):
        raise NotImplementedError

    def _sample_proportional(self, batch_size, p_id=None):
        raise NotImplementedError

    def sample(self, batch_size, beta=0, p_id=None):
        raise NotImplementedError

    def update_priorities(self, idxes, priorities, p_id=None):
        raise NotImplementedError
        