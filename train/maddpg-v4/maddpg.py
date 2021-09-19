import torch
import numpy as np

class MADDPG:
    def __init__(self, config):
        raise NotImplementedError

    def get_update_info():
        raise NotImplementedError

    def train_policy_on_batch(self, update_policy_id, batch):
        """
        Performs a gradient update for the specified policy using a batch of sampled data.
        :param update_policy_id: (str) id of policy to update.
        :param batch: (Tuple) batch of data sampled from buffer. Batch contains observations, global observations,
                      actions, rewards, terminal states, available actions, and priority weights (for PER)
        """
        raise NotImplementedError

    def shared_train_policy_on_batch(self):
        raise NotImplementedError

    def cent_train_policy_on_batch(self):
        raise NotImplementedError

    def prep_training(self):
        """Sets all networks to training mode."""
        raise NotImplementedError
    
    def prep_rollout(self):
        """Sets all networks to eval mode."""
        raise NotImplementedError
    

class MADDPGPolicy:
    def __init__(self, config):
        raise NotImplementedError