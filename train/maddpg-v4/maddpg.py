import torch
import numpy as np
import itertools
import torch.nn as nn

from utils import PopArt, MLPBase, ACTLayer, DecayThenFlatSchedule, to_torch

class MADDPG:
    def __init__(self, config, policies):
        """
        Trainer class for MADDPG. See parent class for more information.
        """
        self.config = config
        self.policies = policies
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.config['num_agents']) if self.config['policy_mapping_fn'](agent_id) == policy_id]) for policy_id in
            self.policies.keys()}
        if self.config['use_popart']:
            self.value_normalizer = {policy_id: PopArt(1) for policy_id in self.policies.keys()}
        self.num_updates = {p_id : 0 for p_id in self.policy_ids} # update record

    def get_update_info(self, update_policy_id, obs_batch, act_batch, nobs_batch, navail_act_batch):
        """
        Form centralized observation and action info for current and next timestep.
        :param update_policy_id: (str) id of policy being updated.
        :param obs_batch: (np.ndarray) batch of observation sequences sampled from buffer.
        :param act_batch: (np.ndarray) batch of action sequences sampled from buffer.
        :param avail_act_batch: (np.ndarray) batch of available action sequences sampled from buffer. None if environment does not limit actions.

        :return cent_act: (list) list of action sequences corresponding to each agent.
        :return replace_ind_start: (int) index of act_sequences from which to replace actions for actor update.
        :return cent_nact: (np.ndarray) batch of centralize next step actions.
        """
        cent_act = []
        cent_nact = []
        replace_ind_start = None

        # iterate through policies to get the target acts and other centralized info
        ind = 0
        for p_id in self.policy_ids:
            batch_size = obs_batch[p_id].shape[1]
            policy = self.policies[p_id]
            if p_id == update_policy_id:
                replace_ind_start = ind # the index of action of this policy which need to be replaced in batch
            num_pol_agents = len(self.policy_agents[p_id]) # number of agents to share the policy
            cent_act.append(list(act_batch[p_id])) # concatatenate all actions 
            # combine all obs acts
            combined_nobs_batch = np.concatenate(nobs_batch[p_id], axis=0) 
            if navail_act_batch[p_id] is not None:
                combined_navail_act_batch = np.concatenate(navail_act_batch[p_id], axis=0)
            else:
                combined_navail_act_batch = None
            # use target actor to get next step actions
            with torch.no_grad():
                pol_nact, _ = policy.get_actions(combined_nobs_batch, combined_navail_act_batch, use_target=True) # get next action
                ind_agent_nacts = pol_nact.cpu().split(split_size=batch_size, dim=0)
            # cat to form the centralized next step actions
            cent_nact.append(torch.cat(ind_agent_nacts, dim=-1))

            ind += num_pol_agents

        cent_act = list(itertools.chain.from_iterable(cent_act))
        cent_nact = np.concatenate(cent_nact, axis=-1)

        return cent_act, replace_ind_start, cent_nact

    def train_policy_on_batch(self, update_policy_id, batch):
        if self.use_same_share_obs:
            return self.shared_train_policy_on_batch(update_policy_id, batch)
        else:
            return self.cent_train_policy_on_batch(update_policy_id, batch)

    def shared_train_policy_on_batch(self, update_policy_id, batch):
        """
        Training function when all agents share the same centralized observation. See train_policy_on_batch.
        """
        update_policy = self.policies[update_policy_id]
        train_info = {}
        # preprocess data
        obs_batch, cent_obs_batch, \
        act_batch, rew_batch, \
        nobs_batch, cent_nobs_batch, \
        dones_batch, dones_env_batch, valid_transition_batch,\
        avail_act_batch, navail_act_batch, \
        importance_weights, idxes = batch
        cent_act, replace_ind_start, cent_nact = self.get_update_info(update_policy_id, batch['obs_batch'], batch['act_batch'], batch['nobs_batch'], batch['navail_act_batch']) # concatatenate all observations
        batch_size = cent_obs.shape[0] # concate actions

        # update critic
        with torch.no_grad():
            next_step_Qs = update_policy.target_critic(cent_nobs, cent_nact)

        next_step_Q = torch.cat(next_step_Qs, dim=-1)
        # take min to prevent overestimation bias
        next_step_Q, _ = torch.min(next_step_Q, dim=-1, keepdim=True)

        rewards = to_torch(rewards).to(**self.tpdv).view(-1, 1)
        dones_env = to_torch(dones_env).to(**self.tpdv).view(-1, 1)

        if self.use_popart:
            target_Qs = rewards + self.config['gamma'] * (1 - dones_env) * self.value_normalizer[p_id].denormalize(next_step_Q)
            target_Qs = self.value_normalizer[p_id](target_Qs)
        else:
            target_Qs = rewards + self.config['gamma'] * (1 - dones_env) * next_step_Q

        predicted_Qs = update_policy.critic(cent_obs, np.concatenate(cent_act, axis=-1))

        update_policy.critic_optimizer.zero_grad()

        errors = [target_Qs.detach() - predicted_Q for predicted_Q in predicted_Qs]
        if self.use_per:
            importance_weights = to_torch(importance_weights).to(**self.tpdv)
            if self.use_huber_loss:
                critic_loss = [huber_loss(error, self.huber_delta).flatten() for error in errors]
            else:
                critic_loss = [mse_loss(error).flatten() for error in errors]
            # weight each loss element by their importance sample weight
            critic_loss = [(loss * importance_weights).mean() for loss in critic_loss]
            critic_loss = torch.stack(critic_loss).sum(dim=0)
            # new priorities are TD error
            new_priorities = np.stack([error.abs().cpu().detach().numpy().flatten() for error in errors]).mean(axis=0) + self.per_eps
        else:
            if self.use_huber_loss:
                critic_loss = [huber_loss(error, self.huber_delta).mean() for error in errors]
            else:
                critic_loss = [mse_loss(error).mean() for error in errors]
            critic_loss = torch.stack(critic_loss).sum(dim=0)
            new_priorities = None
        # update
        critic_loss.backward()
        # optimize
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.critic.parameters(), self.args.max_grad_norm)
        update_policy.critic_optimizer.step()

        train_info['critic_loss'] = critic_loss
        train_info['critic_grad_norm'] = critic_grad_norm

        # update actor
        update_actor = (self.num_updates[update_policy_id] % self.config['actor_update_interval'] == 0)
        if update_actor:
            self.update_actor(update_policy_id, batch)
                        # freeze Q-networks
            for p in update_policy.critic.parameters():
                p.requires_grad = False

            num_update_agents = len(self.policy_agents[update_policy_id])
            mask_temp = []
            for p_id in self.policy_ids:
                if isinstance(self.policies[p_id].act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(self.policies[p_id].act_dim))
                else:
                    sum_act_dim = self.policies[p_id].act_dim
                for _ in self.policy_agents[p_id]:
                    mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))
            # create musks to choose a part of numbers to update
            masks = []
            valid_trans_mask = []
            # need to iterate through agents, but only formulate masks at each step
            # different environment use different masks
            for i in range(num_update_agents):
                curr_mask_temp = copy.deepcopy(mask_temp)
                # set the mask to 1 at locations where the action should come from the actor output
                if isinstance(update_policy.act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(update_policy.act_dim))
                else:
                    sum_act_dim = update_policy.act_dim
                curr_mask_temp[replace_ind_start + i] = np.ones(sum_act_dim, dtype=np.float32)
                curr_mask_vec = np.concatenate(curr_mask_temp)
                # expand this mask into the proper size
                curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
                masks.append(curr_mask)

                # agent valid transitions
                agent_valid_trans_batch = to_torch(valid_transition_batch[update_policy_id][i]).to(**self.tpdv)
                valid_trans_mask.append(agent_valid_trans_batch)
            # cat to form into tensors
            mask = to_torch(np.concatenate(masks)).to(**self.tpdv)
            valid_trans_mask = torch.cat(valid_trans_mask, dim=0)
            pol_agents_obs_batch = np.concatenate(obs_batch[update_policy_id], axis=0)
            if avail_act_batch[update_policy_id] is not None:
                pol_agents_avail_act_batch = np.concatenate(avail_act_batch[update_policy_id], axis=0)
            else:
                pol_agents_avail_act_batch = None
            # get all actions from actor
            pol_acts, _ = update_policy.get_actions(pol_agents_obs_batch, pol_agents_avail_act_batch, use_gumbel=True)
            # separate into individual agent batches
            agent_actor_batches = pol_acts.split(split_size=batch_size, dim=0)

            cent_act = list(map(lambda arr: to_torch(arr).to(**self.tpdv), cent_act))

            actor_cent_acts = copy.deepcopy(cent_act)
            for i in range(num_update_agents):
                actor_cent_acts[replace_ind_start + i] = agent_actor_batches[i]

            actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat((num_update_agents, 1))
            # convert buffer acts to torch, formulate centralized buffer action and repeat as done above
            buffer_cent_acts = torch.cat(cent_act, dim=-1).repeat(num_update_agents, 1)

            # also repeat cent obs
            stacked_cent_obs = np.tile(cent_obs, (num_update_agents, 1))

            # combine the buffer cent acts with actor cent acts and pass into buffer
            # [Q] a kind of like combine on and off policy?
            actor_update_cent_acts = mask * actor_cent_acts + (1 - mask) * buffer_cent_acts
            actor_Qs = update_policy.critic(stacked_cent_obs, actor_update_cent_acts)
            # use only the first Q output for actor loss
            actor_Qs = actor_Qs[0]
            actor_Qs = actor_Qs * valid_trans_mask
            actor_loss = -(actor_Qs).sum() / (valid_trans_mask).sum()

            update_policy.critic_optimizer.zero_grad()
            update_policy.actor_optimizer.zero_grad()
            actor_loss.backward()

            actor_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.actor.parameters(), self.args.max_grad_norm)
            update_policy.actor_optimizer.step()

            for p in update_policy.critic.parameters():
                p.requires_grad = True

            train_info['actor_loss'] = actor_loss # Q-value
            train_info['actor_grad_norm'] = actor_grad_norm # clip actor parameters
            train_info['update_actor'] = update_actor # if actor is updated

        return train_info, new_priorities, idxes # new replay buffer priorities and indexes

    def update_actor(self, update_policy_id, batch):
        update_policy = self.policies[update_policy_id]
        # freeze Q
        for p in update_policy.critic.parameters():
            p.requires_grad = False
        # 
        mask_temp = [] # [Q] what dose musk used for?
        for p_id in self.policy_ids:
            if isinstance(self.policies[p_id].act_dim, np.ndarray):
                # multidiscrete case
                sum_act_dim = int(sum(self.policies[p_id].act_dim))
            else:
                sum_act_dim = self.policies[p_id].act_dim
            for _ in self.policy_agents[p_id]:
                mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))
        # compute loss
        actor_Qs = update_policy.critic(stacked_cent_obs, actor_update_cent_acts)
        actor_Qs = actor_Qs[0] # choose first Q to use
        actor_loss = -(actor_Qs).sum() / (valid_trans_mask).sum()
        # update it.
        update_policy.critic_optimizer.zero_grad()
        update_policy.actor_optimizer.zero_grad()
        actor_loss.backward()

        # unfreeze Q
        for p in update_policy.critic.parameters():
            p.requires_grad = True
    
    def cent_train_policy_on_batch(self, update_policy_id, batch):
        """Training function when each agent has its own centralized observation. See train_policy_on_batch."""
        obs_batch, cent_obs_batch, \
        act_batch, rew_batch, \
        nobs_batch, cent_nobs_batch, \
        dones_batch, dones_env_batch, valid_transition_batch,\
        avail_act_batch, navail_act_batch, \
        importance_weights, idxes = batch

        train_info = {}

        update_actor = self.num_updates[update_policy_id] % self.actor_update_interval == 0

        cent_act, replace_ind_start, cent_nact = self.get_update_info(
            update_policy_id, obs_batch, act_batch, nobs_batch, navail_act_batch)

        cent_obs = cent_obs_batch[update_policy_id]
        cent_nobs = cent_nobs_batch[update_policy_id]
        rewards = rew_batch[update_policy_id][0]
        dones_env = dones_env_batch[update_policy_id]
        dones = dones_batch[update_policy_id]
        valid_trans = valid_transition_batch[update_policy_id]

        update_policy = self.policies[update_policy_id]
        batch_size = obs_batch[update_policy_id].shape[1]

        num_update_agents = len(self.policy_agents[update_policy_id])

        all_agent_cent_obs = np.concatenate(cent_obs, axis=0)
        all_agent_cent_nobs = np.concatenate(cent_nobs, axis=0)
        # since this is the same for each agent, just repeat when stacking
        cent_act_buffer = np.concatenate(cent_act, axis=-1)
        all_agent_cent_act_buffer = np.tile(cent_act_buffer, (num_update_agents, 1))
        all_agent_cent_nact = np.tile(cent_nact, (num_update_agents, 1))
        all_env_dones = np.tile(dones_env, (num_update_agents, 1))
        all_agent_rewards = np.tile(rewards, (num_update_agents, 1))

        # critic update
        update_policy.critic_optimizer.zero_grad()
        all_agent_rewards = to_torch(all_agent_rewards).to(**self.tpdv).reshape(-1, 1)
        all_env_dones = to_torch(all_env_dones).to(**self.tpdv).reshape(-1, 1)
        all_agent_valid_trans = to_torch(valid_trans).to(**self.tpdv).reshape(-1, 1)
        # critic update
        with torch.no_grad():
            next_step_Q = update_policy.target_critic(all_agent_cent_nobs, all_agent_cent_nact).reshape(-1, 1)
        if self.use_popart:
            target_Qs = all_agent_rewards + self.config['gamma'] * (1 - all_env_dones) * \
                self.value_normalizer[p_id].denormalize(next_step_Q)
            target_Qs = self.value_normalizer[p_id](target_Qs)
        else:
            target_Qs = all_agent_rewards + self.config['gamma'] * (1 - all_env_dones) * next_step_Q
        predicted_Qs = update_policy.critic(all_agent_cent_obs, all_agent_cent_act_buffer).reshape(-1, 1)

        error = target_Qs.detach() - predicted_Qs
        if self.use_per:
            agent_importance_weights = np.tile(importance_weights, num_update_agents)
            agent_importance_weights = to_torch(agent_importance_weights).to(**self.tpdv)
            if self.use_huber_loss:
                critic_loss = huber_loss(error, self.huber_delta).flatten()
            else:
                critic_loss = mse_loss(error).flatten()
            # weight each loss element by their importance sample weight
            critic_loss = critic_loss * agent_importance_weights
            if self.use_value_active_masks:
                critic_loss = (critic_loss.view(-1, 1) * (all_agent_valid_trans)).sum() / (all_agent_valid_trans).sum()
            else:
                critic_loss = critic_loss.mean()
            # new priorities are TD error
            agent_new_priorities = error.abs().cpu().detach().numpy().flatten()
            new_priorities = np.mean(np.split(agent_new_priorities, num_update_agents), axis=0) + self.per_eps
        else:
            if self.use_huber_loss:
                critic_loss = huber_loss(error, self.huber_delta)
            else:
                critic_loss = mse_loss(error)

            if self.use_value_active_masks:
                critic_loss = (critic_loss * (all_agent_valid_trans)).sum() / (all_agent_valid_trans).sum()
            else:
                critic_loss = critic_loss.mean()
            new_priorities = None

        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.critic.parameters(),
                                                                 self.args.max_grad_norm)
        update_policy.critic_optimizer.step()

        train_info['critic_loss'] = critic_loss
        train_info['critic_grad_norm'] = critic_grad_norm

        # actor update
        if update_actor:
            for p in update_policy.critic.parameters():
                p.requires_grad = False

            num_update_agents = len(self.policy_agents[update_policy_id])
            mask_temp = []
            for p_id in self.policy_ids:
                if isinstance(self.policies[p_id].act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(self.policies[p_id].act_dim))
                else:
                    sum_act_dim = self.policies[p_id].act_dim
                for _ in self.policy_agents[p_id]:
                    mask_temp.append(np.zeros(sum_act_dim, dtype=np.float32))

            masks = []
            valid_trans_mask = []
            # need to iterate through agents, but only formulate masks at each step
            for i in range(num_update_agents):
                curr_mask_temp = copy.deepcopy(mask_temp)
                # set the mask to 1 at locations where the action should come from the actor output
                if isinstance(update_policy.act_dim, np.ndarray):
                    # multidiscrete case
                    sum_act_dim = int(sum(update_policy.act_dim))
                else:
                    sum_act_dim = update_policy.act_dim
                curr_mask_temp[replace_ind_start + i] = np.ones(sum_act_dim, dtype=np.float32)
                curr_mask_vec = np.concatenate(curr_mask_temp)
                # expand this mask into the proper size
                curr_mask = np.tile(curr_mask_vec, (batch_size, 1))
                masks.append(curr_mask)

                # agent valid transitions
                agent_valid_trans_batch = to_torch(valid_transition_batch[update_policy_id][i]).to(**self.tpdv)
                valid_trans_mask.append(agent_valid_trans_batch)
            # cat to form into tensors
            mask = to_torch(np.concatenate(masks)).to(**self.tpdv)
            valid_trans_mask = torch.cat(valid_trans_mask, dim=0)

            pol_agents_obs_batch = np.concatenate(obs_batch[update_policy_id], axis=0)
            if avail_act_batch[update_policy_id] is not None:
                pol_agents_avail_act_batch = np.concatenate(avail_act_batch[update_policy_id], axis=0)
            else:
                pol_agents_avail_act_batch = None
            # get all actions from actor
            pol_acts, _ = update_policy.get_actions(pol_agents_obs_batch, pol_agents_avail_act_batch, use_gumbel=True)
            # separate into individual agent batches
            agent_actor_batches = pol_acts.split(split_size=batch_size, dim=0)
            # cat along final dim to formulate centralized action and stack copies of the batch
            cent_act = list(map(lambda arr: to_torch(arr).to(**self.tpdv), cent_act))
            actor_cent_acts = copy.deepcopy(cent_act)
            for i in range(num_update_agents):
                actor_cent_acts[replace_ind_start + i] = agent_actor_batches[i]

            actor_cent_acts = torch.cat(actor_cent_acts, dim=-1).repeat((num_update_agents, 1))

            # combine the buffer cent acts with actor cent acts and pass into buffer
            actor_update_cent_acts = mask * actor_cent_acts + (1 - mask) * to_torch(all_agent_cent_act_buffer).to(**self.tpdv)
            actor_Qs = update_policy.critic(all_agent_cent_obs, actor_update_cent_acts)
            # actor_loss = -actor_Qs.mean()
            actor_Qs = actor_Qs * valid_trans_mask
            actor_loss = -(actor_Qs).sum() / (valid_trans_mask).sum()

            update_policy.critic_optimizer.zero_grad()
            update_policy.actor_optimizer.zero_grad()
            actor_loss.backward()

            actor_grad_norm = torch.nn.utils.clip_grad_norm_(update_policy.actor.parameters(),
                                                                    self.args.max_grad_norm)
            update_policy.actor_optimizer.step()

            for p in update_policy.critic.parameters():
                p.requires_grad = True

            train_info['actor_loss'] = actor_loss
            train_info['actor_grad_norm'] = actor_grad_norm

        return train_info, new_priorities, idxes

    def prep_training(self):
        for policy in self.policies.values():
            policy.actor.train()
            policy.critic.train()
            policy.target_actor.train()
            policy.target_critic.train()
    
    def prep_rollout(self):
        """Sets all networks to eval mode."""
        for policy in self.policies.values():
            policy.actor.eval()
            policy.critic.eval()
            policy.target_actor.eval()
            policy.target_critic.eval()
    
class MADDPGPolicy:
    """
    Compute actions using the needed information.
    """
    def __init__(self, config, p_id ,train=True):
        # parameters
        self.config = config
        self.policy_config = self.config['policy_config'][p_id]
        # Actor-Critic
        self.actor = MADDPG_Actor(self.config, self.policy_config['obs_dim'], self.policy_config['act_dim'])
        self.critic = MADDPG_Critic(self.config, self.policy_config['central_obs_dim'], self.config['central_act_dim'])
        self.actor = MADDPG_Actor(self.config, self.policy_config['obs_dim'], self.policy_config['act_dim'])
        self.critic = MADDPG_Critic(self.config, self.policy_config['central_obs_dim'], self.config['central_act_dim'])
        # sync the target weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        if train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

            if self.config['discrete']:
                # eps greedy exploration
                self.exploration = DecayThenFlatSchedule(config['epsilon_start'], config['epsilon_finish'], config['epsilon_anneal_time'], decay="linear")

    def get_actions(self, obs, available_actions=None, t_env=None, explore = False, use_target = False):
        eps = None
        # get actions
        if use_target:
            actor_out = self.target_actor(obs)
        else:
            actor_out = self.actor(obs)
        # wrap acitons
        if self.config['discrete']:
            raise NotImplementedError
        else :
            if explore:
                actions = torch.empty(actor_out.shape).normal_(mean=0, std=self.config['act_noise_std']) + actor_out
            elif use_target and self.config['target_noise']:
                actions = gaussian_notorch.empty(actor_out.shape).normal_(mean=0, std=self.config['act_noise_std']) + actor_out
            else:
                actions = actor_out
        return actions, eps

    def get_random_actions(self, obs, avaliable_actions = None):
        batch_size = obs.shape[0]
        if self.config['discrete']:
            raise NotImplementedError
        else: 
            random_actions = np.random.uniform(self.policy_config['act_space'].low, self.policy_config['act_space'].high, size=(batch_size, self.policy_config['act_dim']))
        return random_actions

    def soft_target_updates(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config['tau']) + param.data * self.cofig['tau'])
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config['tau']) + param.data * self.cofig['tau'])

    def hard_target_updates(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

class MADDPG_Actor(nn.Module):
    def __init__(self, config, obs_dim, act_dim):
        super(MADDPG_Actor, self).__init__()
        self.config = config
        self.mlp = MLPBase(config, obs_dim)
        self.act = ACTLayer(self.config, act_dim)
        self.to(config['device'])

    def forward(self, obs):
        x = torch.from_numpy(obs) if type(obs) == np.ndarray else obs
        x = x.to(**self.config['tpdv'])
        x = self.mlp(x)
        return self.act(x)

class MADDPG_Critic(nn.Module):
    def __init__(self, config, central_obs_dim, central_act_dim, num_q_outs=1):
        super(MADDPG_Critic, self).__init__()
        input_dim = central_act_dim + central_obs_dim
        self.mlp = MLPBase(config, input_dim)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.q_outs = [init_(nn.Linear(self.hidden_size, 1)) for _ in range(num_q_outs)]
        self.to(config['device'])

    def forward(self, central_obs, central_act):
        central_obs = torch.from_numpy(central_obs) if type(central_obs) == np.ndarray else central_obs
        central_obs = central_obs.to(**self.config['tpdv'])
        central_act = torch.from_numpy(central_act) if type(central_act) == np.ndarray else central_act
        central_act = central_act.to(**self.config['tpdv'])

        x = torch.cat([central_obs, central_act], dim=1)

        x = self.mlp(x)
        q_values = [q_out(x) for q_out in self.q_outs]

        return q_values
        