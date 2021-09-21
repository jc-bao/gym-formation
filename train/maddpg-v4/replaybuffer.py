import numpy as np

from utils import get_dim_from_space

class MlpReplayBuffer(object):
    def __init__(self, policy_info, policy_agents, buffer_size, use_same_share_obs, use_avail_acts,
                 use_reward_normalization=False):
        self.policy_info = policy_info
        self.policy_buffer = {
            p_id: MlpReplayBuffer(
                buffer_size,
                len(policy_agents[p_id]), 
                policy_info[p_id]['obs_space'],
                policy_info[p_id]['shared_obs_space'], 
                policy_info[p_id]['act_space'],
                use_same_share_obs,
                use_avail_acts,
                use_reward_normalization
            )
            for p_id in self.policy_info.keys()
        }
        raise NotImplementedError

    def __len__(self):
        return self.policy_buffers['policy_0'].filled_i

    def insert(self, num_insert_steps, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env, valid_transition,
               avail_acts, next_avail_acts):
        idx_range = None
        for p_id in self.policy_info.keys():
            idx_range = self.policy_buffers[p_id].insert(num_insert_steps,
                                                         np.array(obs[p_id]), np.array(share_obs[p_id]),
                                                         np.array(acts[p_id]), np.array(rewards[p_id]),
                                                         np.array(next_obs[p_id]), np.array(next_share_obs[p_id]),
                                                         np.array(dones[p_id]), np.array(dones_env[p_id]),
                                                         np.array(valid_transition[p_id]),
                                                         np.array(avail_acts[p_id]), np.array(next_avail_acts[p_id]))
        return idx_range

    def sample(self, batch_size):
        inds = np.random.choice(len(self), batch_size)
        obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, valid_transition, avail_acts, next_avail_acts = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            obs[p_id], share_obs[p_id], acts[p_id], rewards[p_id], next_obs[p_id], next_share_obs[p_id], \
            dones[p_id], dones_env[p_id], valid_transition[p_id], avail_acts[p_id], next_avail_acts[p_id] = \
            self.policy_buffers[p_id].sample_inds(inds)

        return obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, valid_transition, avail_acts, next_avail_acts, None, None

class PrioritizedMlpReplayBuffer(MlpReplayBuffer):
    def __init__(self, alpha, policy_info, policy_agents, buffer_size, use_same_share_obs, use_avail_acts,
                 use_reward_normalization=False):
        super(PrioritizedMlpReplayBuffer, self).__init__(policy_info, policy_agents,
                                                         buffer_size, use_same_share_obs, use_avail_acts,
                                                         use_reward_normalization)
        self.alpha = alpha
        self.policy_info = policy_info
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2
        self._it_sums = {p_id: SumSegmentTree(it_capacity) for p_id in self.policy_info.keys()}
        self._it_mins = {p_id: MinSegmentTree(it_capacity) for p_id in self.policy_info.keys()}
        self.max_priorities = {p_id: 1.0 for p_id in self.policy_info.keys()}

    def insert(self, num_insert_steps, obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env,
               valid_transition, avail_acts=None, next_avail_acts=None):
        idx_range = super().insert(num_insert_steps, obs, share_obs, acts, rewards, next_obs, next_share_obs, dones,
                                   dones_env, valid_transition, avail_acts, next_avail_acts)
        for idx in range(idx_range[0], idx_range[1]):
            for p_id in self.policy_info.keys():
                self._it_sums[p_id][idx] = self.max_priorities[p_id] ** self.alpha
                self._it_mins[p_id][idx] = self.max_priorities[p_id] ** self.alpha

        return idx_range

    def _sample_proportional(self, batch_size, p_id=None):
        total = self._it_sums[p_id].sum(0, len(self) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sums[p_id].find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0, p_id=None):
        assert len(self) > batch_size, "Not enough samples in the buffer!"
        assert beta > 0

        batch_inds = self._sample_proportional(batch_size, p_id)

        p_min = self._it_mins[p_id].min() / self._it_sums[p_id].sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self._it_sums[p_id][batch_inds] / self._it_sums[p_id].sum()
        weights = (p_sample * len(self)) ** (-beta) / max_weight

        obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, valid_transition, avail_acts, next_avail_acts = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            p_buffer = self.policy_buffers[p_id]
            obs[p_id], share_obs[p_id], acts[p_id], rewards[p_id], next_obs[p_id], next_share_obs[p_id], dones[p_id], \
            dones_env[p_id], valid_transition[p_id], avail_acts[p_id], next_avail_acts[p_id] = p_buffer.sample_inds(batch_inds)

        return obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, valid_transition, avail_acts, next_avail_acts, weights, batch_inds

    def update_priorities(self, idxes, priorities, p_id=None):
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self)

        self._it_sums[p_id][idxes] = priorities ** self.alpha
        self._it_mins[p_id][idxes] = priorities ** self.alpha

        self.max_priorities[p_id] = max(
            self.max_priorities[p_id], np.max(priorities))

class MlpPolicyBuffer(object):
    def __init__(self, buffer_size, num_agents, obs_space, share_obs_space, act_space, use_same_share_obs,
                 use_avail_acts, use_reward_normalization=False):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.use_same_share_obs = use_same_share_obs
        self.use_avail_acts = use_avail_acts
        self.use_reward_normalization = use_reward_normalization
        self.filled_i = 0
        self.current_i = 0

        # obs
        if obs_space.__class__.__name__ == 'Box':
            obs_shape = obs_space.shape
            share_obs_shape = share_obs_space.shape
        elif obs_space.__class__.__name__ == 'list':
            obs_shape = obs_space
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        self.obs = np.zeros(
            (self.buffer_size, self.num_agents, obs_shape[0]), dtype=np.float32)

        if self.use_same_share_obs:
            self.share_obs = np.zeros((self.buffer_size, share_obs_shape[0]), dtype=np.float32)
        else:
            self.share_obs = np.zeros((self.buffer_size, self.num_agents, share_obs_shape[0]), dtype=np.float32)
        
        self.next_obs = np.zeros_like(self.obs, dtype=np.float32)
        self.next_share_obs = np.zeros_like(self.share_obs, dtype=np.float32)

        # action
        act_dim = np.sum(get_dim_from_space(act_space))
        self.acts = np.zeros((self.buffer_size, self.num_agents, act_dim), dtype=np.float32)
        if self.use_avail_acts:
            self.avail_acts = np.ones_like(self.acts, dtype=np.float32)
            self.next_avail_acts = np.ones_like(self.avail_acts, dtype=np.float32)

        # rewards
        self.rewards = np.zeros((self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # default to done being True
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones((self.buffer_size, 1), dtype=np.float32)
        self.valid_transition = np.zeros_like(self.dones, dtype=np.float32) # if the agent is dead

    def __len__(self):
        return self.filled_i

    def insert(self, num_insert_steps, obs, share_obs, acts, rewards,
               next_obs, next_share_obs, dones, dones_env, valid_transition,
               avail_acts=None, next_avail_acts=None):
        assert obs.shape[0] == num_insert_steps, ("different size!")
        # calculate index
        if self.current_i + num_insert_steps <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_insert_steps)
        else:
            num_left_steps = self.current_i + num_insert_steps - self.buffer_size
            idx_range = np.concatenate((np.arange(self.current_i, self.buffer_size), np.arange(num_left_steps)))
        # store
        self.obs[idx_range] = obs.copy()
        self.share_obs[idx_range] = share_obs.copy()
        self.acts[idx_range] = acts.copy()
        self.rewards[idx_range] = rewards.copy()
        self.next_obs[idx_range] = next_obs.copy()
        self.next_share_obs[idx_range] = next_share_obs.copy()
        self.dones[idx_range] = dones.copy()
        self.dones_env[idx_range] = dones_env.copy()
        self.valid_transition[idx_range] = valid_transition.copy()
        if self.use_avail_acts:
            self.avail_acts[idx_range] = avail_acts.copy()
            self.next_avail_acts[idx_range] = next_avail_acts.copy()
        # update parameters
        self.current_i = idx_range[-1] + 1
        self.filled_i = min(self.filled_i + len(idx_range), self.buffer_size)

        return idx_range
        
    def sample(self, sample_inds):
        obs = _cast(self.obs[sample_inds])
        acts = _cast(self.acts[sample_inds])
        if self.use_reward_normalization:
            mean_reward = self.rewards[:self.filled_i].mean()
            std_reward = self.rewards[:self.filled_i].std()
            rewards = _cast(
                (self.rewards[sample_inds] - mean_reward) / std_reward)
        else:
            rewards = _cast(self.rewards[sample_inds])

        next_obs = _cast(self.next_obs[sample_inds])

        if self.use_same_share_obs:
            share_obs = self.share_obs[sample_inds]
            next_share_obs = self.next_share_obs[sample_inds]
        else:
            share_obs = _cast(self.share_obs[sample_inds])
            next_share_obs = _cast(self.next_share_obs[sample_inds])

        dones = _cast(self.dones[sample_inds])
        dones_env = self.dones_env[sample_inds]
        valid_transition = _cast(self.valid_transition[sample_inds])

        if self.use_avail_acts:
            avail_acts = _cast(self.avail_acts[sample_inds])
            next_avail_acts = _cast(self.next_avail_acts[sample_inds])
        else:
            avail_acts = None
            next_avail_acts = None

        return obs, share_obs, acts, rewards, next_obs, next_share_obs, dones, dones_env, valid_transition, avail_acts, next_avail_acts


        

def _cast(x):
    return x.transpose(1, 0, 2)
            
def unique(sorted_array):
    """
    More efficient implementation of np.unique for sorted arrays
    :param sorted_array: (np.ndarray)
    :return:(np.ndarray) sorted_array without duplicate elements
    """
    if len(sorted_array) == 1:
        return sorted_array
    left = sorted_array[:-1]
    right = sorted_array[1:]
    uniques = np.append(right != left, True)
    return sorted_array[uniques]


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array that supports Index arrays, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        :param capacity: (int) Total size of the array - must be a power of two.
        :param operation: (lambda (Any, Any): Any) operation for combining elements (eg. sum, max) must form a
            mathematical group together with the set of possible values for array elements (i.e. be associative)
        :param neutral_element: (Any) neutral element for the operation above. eg. float('-inf') for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (
            capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(
                        mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """
        Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        :param start: (int) beginning of the subsequence
        :param end: (int) end of the subsequences
        :return: (Any) result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # indexes of the leaf
        idxs = idx + self._capacity
        self._value[idxs] = val
        if isinstance(idxs, int):
            idxs = np.array([idxs])
        # go up one level in the tree and remove duplicate indexes
        idxs = unique(idxs // 2)
        while len(idxs) > 1 or idxs[0] > 0:
            # as long as there are non-zero indexes, update the corresponding values
            self._value[idxs] = self._operation(
                self._value[2 * idxs],
                self._value[2 * idxs + 1]
            )
            # go up one level in the tree and remove duplicate indexes
            idxs = unique(idxs // 2)

    def __getitem__(self, idx):
        assert np.max(idx) < self._capacity
        assert 0 <= np.min(idx)
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.add,
            neutral_element=0.0
        )
        self._value = np.array(self._value)

    def sum(self, start=0, end=None):
        """
        Returns arr[start] + ... + arr[end]
        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of SumSegmentTree
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum for each entry in prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        :param prefixsum: (np.ndarray) float upper bounds on the sum of array prefix
        :return: (np.ndarray) highest indexes satisfying the prefixsum constraint
        """
        if isinstance(prefixsum, float):
            prefixsum = np.array([prefixsum])
        assert 0 <= np.min(prefixsum)
        assert np.max(prefixsum) <= self.sum() + 1e-5
        assert isinstance(prefixsum[0], float)

        idx = np.ones(len(prefixsum), dtype=int)
        cont = np.ones(len(prefixsum), dtype=bool)

        while np.any(cont):  # while not all nodes are leafs
            idx[cont] = 2 * idx[cont]
            prefixsum_new = np.where(
                self._value[idx] <= prefixsum, prefixsum - self._value[idx], prefixsum)
            # prepare update of prefixsum for all right children
            idx = np.where(np.logical_or(
                self._value[idx] > prefixsum, np.logical_not(cont)), idx, idx + 1)
            # Select child node for non-leaf nodes
            prefixsum = prefixsum_new
            # update prefixsum
            cont = idx < self._capacity
            # collect leafs
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=np.minimum,
            neutral_element=float('inf')
        )
        self._value = np.array(self._value)

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])
        :param start: (int) start position of the reduction (must be >= 0)
        :param end: (int) end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: (Any) reduction of MinSegmentTree
        """
        return super(MinSegmentTree, self).reduce(start, end)

