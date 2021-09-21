import yaml
import os
from multiprocessing import Process, Pipe
import numpy as np
import torch
import torch.nn as nn

from gym.spaces import Box, Discrete, Tuple
from wrapper import DummyVecEnv, SubprocVecEnv
import formation_gym

def make_train_env(config):
    def get_env_fn(rank):
        def init_env():
            if config['env_name'] == "MPE":
                env = formation_gym.make_env(config['scenario_name'], benchmark = False, num_agents = config['num_agents'])
            else:
                print("Can not support the " +
                      config['env_name'] + "environment.")
                raise NotImplementedError
            env.seed(config['seed'] + rank * 1000)
            return env
        return init_env
    if config['n_rollout_threads'] == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(config['n_rollout_threads'])])

def to_torch(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input

def get_config():
    with open(os.path.dirname(__file__)+"/parameters.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = space.shape[0]
    elif isinstance(space, Discrete):
        dim = space.n
    elif isinstance(space, Tuple):
        dim = sum([get_dim_from_space(sp) for sp in space])
    elif "MultiDiscrete" in space.__class__.__name__:
        return (space.high - space.low) + 1
    elif isinstance(space, list):
        dim = space[0]
    else:
        raise Exception("Unrecognized space: ", type(space))
    return dim

def get_cent_act_dim(action_space):
    cent_act_dim = 0
    for space in action_space:
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            cent_act_dim += int(sum(dim))
        else:
            cent_act_dim += dim
    return cent_act_dim
    
def get_state_dim(observation_dict, action_dict):
    combined_obs_dim = sum([get_dim_from_space(space)
                            for space in observation_dict.values()])
    combined_act_dim = 0
    for space in action_dict.values():
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            combined_act_dim += int(sum(dim))
        else:
            combined_act_dim += dim
    return combined_obs_dim, combined_act_dim, combined_obs_dim+combined_act_dim

class DecayThenFlatSchedule():
    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / \
                np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

class ACTLayer(nn.Module):
    def __init__(self, config ,act_dim):
        super(ACTLayer, self).__init__()
        
        self.multi_discrete = False
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][config['use_orthogonal']]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), config['gain'])

        if isinstance(act_dim, np.ndarray):
            # MultiDiscrete setting: have n Linear layers for each action
            self.multi_discrete = True
            self.action_outs = nn.ModuleList([init_(nn.Linear(config['hidden_size'], a_dim)) for a_dim in act_dim])
        else:
            self.action_out = init_(nn.Linear(config['hidden_size'], act_dim))

    def forward(self, x, no_sequence=False):

        if self.multi_discrete:
            act_outs = []
            for a_out in self.action_outs:
                act_out = a_out(x)
                if no_sequence:
                    # remove the dummy first time dimension if the input didn't have a time dimension
                    act_out = act_out[0, :, :]
                act_outs.append(act_out)
        else:
            act_outs = self.action_out(x)
            if no_sequence:
                # remove the dummy first time dimension if the input didn't have a time dimension
                act_outs = act_outs[0, :, :]

        return act_outs
class PopArt(nn.Module):
    """ Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape, dtype=torch.float), requires_grad=False).to(self.device)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape, dtype=torch.float), requires_grad=False).to(self.device)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=False).to(self.device)

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(max=self.alpha, min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector, train=True):
        # Make sure input is float32
        input_vector = input_vector.to(**self.tpdv)

        if train:
            # Detach input before adding it to running means to avoid backpropping through it on
            # subsequent batches.
            
            detached_input = input_vector.detach()           
            batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))           
            batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))
            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[:self.norm_axes])
                weight = self.beta ** batch_size
            else:
                weight = self.beta
            
            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        return out

    def denormalize(self, input_vector):
        """ Transform normalized data back into original distribution """
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        return out

class MLPBase(nn.Module):
    def __init__(self, config, inputs_dim):
        super(MLPBase, self).__init__()
        self.config = config
        if self.config['use_feature_normalization']:
            self.feature_norm = nn.LayerNorm(inputs_dim)

        if self.config['use_conv1d']:
            self.conv = CONVLayer(self.config['stacked_frames'], self.config['hidden_size'], self.config['use_orthogonal'], self.config['use_ReLU'])
            random_x = torch.FloatTensor(1, self.config['stacked_frames'], self.config['inputs_dim'])
            random_out = self.conv(random_x)
            assert len(random_out.shape)==3
            inputs_dim = random_out.size(-1) * random_out.size(-2)

        self.mlp = MLPLayer(inputs_dim, self.config['hidden_size'],
                              self.config['layer_N'], self.config['use_orthogonal'], self.config['use_ReLU'])

    def forward(self, x):
        if self.config['use_feature_normalization']:
            x = self.feature_norm(x)

        if self.config['use_conv1d']:
            batch_size = x.size(0)
            x = x.view(batch_size, self.config['stacked_frames'], -1)
            x = self.conv(x)
            x = x.view(batch_size, -1)

        x = self.mlp(x)

        return x

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x

class CONVLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, use_orthogonal, use_ReLU):
        super(CONVLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.conv = nn.Sequential(
                init_(nn.Conv1d(in_channels=input_dim, out_channels=hidden_size//4, kernel_size=3, stride=2, padding=0)), active_func, #nn.BatchNorm1d(hidden_size//4),
                init_(nn.Conv1d(in_channels=hidden_size//4, out_channels=hidden_size//2, kernel_size=3, stride=1, padding=1)), active_func, #nn.BatchNorm1d(hidden_size//2),
                init_(nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)), active_func)#, nn.BatchNorm1d(hidden_size))

    def forward(self, x):
        x = self.conv(x)
        return x

def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(
        list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c