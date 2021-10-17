import gym, torch, numpy as np, tianshou as ts
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

# make  env
train_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(4)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make('CartPole-v0') for _ in range(1)])

# build the network
class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        ) # inplace: calculate without copy prod: flatten the shape

    def forward(self, obs, state = None, info = {}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state

state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
action_shape = train_envs.action_space[0].shape or train_envs.action_space[0].n
net = Net(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

# set up policy
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)

# setup collector
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

# logger
writer = SummaryWriter('log/dqn')
logger = TensorboardLogger(writer)

# trainer
# step/epoch: collect  
# update/step: train after these steps
# step/collect: update times according to collect
result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=10000, step_per_collect=10,
    update_per_step=0.1, episode_per_test=100, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= train_envs.spec[0].reward_threshold,
    logger = logger)
print(f'Finished training! Use {result["duration"]}')

# save policy
torch.save(policy.state_dict(), 'models/dqn.pth')
# policy.load_state_dict(torch.load('dqn.pth'))
# evaluate
test_collector.collect(n_episode = 1, render = 1/30)
