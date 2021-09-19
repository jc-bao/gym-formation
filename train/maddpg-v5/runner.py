import numpy as np

class Runner():
    def __init__(self, episode_num = 30):
        # paramters
        self.episode_num = episode_num # number of episode to update network
        self.env = 

    def run(self):
        # warm up
        self.warmup()
        # collect data
        self.collect()
        # train
        self.train()
        # save
        # log
        # eval
        raise NotImplementedError

    def collect(self):
        # collect
        for ep in range(self.episode_num):
            # reset env
            obs_batch = np.concatenate(env.reset())

        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def warmup(self):
        raise NotImplementedError
