import torch

import gym
from gym.spaces import Discrete, Box

from logger import EpochLogger


class BaseAgent:
    def __init__(self, cfg, logger, obs_space, act_space):
        self.logger = logger

        self.buffer_size = cfg['buffer_size']
        # Discount of critic for future value
        self.gamma = cfg['gamma']
        # NN parameters
        self.hidden_sizes = cfg['hidden_sizes']
        self.device = torch.device(cfg['device'])

        # Create buffer to store data
        self.obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            self.act_dim = act_space.shape[0]
            self.continuous = True
        elif isinstance(act_space, Discrete):
            self.act_dim = act_space.n
            self.continuous = False
        else:
            raise NameError("Action space is not supported!")


class BaseTrainer:
    def __init__(self, cfg, logger_kwargs=None, test=False):
        self.test = test

        self.cfg = cfg
        self.logger = EpochLogger(logger_kwargs)
        self.save_per = self.cfg['save_per']
        self.device = torch.device(cfg['device'])

        self.env_name = self.cfg['env_name']
        self.render = self.cfg['render']
        # Env loader
        self.env = gym.make(self.env_name)



