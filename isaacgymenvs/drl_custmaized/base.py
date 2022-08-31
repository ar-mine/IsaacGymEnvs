import numpy as np
import torch

import gym
from gym.spaces import Discrete, Box

from models import combined_shape
from logger import EpochLogger


class BaseBuffer:
    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_buf = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(max_size, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros(max_size, dtype=np.float32)
        self.obs_next_buf = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.ptr, self.max_size, self.full_flag = 0, max_size, False

    def store(self, obs, act, reward, obs_next):
        # Overwrite the old data when the buffer is full
        if self.ptr == self.max_size:
            self.ptr = 0
            self.full_flag = True

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.reward_buf[self.ptr] = reward
        self.obs_next_buf[self.ptr] = obs_next
        self.ptr += 1

    def sample(self, batch_size, device):
        # Sample batch from current size or all memory
        if self.full_flag:
            sample_index = np.random.choice(self.max_size, size=batch_size)
        else:
            sample_index = np.random.choice(self.ptr, size=batch_size)

        data = dict(obs=self.obs_buf[sample_index, :], act=self.act_buf[sample_index, :],
                    reward=self.reward_buf[sample_index], obs_next=self.obs_next_buf[sample_index, :])

        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


class BaseAgent:
    def __init__(self, cfg, logger, obs_space, act_space, buffer_type):
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
            self.buffer = buffer_type(self.obs_dim, self.act_dim, self.buffer_size)
            self.continuous = True
        elif isinstance(act_space, Discrete):
            self.act_dim = act_space.n
            self.buffer = buffer_type(self.obs_dim, 1, self.buffer_size)
            self.continuous = False
        else:
            raise NameError("Action space is not supported!")

    def update(self):
        raise NotImplementedError


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
