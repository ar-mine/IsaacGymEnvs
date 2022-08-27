import time

import hydra
import numpy as np
import torch
from torch import nn
from torch.optim import SGD

import gym
from gym.spaces import Discrete, Box

from models import mlp, combined_shape
from logger import EpochLogger


class DQNBuffer:
    def __init__(self, obs_dim, act_dim, size, batch_size):
        act_dim = 1
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.obs_next_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.done_buffer = np.zeros(combined_shape(size, 1), dtype=bool)
        self.ptr, self.max_size, self.batch_size, self.full_flag = 0, size, batch_size, False

    def store(self, obs, act, rew, done, obs_next):
        if self.ptr == self.max_size:
            self.ptr = 0
            self.full_flag = True
        # Overwrite the old data when the buffer is full
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buffer[self.ptr] = done
        self.obs_next_buf[self.ptr] = obs_next
        self.ptr += 1

    def get(self, device):
        # sample batch memory from all memory
        if self.full_flag:
            sample_index = np.random.choice(self.max_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.ptr, size=self.batch_size)

        data = dict(obs=self.obs_buf[sample_index, :], act=self.act_buf[sample_index, :],
                    reward=self.rew_buf[sample_index, :], done=self.done_buffer[sample_index, :],
                    obs_next=self.obs_next_buf[sample_index, :])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


class DQNNetwork:
    def __init__(self, cfg, obs_dim: int, act_dim: int):
        self.epsilon = cfg['epsilon']
        self.epsilon_delta = (1 - self.epsilon) / cfg['max_epochs']
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']
        self.hidden_sizes = cfg['hidden_sizes']
        self.device = torch.device(cfg['device'])

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_list = range(act_dim)

        self.buffer = DQNBuffer(self.obs_dim, self.act_dim, cfg["epoch_length"], cfg["batch_size"])

        self.predict_network = mlp([obs_dim] + list(self.hidden_sizes) + [act_dim], nn.Tanh).to(self.device)
        self.target_network = mlp([obs_dim] + list(self.hidden_sizes) + [act_dim], nn.Tanh).to(self.device)
        self.network_sync()
        self.optimizer = SGD(self.predict_network.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.count = 0

    def get_action(self, obs):
        obs = torch.Tensor(obs).to(self.device)

        # Choose action with epsilon-greedy
        if np.random.uniform() < self.epsilon:
            with torch.no_grad():
                act = torch.argmax(self.predict_network(obs)).cpu().numpy()
        else:
            # Choose action randomly
            act = np.random.choice(self.action_list)

        return act

    def update(self, obs, act, reward, done, obs_next):
        self.count += 1

        self.buffer.store(obs, act, reward, done, obs_next)
        data = self.buffer.get(self.device)
        obs, act, reward, done, obs_next = \
            data["obs"], data["act"].long(), data["reward"], data["done"].squeeze().bool(), data["obs_next"]
        not_done = done.logical_not()

        q_predict = torch.gather(self.predict_network(obs), dim=1, index=act)
        with torch.no_grad():
            q_target = torch.zeros_like(q_predict)
            q_target[not_done, :] = reward[not_done, :] + \
                                    self.gamma * self.target_network(obs_next[not_done, :]).max(dim=1).values.view(-1, 1)
            q_target[done, :] = reward[done, :]

        self.optimizer.zero_grad()
        loss = self.loss_func(q_target, q_predict)
        loss.backward()
        self.optimizer.step()

        if self.count % 10 == 0:
            self.network_sync()
            self.count = 0

    def update_epsilon(self):
        self.epsilon += self.epsilon_delta

    def network_sync(self):
        for target_param, predict_param in zip(self.target_network.parameters(), self.predict_network.parameters()):
            target_param.data.copy_(predict_param.data)


class DQNTrainer:
    def __init__(self, cfg, logger_kwargs=None):
        self.cfg = cfg
        self.logger = EpochLogger(logger_kwargs)

        self.env_name = self.cfg['env_name']
        self.batch_size = self.cfg['batch_size']
        self.epoch_length = self.cfg['epoch_length']
        self.max_epochs = self.cfg['max_epochs']
        self.render = self.cfg['render']

        # Env loader
        self.env = self.create_env()
        self.obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            self.act_dim = self.env.action_space.n

        self.estimator = DQNNetwork(self.cfg, self.obs_dim, self.act_dim)

    def create_env(self):
        # make environment, check spaces, get obs / act dims
        # Stochastic and Categorical Policy
        env = gym.make(self.env_name)
        assert isinstance(env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(env.action_space, Discrete), \
            "This example only works for envs with discrete action spaces."
        return env

    @staticmethod
    def reward_modify(obs, done):
        pole_angle = obs[2]
        return 1 - done * 3

    def train_one_epoch(self):
        # reset episode-specific variables
        obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0  # first obs comes from starting distribution
        finished_rendering_this_epoch = False

        game_count = 0
        # collect experience by acting in the environment with current policy
        for t in range(self.epoch_length):
            game_count += 1

            # rendering
            if self.render and (not finished_rendering_this_epoch):
                self.env.render()

            # act in the environment
            act = self.estimator.get_action(obs)
            obs_next, reward, done, _ = self.env.step(act)

            terminal = done
            epoch_ended = t == self.epoch_length - 1

            if game_count == 500:
                done = False

            # Modify the reward based on specific task
            # reward = self.reward_modify(obs_next, done)
            if (t+1) % 5 == 0:
                self.estimator.update(obs, act, reward, done, obs_next)

            ep_ret += reward
            ep_len += 1

            # Update obs
            obs = obs_next

            if terminal or epoch_ended:
                # Do not render in this epoch
                finished_rendering_this_epoch = True

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    game_count = 0
                # reset episode-specific variables
                obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0  # first obs comes from starting distribution

    def train(self):
        start_time = time.time()

        # training loop
        for epoch in range(self.max_epochs):
            self.train_one_epoch()
            self.estimator.update_epsilon()

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('Time', time.time() - start_time)
            self.logger.dump_tabular()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    agent = DQNTrainer(cfg)
    agent.train()


if __name__ == "__main__":
    main()
