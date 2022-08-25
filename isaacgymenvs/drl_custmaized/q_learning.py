from typing import Optional, List
import time

import numpy as np
import pandas as pd
import hydra
import gym
from gym.spaces import Discrete, Box

from logger import EpochLogger


class QLearningTable:
    def __init__(self, cfg, obs_dim: int, act_dim: int,
                 obs_discrete_factor: Optional[List] = None, action_list: Optional[List] = None):
        self.epsilon = cfg['epsilon']
        self.epsilon_delta = (1 - self.epsilon)/cfg['max_epochs']
        self.lr = cfg['lr']
        self.gamma = cfg['gamma']

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if action_list is not None:
            self.action_list = action_list
        else:
            self.action_list = range(act_dim)
        self.obs_factor = np.array(obs_discrete_factor)
        # Only used in tasks whose action space is discrete
        self.q_table = pd.DataFrame(columns=self.action_list, dtype=np.float64)

        self.statistic_list = [[]]*obs_dim
        self.statistic_count = [0]*obs_dim

    def check_state_exist(self, obs: str):
        if obs not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * self.act_dim,
                    index=self.q_table.columns,
                    name=obs,
                )
            )

    def get_action(self, obs):
        obs = self.discrete_serialize(obs)
        # Check if the state exists in the q_table
        self.check_state_exist(obs)

        # Choose action with epsilon-greedy
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[obs, :]

            # Choose action randomly from the set containing maximum Q value
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # Choose action randomly
            action = np.random.choice(self.action_list)

        return action

    def update(self, obs, act, reward, obs_next, done):
        obs = self.discrete_serialize(obs)
        obs_next = self.discrete_serialize(obs_next)

        self.check_state_exist(obs_next)
        q_predict = self.q_table.loc[obs, act]
        if not done:
            q_target = reward + self.gamma * self.q_table.loc[obs_next, :].max()
        else:
            q_target = reward
        self.q_table.loc[obs, act] += self.lr * (q_target - q_predict)
        # print("Total {} index in the table".format(self.q_table.index.shape[0]))

    def discrete_serialize(self, obs: np.ndarray):
        obs = (obs / self.obs_factor).astype(int)
        for idx, o in enumerate(obs):
            if o not in self.statistic_list[idx]:
                self.statistic_list[idx].append(o)
                self.statistic_count[idx] += 1
        return str(obs)

    def update_epsilon(self):
        self.epsilon += self.epsilon_delta


class QLearning:
    def __init__(self, cfg, logger_kwargs=None):
        self.cfg = cfg
        self.logger = EpochLogger(logger_kwargs)

        self.env_name = self.cfg['env_name']
        self.batch_size = self.cfg['batch_size']
        self.max_epochs = self.cfg['max_epochs']
        self.render = self.cfg['render']

        # Env loader
        self.env = self.create_env()
        self.obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            self.act_dim = self.env.action_space.n

        self.q_table = QLearningTable(self.cfg, self.obs_dim, self.act_dim,
                                      obs_discrete_factor=[0.4, 0.4, 0.04, 0.4])

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
        return 1-abs(pole_angle)/0.2 - done*200

    def train_one_epoch(self):
        # reset episode-specific variables
        obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0  # first obs comes from starting distribution
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        for t in range(self.batch_size):
            # rendering
            if self.render and (not finished_rendering_this_epoch):
                self.env.render()

            # act in the environment
            act = self.q_table.get_action(obs)
            obs_next, reward, done, _ = self.env.step(act)

            # Modify the reward based on specific task
            reward = self.reward_modify(obs_next, done)
            self.q_table.update(obs, act, reward, obs_next, done)

            ep_ret += reward
            ep_len += 1

            # Update obs
            obs = obs_next

            terminal = done
            epoch_ended = t == self.batch_size - 1

            if terminal or epoch_ended:
                # Do not render in this epoch
                finished_rendering_this_epoch = True

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)

                # reset episode-specific variables
                obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0  # first obs comes from starting distribution

    def train(self):
        start_time = time.time()

        # training loop
        for epoch in range(self.max_epochs):
            self.train_one_epoch()
            self.q_table.update_epsilon()

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
            self.logger.log_tabular('Time', time.time() - start_time)
            self.logger.dump_tabular()
            print(self.q_table.q_table.index.shape[0])
            print(self.q_table.statistic_count)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    agent = QLearning(cfg)
    agent.train()


if __name__ == "__main__":
    main()
