import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from typing import Optional, List, Tuple


class SimplePolicyGradient:
    def __init__(self):
        # Will be loaded from cfg later
        self.env_name = 'CartPole-v1'
        self.hidden_sizes = [32]
        self.lr = 1e-2
        self.max_epochs = 50
        self.batch_size = 5000
        self.render = True
        self.device = torch.device("cuda:0")

        # Env loader
        self.env = None
        self.obs_dim: Optional[int] = None
        self.act_dim: Optional[int] = None
        self.create_env()

        # Network
        # make core of policy network
        self.network = self.mlp(sizes=[self.obs_dim] + self.hidden_sizes + [self.act_dim])

    def create_env(self):
        # make environment, check spaces, get obs / act dims
        # Stochastic and Categorical Policy
        self.env = gym.make(self.env_name)
        assert isinstance(self.env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(self.env.action_space, Discrete), \
            "This example only works for envs with discrete action spaces."

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

    def get_policy(self, obs):
        # make function to compute action distribution
        act_prob = self.network(obs)
        return Categorical(logits=act_prob)

    def get_action(self, obs):
        # make action selection function (outputs int actions, sampled from policy)
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs, act, weights):
        # make loss function whose gradient, for the right data, is policy gradient
        log_prob = self.get_policy(obs).log_prob(act)
        return -(log_prob * weights).mean()

    # for training policy
    def train_one_epoch(self, optimizer) -> (Tuple, List, List):
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rewards = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and self.render:
                self.env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = self.get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = self.env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rewards.append(reward)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rewards), len(ep_rewards)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rewards = self.env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > self.batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = self.compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                       act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                       weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                       )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    def train(self):
        # make optimizer
        optimizer = Adam(self.network.parameters(), lr=self.lr)

        # training loop
        for i in range(self.max_epochs):
            batch_loss, batch_rets, batch_lens = self.train_one_epoch(optimizer)
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (i, batch_loss, np.mean(batch_rets).item(), np.mean(batch_lens).item()))

    @staticmethod
    def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
        # Build a feedforward neural network.
        layers = []
        for j in range(len(sizes) - 1):
            act = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        return nn.Sequential(*layers)


def main():
    agent = SimplePolicyGradient()
    agent.train()


if __name__ == "__main__":
    main()
