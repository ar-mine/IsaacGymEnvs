import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def increment_cumsum(x, gamma):
    buffer = np.zeros((len(x),))
    for i, x_ in enumerate(x):
        if i == 0:
            buffer[0] = x_
        else:
            buffer[i] = gamma*buffer[i-1]+x_
    return buffer


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def get_action(self, obs):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Compute the log prob according to the batch of obs and act
        # With gradient to compute loss
        pi = self._distribution(obs)
        if act is not None:
            log_prob = self._log_prob_from_distribution(pi, act)
            return log_prob
        else:
            return pi


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        self.network = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation).to(device)

    def _distribution(self, obs):
        logits = self.network(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act.squeeze())

    def get_action(self, obs):
        # Choose action by sampling from distribution
        # Return action and its log prob(without gradient)
        with torch.no_grad():
            pi = self._distribution(obs)
            a = pi.sample()
            log_prob = self._log_prob_from_distribution(pi, a)
        return a, log_prob


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)).to(device)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation).to(device)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

    def get_action(self, obs):
        with torch.no_grad():
            pi = self._distribution(obs)
            a = pi.sample()
            log_prob = self._log_prob_from_distribution(pi, a)
        return a, log_prob


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation, device):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation).to(device)

    def forward(self, obs):
        return self.v_net(obs)


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh, device=torch.device('cpu')):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, device)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation, device)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation, device)

    def get_action(self, obs):
        return self.pi.get_action(obs)
