import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from typing import Optional, List, Tuple
from models import MLPCategoricalActor, combined_shape, discount_cumsum
import hydra


class SPGBuffer:
    """
       A buffer for storing trajectories experienced by a VPG agent interacting
       with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
       for calculating the advantages of state-action pairs.
       """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99):
        # For discrete space, act_dim = 1(classification)
        act_dim = 1
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)

        # the next line computes rewards-to-go, to be targets for the value function
        # self.ret_buf[path_slice] = np.sum(rews[:-1])
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class SimplePolicyGradient:
    def __init__(self, cfg):
        self.cfg = cfg

        # Will be loaded from cfg later
        self.env_name = self.cfg['env_name']
        self.hidden_sizes = self.cfg['hidden_sizes']
        self.lr = self.cfg['lr']
        self.max_epochs = self.cfg['max_epochs']
        self.batch_size = self.cfg['batch_size']
        self.render = self.cfg['render']
        self.device = torch.device(self.cfg['device'])

        # Env loader
        self.env = None
        self.obs_dim: Optional[int] = None
        self.act_dim: Optional[int] = None
        self.create_env()

        # Network
        # make core of policy network
        self.actor, self.buffer = self.create_estimator()

        self.pi_optimizer = Adam(self.actor.parameters(), lr=self.lr)

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

    def create_estimator(self):
        return MLPCategoricalActor(self.obs_dim, self.act_dim, self.hidden_sizes, nn.Tanh, device=self.device), \
               SPGBuffer(self.obs_dim, self.act_dim, self.batch_size)

    def compute_loss_pi(self, data):
        obs, act, logp_old, ret = data['obs'].to(self.device), data['act'].to(self.device), \
                                  data['logp'].to(self.device), data['ret'].to(self.device)

        # Policy loss
        pi, logp = self.actor(obs, act.squeeze())

        # make loss function whose gradient, for the right data, is policy gradient
        loss_pi = -(logp * ret).mean()

        return loss_pi

    @staticmethod
    def reward_to_go(rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

    # for training policy
    def train_one_epoch(self) -> (Tuple, List, List):
        # make some empty lists for logging.
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0  # first obs comes from starting distribution

        # collect experience by acting in the environment with current policy
        for t in range(self.batch_size):
            # rendering
            if self.render:
                self.env.render()

            # act in the environment
            act, logp = self.actor.step(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
            act, logp = act.cpu().numpy(), logp.cpu().numpy()
            obs_next, reward, done, _ = self.env.step(act)
            ep_ret += reward
            ep_len += 1

            # save obs, action, reward
            self.buffer.store(obs, act, reward, logp)

            # Update obs
            obs = obs_next

            epoch_ended = t == self.batch_size - 1

            if done or epoch_ended:
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                self.buffer.finish_path()

                # reset episode-specific variables
                obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0  # first obs comes from starting distribution

        # take a single policy gradient update step
        batch_loss = self.opt_update()

        return batch_loss.cpu(), batch_rets, batch_lens

    def opt_update(self):
        # take a single policy gradient update step
        data = self.buffer.get()

        # Train policy with a single step of gradient descent
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        return loss_pi

    def train(self):
        # training loop
        for i in range(self.max_epochs):
            batch_loss, batch_rets, batch_lens = self.train_one_epoch()
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (i, batch_loss, np.mean(batch_rets).item(), np.mean(batch_lens).item()))


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    agent = SimplePolicyGradient(cfg)
    agent.train()


if __name__ == "__main__":
    main()
