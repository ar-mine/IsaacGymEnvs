import time

import hydra

import torch
from torch.optim import Adam

import gym
from gym.spaces import Discrete, Box

from base import BaseBuffer
from models import MLPActorCritic
from logger import EpochLogger


class A2CBuffer(BaseBuffer):
    def __init__(self, obs_dim, act_dim, max_size):
        super(A2CBuffer, self).__init__(obs_dim, act_dim, max_size)


class A2CAgent:
    def __init__(self, cfg, logger, obs_space, act_space):
        self.logger = logger

        self.buffer_size = cfg['buffer_size']
        # Discount of critic for future value
        self.gamma = cfg['gamma']
        # NN parameters
        self.lr_actor = cfg['lr_actor']
        self.lr_critic = cfg['lr_critic']
        self.hidden_sizes = cfg['hidden_sizes']
        self.device = torch.device(cfg['device'])

        # Create buffer to store data
        self.obs_dim = obs_space.shape[0]
        if isinstance(act_space, Box):
            self.act_dim = act_space.shape[0]
            self.buffer = A2CBuffer(self.obs_dim, self.act_dim, self.buffer_size)
        elif isinstance(act_space, Discrete):
            self.act_dim = act_space.n
            self.buffer = A2CBuffer(self.obs_dim, 1, self.buffer_size)
        else:
            raise NameError("Action space is not supported!")

        # Network and optimizer
        self.ac_network = MLPActorCritic(obs_space, act_space, self.hidden_sizes, device=self.device)
        self.actor_optimizer = Adam(self.ac_network.pi.parameters(), lr=self.lr_actor)
        self.critic_optimizer = Adam(self.ac_network.v.parameters(), lr=self.lr_critic)

    def update(self, data):
        # Get corresponding data from sampling batch
        obs, act, reward, obs_next = data["obs"], data["act"], data["reward"], data["obs_next"]

        # Gradient descent for critic
        self.critic_optimizer.zero_grad()
        loss_critic = self._compute_loss_critic(obs, reward, obs_next)
        loss_critic_mse = loss_critic.norm().pow(2) / loss_critic.shape[0]
        loss_critic_mse.backward()
        self.critic_optimizer.step()

        # Record loss-critic
        self.logger.store(LossC=loss_critic_mse.item())

        # Gradient descent for actor
        self.actor_optimizer.zero_grad()
        loss_actor = self._compute_loss_actor(obs, act, loss_critic_mse.detach())
        loss_actor.backward()
        self.actor_optimizer.step()

        # Record loss-actor
        self.logger.store(LossA=loss_actor.item())

    def _compute_loss_critic(self, obs, reward, obs_next):
        v = self.ac_network.v(obs)
        v_next = self.ac_network.v(obs_next)
        return reward + self.gamma * v_next - v

    def _compute_loss_actor(self, obs, act, loss_critic):
        log_prob = self.ac_network.pi(obs, act).view((-1, 1))
        loss = -(log_prob * loss_critic).mean()
        return loss

    def get_action(self, obs):
        act, log_prob = self.ac_network.get_action(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        return act.cpu().numpy(), log_prob.cpu().numpy()


class A2CTrainer:
    def __init__(self, cfg, logger_kwargs=None):
        self.cfg = cfg
        self.logger = EpochLogger(logger_kwargs)
        self.device = torch.device(cfg['device'])

        self.env_name = self.cfg['env_name']
        self.render = self.cfg['render']

        self.batch_size = self.cfg['batch_size']

        self.epoch_length = self.cfg['epoch_length']
        self.max_epochs = self.cfg['max_epochs']
        self.step_start_train = self.cfg['step_start_train']
        self.train_freq = self.cfg['train_freq']

        # Env loader
        self.env = gym.make(self.env_name)

        self.estimator = A2CAgent(self.cfg, self.logger, self.env.observation_space, self.env.action_space)

    @staticmethod
    def reward_modify(obs, obs_next):
        pos, vel = obs
        pos_next, vel_next = obs_next
        h = abs(pos+0.5)
        h_next = abs(pos_next+0.5)
        reward = (h_next - h) + (vel_next**2 - vel**2)
        return -1+reward*10

    def train(self):
        # Timer starts
        start_time = time.time()
        epoch = 0

        # Reset episode-specific variables
        obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0
        # Reset epoch-specific variables
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        for t in range(self.max_epochs * self.epoch_length):
            # rendering
            if self.render and (not finished_rendering_this_epoch):
                self.env.render()

            # Act in the environment
            if (t + 1) > self.step_start_train:
                act, log_prob = self.estimator.get_action(obs)
            else:
                act = self.env.action_space.sample()
            obs_next, reward, done, *info = self.env.step(act)

            # Modify the reward based on specific task
            # reward_fine = self.reward_modify(obs, obs_next)
            reward_fine = reward
            # Record experience
            self.estimator.buffer.store(obs, act, reward_fine, obs_next)

            # Train after some steps with train_freq
            if (t + 1) % self.train_freq == 0 and (t + 1) > self.step_start_train:
                for count in range(self.train_freq):
                    data = self.estimator.buffer.sample(self.batch_size, self.device)
                    self.estimator.update(data)

            # Update obs
            obs = obs_next

            # Update epoch-specific variables
            ep_ret += reward
            ep_len += 1

            # Set corresponding end flag
            episode_end = done
            epoch_end = (t + 1) % self.epoch_length == 0

            if episode_end or epoch_end:
                if episode_end:
                    # Do not render in this epoch
                    finished_rendering_this_epoch = True
                    # only save EpRet / EpLen if trajectory finished
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)

                # reset episode-specific variables
                obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0

                if epoch_end:
                    epoch += 1
                    # Reset epoch-specific variables
                    finished_rendering_this_epoch = False

                    # Log info about epoch
                    self.logger.log_tabular('Epoch', epoch)
                    self.logger.log_tabular('LossA', average_only=True)
                    self.logger.log_tabular('LossC', average_only=True)
                    self.logger.log_tabular('EpRet', with_min_and_max=True)
                    self.logger.log_tabular('EpLen', average_only=True)
                    self.logger.log_tabular('Time', time.time() - start_time)
                    self.logger.dump_tabular()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    agent = A2CTrainer(cfg)
    agent.train()


if __name__ == "__main__":
    main()
