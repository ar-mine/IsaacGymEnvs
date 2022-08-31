import time
from multiprocessing import Process, Array

import numpy as np
import hydra

import torch
from torch import nn
from torch.optim import Adam

from base import BaseBuffer, BaseAgent, BaseTrainer
from models import MLPCategoricalActor, MLPGaussianActor, MLPCritic, discount_cumsum

PATH = 'F:\\Github\\IsaacGymEnvs\\isaacgymenvs\\drl_custmaized\\experiments\\1661911727\\pyt_save\\model80.pt'


class ACBuffer(BaseBuffer):
    def __init__(self, obs_dim, act_dim, max_size, gamma=0.95):
        super(ACBuffer, self).__init__(obs_dim, act_dim, max_size)
        self.ret_buf = np.zeros(max_size, dtype=np.float32)
        self.gamma = gamma
        self.path_start_idx = 0

    def get(self, device):
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf, act=self.act_buf, reward=self.reward_buf, obs_next=self.obs_next_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}


class ACAgent(BaseAgent):
    def __init__(self, cfg, logger, obs_space, act_space, test):
        super().__init__(cfg, logger, obs_space, act_space, buffer_type=ACBuffer)

        self.test = test
        self.buffer.gamma = self.gamma

        # Network(depends on action space) and optimizer
        if self.continuous:
            self.actor_network = MLPGaussianActor(self.obs_dim, self.act_dim, self.hidden_sizes,
                                                  activation=nn.Tanh, device=self.device)
        else:
            self.actor_network = MLPCategoricalActor(self.obs_dim, self.act_dim, self.hidden_sizes,
                                                     activation=nn.Tanh, device=self.device)
        self.critic_network = MLPCritic(self.obs_dim, self.hidden_sizes,
                                        activation=nn.Tanh, device=self.device)

        if not test:
            self.actor_optimizer = Adam(self.actor_network.parameters(), lr=cfg['lr_actor'])
            self.critic_optimizer = Adam(self.critic_network.parameters(), lr=cfg['lr_critic'])
            self.logger.setup_pytorch_saver(self.actor_network)
        else:
            self.actor_network.load_state_dict(torch.load(PATH))

    def update(self):
        data = self.buffer.get(self.device)
        # Get corresponding data from sampling batch
        obs, act, reward, obs_next = data["obs"], data["act"], data["reward"], data["obs_next"]

        # Gradient descent for critic
        self.critic_optimizer.zero_grad()
        loss_critic, td_error = self._compute_loss_critic(obs, reward, obs_next)
        loss_critic.backward()
        self.critic_optimizer.step()

        # Record loss-critic
        self.logger.store(LossC=loss_critic.item())

        # Gradient descent for actor
        self.actor_optimizer.zero_grad()
        loss_actor = self._compute_loss_actor(obs, act, td_error.detach())
        loss_actor.backward()
        self.actor_optimizer.step()

        # Record loss-actor
        self.logger.store(LossA=loss_actor.item())

    def _compute_loss_actor(self, obs, act, ret):
        log_prob = self.actor_network(obs, act)
        loss = -(log_prob * ret).mean()
        return loss

    def _compute_loss_critic(self, obs, reward, obs_next):
        q_value = self.critic_network(obs)
        q_value_next = self.critic_network(obs_next)
        td_error = reward + self.gamma * q_value_next - q_value
        loss = -td_error.mean()
        return loss, td_error

    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if not self.test:
            act, log_prob = self.actor_network.get_action(obs)
            return act.cpu().numpy(), log_prob.cpu().numpy()
        else:
            if self.continuous:
                act, log_prob = self.actor_network.get_action(obs)
            else:
                act = torch.argmax(self.actor_network(obs).logits)
            return act.cpu().numpy()


class ACTrainer(BaseTrainer):
    def __init__(self, cfg, logger_kwargs=None, test=False):
        super().__init__(cfg, logger_kwargs, test)
        self.epoch_length = self.cfg['epoch_length']
        self.max_epochs = self.cfg['max_epochs']
        self.step_start_train = self.cfg['step_start_train']
        self.train_freq = self.cfg['train_freq']

        self.agent = ACAgent(self.cfg, self.logger, self.env.observation_space, self.env.action_space, self.test)

        # Current steps, render flag
        self.process_exchange = Array('i', [0, 0])

    @staticmethod
    def reward_modify(obs, obs_next):
        pos, vel = obs
        pos_next, vel_next = obs_next
        h = abs(pos + 0.5)
        h_next = abs(pos_next + 0.5)
        alpha = 0.2
        reward = (alpha * (h_next - h) + (1 - alpha) * ((vel_next * 100 / 7) ** 2 - (vel * 100 / 7) ** 2)) * 500
        # pos, vel = obs
        # pos_next, vel_next = obs_next
        # h = abs(pos + 0.5)
        # h_next = abs(pos_next + 0.5)
        # alpha = 0.2
        # reward = (alpha*h_next + (1-alpha)*((vel_next*100/7) ** 2))*10
        return -1 + reward

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
            act, log_prob = self.agent.get_action(obs)
            obs_next, reward, done, *info = self.env.step(act)

            # Modify the reward based on specific task
            # reward_fine = self.reward_modify(obs, obs_next)
            reward_fine = reward
            # reward = reward_fine

            # Record experience
            self.agent.buffer.store(obs, act, reward_fine, obs_next)

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
                    self.agent.update()

                    epoch += 1
                    # Reset epoch-specific variables
                    finished_rendering_this_epoch = False

                    # Log info about epoch
                    self.logger.log_tabular('Epoch', epoch)
                    self.logger.log_tabular('LossC', average_only=True)
                    self.logger.log_tabular('LossA', average_only=True)
                    self.logger.log_tabular('EpRet', average_only=True)
                    self.logger.log_tabular('EpLen', average_only=True)
                    self.logger.log_tabular('Time', time.time() - start_time)
                    self.logger.dump_tabular()

                    # Save model
                    if (epoch % self.save_per == 0) or (epoch == self.max_epochs):
                        self.logger.save_state({'env': self.env}, epoch)

    def evaluate(self):
        obs, done = self.env.reset(), False
        count = 0
        while True:
            self.env.render()

            # Act in the environment
            act = self.agent.get_action(obs)
            obs_next, reward, done, *info = self.env.step(act)
            obs = obs_next
            count += 1

            if done:
                obs, done = self.env.reset(), False
                print(count)
                count = 0


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    agent = ACTrainer(cfg, test=cfg['test'])
    if not cfg['test']:
        agent.train()
    else:
        agent.evaluate()


if __name__ == "__main__":
    main()
