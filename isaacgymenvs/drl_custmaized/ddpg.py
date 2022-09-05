import time
from copy import deepcopy

import hydra

import torch
from torch import nn
from torch.optim import Adam

from base.base import BaseAgent, BaseTrainer
from base.buffer import ReplayBuffer
from base.models import MLPActorCritic, count_vars

PATH = "/home/armine/Code/IsaacGymEnvs/isaacgymenvs/drl_custmaized/experiments/1662367834/pyt_save/model200.pt"


class DDPGAgent(BaseAgent):
    def __init__(self, cfg, logger, obs_space, act_space, evaluate, act_limit):
        # Whether in evaluate mode
        self.evaluate = evaluate
        self.act_limit = act_limit
        super().__init__(cfg, logger, obs_space, act_space)
        assert self.continuous, "DDPG only works on continuous action space."

        self.sync_factor = cfg['soft_update_factor']

        # Buffer init
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size, self.continuous)

        # Network(depends on action space) and optimizer init
        self.actor_critic = MLPActorCritic(self.obs_dim, self.act_dim, hidden_sizes=self.hidden_sizes,
                                           act_limit=act_limit, activation=nn.ReLU, device=self.device)

        if not self.evaluate:
            self.actor_critic_targ = deepcopy(self.actor_critic)

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.actor_critic_targ.parameters():
                p.requires_grad = False

            self.actor_optimizer = Adam(self.actor_critic.pi.parameters(), lr=cfg['lr_actor'])
            self.critic_optimizer = Adam(self.actor_critic.q.parameters(), lr=cfg['lr_critic'])

            self.logger.setup_pytorch_saver(self.actor_critic)
        else:
            self.actor_critic.load_state_dict(torch.load(PATH))

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(count_vars(module) for module in [self.actor_critic.pi, self.actor_critic.q])
        logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n' % var_counts)

    def update(self, batch_size):
        data = self.buffer.sample(batch_size, self.device)

        # Update Q-function
        self.critic_optimizer.zero_grad()
        loss_critic = self._compute_loss_critic(data)
        loss_critic.backward()
        self.critic_optimizer.step()

        # Record loss-critic
        self.logger.store(LossC=loss_critic.item())

        for p in self.actor_critic.q.parameters():
            p.requires_grad = False

        # Gradient ascent for actor
        self.actor_optimizer.zero_grad()
        loss_actor = self._compute_loss_actor(data)
        loss_actor.backward()
        self.actor_optimizer.step()

        # Record loss-actor
        self.logger.store(LossA=loss_actor.item())

        for p in self.actor_critic.q.parameters():
            p.requires_grad = True

        # Soft update
        self.soft_update()

    def _compute_loss_actor(self, data):
        obs = data["obs"]
        q_value = self.actor_critic.q(obs, self.actor_critic.pi(obs))
        return -q_value.mean()

    def _compute_loss_critic(self, data):
        obs, act, reward, obs_next, done = data["obs"], data["act"], data["reward"], data["obs_next"], data['done']
        q = self.actor_critic.q(obs, act)
        with torch.no_grad():
            q_target = self.actor_critic_targ.q(obs_next, self.actor_critic_targ.pi(obs_next))
            targets = reward + self.gamma * (1 - done) * q_target
        msbe_loss = (q - targets) ** 2
        return msbe_loss.mean()

    def get_action(self, obs, noise_scale=None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            act = self.actor_critic.pi(obs)
            if not self.evaluate:
                assert noise_scale is not None, "Train model needs noise_scale parameters."
                act += noise_scale * torch.randn(self.act_dim)
                act[act < self.act_limit[0]] = self.act_limit[0]
                act[act > self.act_limit[1]] = self.act_limit[1]
        return act.cpu().numpy()

    def soft_update(self):
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_targ.parameters()):
                p_targ.data.mul_(self.sync_factor)
                p_targ.data.add_((1 - self.sync_factor) * p.data)


class DDPGTrainer(BaseTrainer):
    def __init__(self, cfg, seed=0, logger_kwargs=None, evaluate=False):
        super().__init__(cfg, seed, logger_kwargs, evaluate)
        self.batch_size = self.cfg['batch_size']
        self.epoch_length = self.cfg['epoch_length']
        self.max_epochs = self.cfg['max_epochs']
        self.step_start_train = self.cfg['step_start_train']
        self.train_freq = self.cfg['train_freq']

        self.act_limit = torch.Tensor([[-2.0], [2.0]])

        self.agent = DDPGAgent(self.cfg, self.logger, self.env.observation_space,
                               self.env.action_space, self.evaluate, act_limit=self.act_limit)

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
            if t > 2*self.epoch_length:
                act = self.agent.get_action(obs, noise_scale=0.1)
            else:
                act = self.env.action_space.sample()
            obs_next, reward, done, *info = self.env.step(act)

            # Modify the reward based on specific task
            # reward_fine = self.reward_modify(obs, obs_next)
            reward_fine = reward

            # Record experience
            self.agent.buffer.store(obs=obs, act=act, reward=reward_fine, obs_next=obs_next, done=done)

            # Update obs
            obs = obs_next

            # Update epoch-specific variables
            ep_ret += reward
            ep_len += 1

            # Set corresponding end flag
            episode_end = done
            epoch_end = (t + 1) % self.epoch_length == 0

            # Update handling
            if t >= self.step_start_train and t % self.train_freq == 0:
                for _ in range(self.train_freq):
                    self.agent.update(self.batch_size)

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
                    self.logger.log_tabular('LossC', average_only=True)
                    self.logger.log_tabular('LossA', average_only=True)
                    self.logger.log_tabular('EpRet', average_only=True)
                    self.logger.log_tabular('EpLen', average_only=True)
                    self.logger.log_tabular('Time', time.time() - start_time)
                    self.logger.dump_tabular()

                    # Save model
                    if (epoch % self.save_per == 0) or (epoch == self.max_epochs):
                        self.logger.save_state({'env': self.env}, epoch)

    def test(self):
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
    agent = DDPGTrainer(cfg, evaluate=cfg['test'])
    if not cfg['test']:
        agent.train()
    else:
        agent.test()


if __name__ == "__main__":
    main()
