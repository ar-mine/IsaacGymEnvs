from typing import Optional, List

import hydra

from q_learning import QLearning, QLearningTable


class SarsaTable(QLearningTable):
    def __init__(self, cfg, obs_dim: int, act_dim: int,
                 obs_discrete_factor: Optional[List] = None, action_list: Optional[List] = None):
        super().__init__(cfg, obs_dim, act_dim, obs_discrete_factor, action_list)

    def update(self, obs, act, reward, done, obs_next, act_next=None):
        obs = self.discrete_serialize(obs)
        obs_next = self.discrete_serialize(obs_next)

        self.check_state_exist(obs_next)
        q_predict = self.q_table.loc[obs, act]
        if not done:
            q_target = reward + self.gamma * self.q_table.loc[obs_next, act_next]
        else:
            q_target = reward
        self.q_table.loc[obs, act] += self.lr * (q_target - q_predict)


class SarsaLearning(QLearning):
    def __init__(self, cfg, logger_kwargs=None):
        super(SarsaLearning, self).__init__(cfg, logger_kwargs, table_cls=SarsaTable)

    def train_one_epoch(self):
        # reset episode-specific variables
        obs, done, ep_ret, ep_len = self.env.reset(), False, 0, 0  # first obs comes from starting distribution
        finished_rendering_this_epoch = False

        act = self.q_table.get_action(obs)

        # collect experience by acting in the environment with current policy
        for t in range(self.batch_size):
            # rendering
            if self.render and (not finished_rendering_this_epoch):
                self.env.render()

            # act in the environment
            obs_next, reward, done, _ = self.env.step(act)
            if not done:
                act_next = self.q_table.get_action(obs)

            # Modify the reward based on specific task
            reward = self.reward_modify(obs_next, done)
            self.q_table.update(obs, act, reward, done, obs_next, act_next)

            ep_ret += reward
            ep_len += 1

            # Update obs
            obs = obs_next
            act = act_next

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


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    agent = SarsaLearning(cfg)
    agent.train()


if __name__ == "__main__":
    main()
