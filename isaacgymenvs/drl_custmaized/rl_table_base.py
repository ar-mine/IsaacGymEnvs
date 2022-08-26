from typing import Optional, List

import numpy as np
import pandas as pd


class RLTableBase:
    def __init__(self, obs_dim: int, act_dim: int,
                 obs_discrete_factor: Optional[List] = None, action_list: Optional[List] = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        if action_list is not None:
            self.action_list = action_list
        else:
            self.action_list = range(act_dim)
        self.obs_factor = np.array(obs_discrete_factor)
        # Only used in tasks whose action space is discrete
        self.q_table = pd.DataFrame(columns=self.action_list, dtype=np.float64)

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

    def get_action(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def discrete_serialize(self, obs: np.ndarray):
        obs = (obs / self.obs_factor).astype(int)
        return str(obs)
