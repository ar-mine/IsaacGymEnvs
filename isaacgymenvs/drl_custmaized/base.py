import numpy as np
import torch

from models import combined_shape


class BaseBuffer:
    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_buf = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(max_size, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros(combined_shape(max_size, 1), dtype=np.float32)
        self.obs_next_buf = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.ptr, self.max_size, self.full_flag = 0, max_size, False

    def store(self, obs, act, reward, obs_next):
        # Overwrite the old data when the buffer is full
        if self.ptr == self.max_size:
            self.ptr = 0
            self.full_flag = True

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.reward_buf[self.ptr] = reward
        self.obs_next_buf[self.ptr] = obs_next
        self.ptr += 1

    def sample(self, batch_size, device):
        # Sample batch from current size or all memory
        if self.full_flag:
            sample_index = np.random.choice(self.max_size, size=batch_size)
        else:
            sample_index = np.random.choice(self.ptr, size=batch_size)

        data = dict(obs=self.obs_buf[sample_index, :], act=self.act_buf[sample_index, :],
                    reward=self.reward_buf[sample_index, :], obs_next=self.obs_next_buf[sample_index, :])

        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in data.items()}