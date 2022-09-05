import torch
import numpy as np

from models import combined_shape


class BaseBuffer:
    def __init__(self, obs_dim, act_dim, buffer_size, continuous: bool):
        self.ptr, self.size, self.buffer_size = 0, 0, buffer_size

        self.data = {'obs': np.zeros(combined_shape(buffer_size, obs_dim), dtype=np.float32)}
        if continuous:
            self.data['act'] = np.zeros(combined_shape(buffer_size, act_dim), dtype=np.float32)
        else:
            self.data['act'] = np.zeros(buffer_size, act_dim, dtype=np.int)
        self.data['reward'] = np.zeros(buffer_size, dtype=np.float32)
        self.data['obs_next'] = np.zeros(combined_shape(buffer_size, obs_dim), dtype=np.float32)

    def store(self, **kwargs):
        for k in kwargs.keys():
            self.data[k][self.ptr] = kwargs[k]

        # Overwrite the old data when data is more than buffer size
        self.ptr = (self.ptr + 1) % self.buffer_size
        # Represent how much data is available
        self.size = min(self.size + 1, self.buffer_size)


class ReplayBuffer(BaseBuffer):
    def __init__(self, obs_dim, act_dim, buffer_size, continuous: bool = True):
        super(ReplayBuffer, self).__init__(obs_dim, act_dim, buffer_size, continuous)
        self.data['done'] = np.zeros(buffer_size, dtype=np.float32)

    def sample(self, batch_size, device=torch.device("cpu")):
        # Sample batch from current size or all memory

        sample_index = np.random.choice(self.size, size=batch_size)

        batch_data = {}
        for k in self.data.keys():
            if len(self.data[k].shape) == 2:
                batch_data[k] = self.data[k][sample_index, :]
            else:
                batch_data[k] = self.data[k][sample_index]

        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch_data.items()}
