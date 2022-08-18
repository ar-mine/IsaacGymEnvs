import math
from typing import Tuple

import torch
from torch import Tensor

from isaacgym.torch_utils import *
from isaacgym import gymutil, gymapi

from .franka_base import FrankaBase


class FrankaReach(FrankaBase):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Euclidian distance < threshold, we regard task are completed
        self.threshold = 0.005
        self.max_dist = 1.0

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        # Target pose
        self._target_pose = torch.zeros(self._eef_state.shape).to(self.device)

        self.set_random_pose(0)

    def set_random_pose(self, ids):
        if isinstance(ids, Tensor):
            random_temp = torch.rand((ids.shape[0], 3)).to(self.device)
        else:
            random_temp = torch.rand((self._eef_state.shape[0], 3)).to(self.device)
        random_temp[:, 0] += 0.2
        random_temp[:, 0] *= 0.5
        random_temp[:, 1] -= 0.5
        random_temp[:, 2] *= 0.5

        if isinstance(ids, Tensor):
            self._target_pose[ids, :3] = random_temp
            self._target_pose[ids, 3] = 1.0
        else:
            self._target_pose[:, :3] = random_temp
            self._target_pose[:, 3] = 1.0

        # print(random_temp)

        self.gym.clear_lines(self.viewer)
        for i in range(self.num_envs):
            # Define start pose for table stand
            target_pose = gymapi.Transform()
            target_pose.p = gymapi.Vec3(*self._target_pose[i, :3])
            target_pose.r = gymapi.Quat(*self._target_pose[i, 3:7])
            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], target_pose)
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], target_pose)

    def _compute_reward(self):
        eef_target_t = self._target_pose[:, :3]
        eef_state_t = self._eef_state[:, :3]

        self.rew_buf[:], self.reset_buf[:] = \
            compute_reward(eef_target_t, eef_state_t, self.max_dist,
                           self.threshold, self.max_episode_length, self.progress_buf, self.reset_buf)

    def _customized_asset(self):
        # Create helper geometry used for visualization
        # Create a wireframe axis
        self.axes_geom = gymutil.AxesGeometry(0.1)
        # Create a wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.set_random_pose(env_ids)

        self._compute_observations()
        self._compute_reward()

        self.debug_viz()


@torch.jit.script
def compute_reward(eef_target_t: Tensor, eef_state_t: Tensor, max_dist: float,
                   threshold: float, max_episode_length: float,
                   progress_buf: Tensor, reset_buf: Tensor) -> Tuple[Tensor, Tensor]:
    euclidian_dist = torch.norm(eef_target_t - eef_state_t, dim=1)

    dist_reward = (max_dist - euclidian_dist) / max_dist
    win_mask = euclidian_dist <= threshold
    timeout_mask = progress_buf >= max_episode_length - 1
    abort_mask = dist_reward <= 0

    reward = dist_reward + win_mask * (max_episode_length - progress_buf)

    # Compute resets
    reset_buf = torch.where(
        timeout_mask | win_mask | abort_mask,
        torch.ones_like(reset_buf), reset_buf)

    return reward, reset_buf
