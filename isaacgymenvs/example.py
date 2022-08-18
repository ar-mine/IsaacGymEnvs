import isaacgym
import isaacgymenvs
import torch
import math

num = 1

envs = isaacgymenvs.make(
	seed=0,
	task="FrankaReach",
	num_envs=num,
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
)
# print("Observation space is", envs.observation_space)
# print("Action space is", envs.action_space)
envs.reset()

# Keep stable
time_skip = 1
for _ in range(10):
	obs, reward, done, info = envs.step(
		torch.zeros((num,) + envs.action_space.shape, device="cuda:0")
	)

# torch.linspace(0, 2*math.pi, 60, device="cuda:0")

while True:
	obs, reward, done, info = envs.step(
		torch.tensor([[0.0, 0, 0.0, 0.0, 0.0, 0.0]] * num, device="cuda:0")
	)

for t in range(60*2):
	if t < 60:
		obs, reward, done, info = envs.step(
			torch.tensor([[0.2, 0, 0.0, 0.0, 0.0, 0.0]]*num, device="cuda:0")
		)
	else:
		obs, reward, done, info = envs.step(
			torch.tensor([[-0.2, 0, 0.0, 0.0, 0.0, 0.0]] * num, device="cuda:0")
		)
