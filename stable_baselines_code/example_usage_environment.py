import torch
import numpy as np
from skimage.morphology import convex_hull_image
from torch import nn

from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import dummy_reward


# Dummy implementations for rec_net, loss_func, reward_func
class DummyRecNet(torch.nn.Module):
    def forward(self, x):
        # x is assumed to be a tensor of shape (batch_size, channels, res, res)
        # Convert x to numpy array and reshape if necessary
        x_np = x.squeeze().cpu().numpy()  # Convert to numpy array and squeeze out batch and channel dimensions

        hull = convex_hull_image(x[0, 0])

        # Example: Return convex hull vertices as a tensor (you may need to adjust this based on your requirement)
        # Return convex hull vertices as a tensor with singleton dimension
        convex_hull_vertices = torch.tensor(hull, dtype=torch.float).reshape((1, 1, 256, 256))

        return convex_hull_vertices


# Instantiate the environment
rec_net = DummyRecNet()
dataset = None  # Replace with actual dataset
loss_func = nn.BCELoss()
reward_func = dummy_reward

env = ShapeEnv(rec_net, dataset, loss_func, reward_func)

# Reset environment
observation = env.reset()

# Run a sample loop
for _ in range(env.max_steps):
    action = env.action_space.sample()  # Sample random action
    observation, reward, done, info = env.step(action)
    env.render()  # Optional: Implement the render function for visualization
    if done:
        break

env.close()
