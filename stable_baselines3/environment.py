import os
import gym
from gym import spaces
import numpy as np

def next_shape_path():
    return './datasets/2D_shapes/bottle/1a7ba1f4c892e2da30711cdbdbc739240'

def render_observation(grasping_points, reconstruction):
  

class ShapeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(ShapeEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype='float32')
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32)

    def step(self, action):

        alpha = action[0] * 2 * np.pi
        beta = action[0] * np.pi

        """
        TODO:

        1, Cast ray based on both angles.
        2. Calculate intersection pixel of outline, handle case if object is missed (negative Reward?)
        3. Infer reconstruction with new grasp point.
        4. Calculate reward (rel/abs decrease in loss)
        5. Update observations

        """

        return observation, reward, done, info
    def reset(self):
        self.done = False
        path = next_shape_path()
        self.shape = np.load(os.path.join(path, 'image.npy'))
        self.outline = np.load(os.path.join(path, 'outline.npy'))
        self.input = np.full_like(self.shape, False)
        self.grasp_points = []

        self.observation 
        ## grasp_points_coordinates, reconstruction_output, so a two layer image for each grasp points and output?

        return self.observation  # reward, done, info can't be included
    def render(self, mode='human'):
        ...
    def close (self):
        ...