import os
import gym
from gym import spaces
import numpy as np
import skimage as ski

def next_shape_path():
    return './datasets/2D_shapes/bottle/1a7ba1f4c892e2da30711cdbdbc739240'


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

        alpha = action[0] * np.pi
        beta = action[1] * np.pi
        r = 128

        x1 = round(r * np.cos(alpha)) + r
        y1 = round(r * np.sin(alpha)) + r
        x2 = round(r * np.cos(beta)) + r
        y2 = round(r * np.sin(beta)) + r

        if x1 == x2 and y1 == y2:
            ...
            #TODO

        rr, cc = ski.draw.line(x1, y1, x2, y2)

        """
        TODO:

        1, Cast ray based on both angles.
        2. Calculate intersection pixel of outline, handle case if object is missed (negative Reward?)
        3. Infer reconstruction with new grasp point.
        4. Calculate reward (rel/abs decrease in loss)
        5. Update observations

        """
        return self.observation, self.reward, self.done, self.info
    
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
        '''
        TODO:

        - Rendering of grasping points, rays, shape prediciton, actual shape
        - Color coding

        '''
    def close (self):
        '''
        Probably not much to do here, since we dont need to close down any processes or similar
        '''