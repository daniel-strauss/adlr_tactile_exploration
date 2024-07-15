import os
import gym
from gym import spaces
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from torch import from_numpy
import torch

'''
TODOs:
  - implement the stable baseline policies and all of that shit
  - optional: instead of numpy use torch for better performance 
'''

class ShapeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    res = 256  #todo, automate res extraction from data?

    max_steps = 10

    def __init__(self, rec_net, dataset, loss_func, reward_func):
        super(ShapeEnv, self).__init__()

        mid = int(self.res / 2)
        r = mid - 1
        self.c_rr, self.c_cc = ski.draw.circle_perimeter(mid, mid, r)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=len(self.c_cc)-1, shape=(2,), dtype=int)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.res, self.res, 2), dtype=np.float32)

        self.rec_net = rec_net  # reconstruction network
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.rec_net.to(self.device)
        self.rec_net.eval()

        self.dataset = dataset
        self.loss_func = loss_func  # loss function to evaluate reconstruction
        self.reward_func = reward_func  # reward_functions.py contains different reward functions for rl

        # Matplotlib figures for rendering
        plt.ion()
        self.fig, (self.ax_1, self.ax_2) = plt.subplots(1, 2, figsize=(12, 6))
        self.render_initialized = False
        plt.show()

        ######### on run variables #################
        self.done = None
        self.label = None  # 2d shape to be reconstructed
        self.outline_img = None  # image of outline
        self.grasp_points = None  # list of grasp points
        self.grasp_point_img = None  # grasp points feed to reconstruction network
        self.reconstruction_img = None  # output of network
        self.observation = None
        self.losses = []
        self.reward = None
        self.step_i = None
        self.info = None
        self.render_initialized = None
        self.rc_points = None # raycast points, needed for render
        self.rc_line = None

    def step(self, action):

        occurrences = []
        self.step_i += 1

        # 1. Get start and end point of ray
        alpha = action[0]
        beta = action[1]

        r1 = self.c_rr[alpha]
        c1 = self.c_cc[alpha]
        r2 = self.c_rr[beta]
        c2 = self.c_cc[beta]

        self.rc_points = np.array([[r1, c1], [r2, c2]])

        # 2. Cast a ray with anti-aliasing to prevent slip-throughs
        rr, cc, value = ski.draw.line_aa(r1, c1, r2, c2)
        self.rc_line = np.array([rr,cc,value])
        inters_i = np.argmax(self.outline_img[0, rr, cc] * value)

        # 3. Check if ray missed
        r_g, c_g, v_g = rr[inters_i], cc[inters_i], value[inters_i]
        if self.outline_img[0, r_g, c_g] * v_g < 0.01:
            occurrences.append('missed')
        else:
            self.grasp_points.append([r_g, c_g])
            self.grasp_point_img[0, r_g, c_g] = 1
            
        # 4. Infer reconstruction with new grasp point.
        loss, self.reconstruction_img = self.infer_reconstruction()
        self.losses.append(loss)

        # 5. Calculate reward (rel/abs decrease in loss)
        self.reward = self.reward_func(self.losses, occurrences)

        # 6. Update observation
        self.observation = self.pack_observation()

        # self.done = self.step_i == self.max_steps
        self.done = len(self.grasp_points) == self.max_steps

        return self.observation, self.reward, self.done, self.info

    def reset(self):
        self.step_i = 0
        self.done = False
        self.render_initialized = False

        sample = self.new_sample()

        self.label = sample['label']
        self.outline_img = self.p_list_to_img_array(sample['outline'].squeeze())
        self.grasp_points = []
        self.grasp_point_img = self.p_list_to_img_array(self.grasp_points)

        initial_loss = 0
        self.reconstruction_img = np.zeros((1, self.res, self.res))
        self.losses = [initial_loss]

        # grasp_point_image, reconstruction_output, so a two layer image for each grasp points and output
        self.observation = self.pack_observation()

        return self.observation  # reward, done, info can't be included

    def render(self, mode='human'):

        self.ax_1.clear()
        self.ax_1.imshow(self.convert_for_imshow(self.outline_img))

        if len(self.grasp_points) > 0:
            self.ax_1.plot(self.grasp_points[-1][0], self.grasp_points[-1][1], 'ro', label='Last Grasp Point')

        self.ax_1.plot(self.c_rr, self.c_cc, 'b.', markersize=1)

        self.ax_1.plot(self.rc_points[0,0], self.rc_points[0,1], 'go', label='Alpha')
        self.ax_1.plot(self.rc_points[1,0], self.rc_points[1,1], 'bo', label='Beta')
        self.ax_1.scatter(self.rc_line[0], self.rc_line[1], s=.5, c='r')

        self.ax_1.legend()

        self.ax_2.clear()
        self.ax_2.imshow(self.convert_for_imshow(self.add_zero_channel(self.observation)))
        self.fig.canvas.draw()
        plt.pause(.1)

    def close(self):
        '''
        Probably not much to do here, since we dont need to close down any processes or similar
        '''

    ############################# Helpfull Functions ############################

    def new_sample(self):
        index = np.random.randint(0, len(self.dataset))
        return self.dataset[index]

    # runs grasp points through reconstruction network and return loss and reconstruction
    def infer_reconstruction(self):
        reconstruction = self.rec_net(self.to_torch(self.grasp_point_img))
        loss = self.loss_func(reconstruction, self.to_torch(self.label))
        return loss.item(), self.from_torch(reconstruction)

    # Update observation
    def pack_observation(self):
        return self.two_img_to_one(self.grasp_point_img, self.reconstruction_img)

    # converts a list of points to image array, where each point has value one
    def p_list_to_img_array(self, p_list):
        if len(p_list) > 0 and len(p_list[0]) != 2:
            raise ValueError("point list has wrong shape, shape: " + str(p_list.shape))
        a = np.zeros((1, self.res, self.res)).astype(np.float32)
        if len(p_list) > 0:
            a[0, p_list[:, 0], p_list[:, 1]] = 1
        return a

    def add_zero_channel(self, a):
        zeros = self.add_color_dim(np.zeros(a.shape[1:]))
        return self.two_img_to_one(a, zeros)

    # converts imgarray to list of points
    @staticmethod
    def img_array_to_p_list(a):
        return np.argwhere(a > 0)

    def to_torch(self, a):
        if len(a.shape) == 2:
            b = self.add_color_dim(a)
        else:
            b = a.copy()
        return from_numpy(b.reshape(np.concatenate(([1], b.shape)))).to(self.device)

    @staticmethod
    def from_torch(a):
        return a[0].detach().numpy()

    # converts a n,m array to a n,m,1 array
    @staticmethod
    def add_color_dim(a):
        return a.reshape((1, a.shape[0], a.shape[1]))

    # converts array of shape n,m,c to shape c,n,m
    @staticmethod
    def convert_for_imshow(a):
        return a.transpose(2, 1, 0)

    @staticmethod
    def two_img_to_one(a, b):
        if a.shape[1:-1] != b.shape[1:-1]:
            print("WRONG SHAPES:")
            print(a.shape)
            print(b.shape)
            raise ValueError("images are not compatible, need same amount of rows and cols")
        return np.concatenate((a, b), axis=0)
