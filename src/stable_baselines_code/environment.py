import gymnasium as gym
from gymnasium import spaces
import numpy as np
import skimage as ski
from matplotlib import pyplot as plt
from torch import from_numpy
import torch

from outputs.plots_plakat.color_shemes import  main_colorscheme
from src.util_functions import from_torch, add_color_dim, two_img_to_one, convert_for_imshow, \
    add_zero_channel



class ShapeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'matrix'], "render_fps": 30}

    res = 256  #todo, automate res extraction from data_preprocessing?

    max_steps = 10
    def __init__(self, rec_net, dataset, reward_func, observation_1D=False, smoke=False):

        super(ShapeEnv, self).__init__()

        mid = int(self.res / 2)
        r = mid - 1
        self.c_rr, self.c_cc = ski.draw.circle_perimeter(mid, mid, r)

        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([len(self.c_cc) for _ in range(2)])

        # using image as input:
        self.observation_1D = observation_1D
        if self.observation_1D:
            self.observation_space = spaces.Box(low=0, high=255, shape=(1, self.res, self.res), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(2, self.res, self.res), dtype=np.uint8)

        self.rec_net = rec_net  # reconstruction network
        self.smoke = smoke
        self.dataset = dataset
        self.reward_func = reward_func  # reward_functions.py contains different reward functions for rl

        # Matplotlib figures for rendering
        self.fig, self.ax_1, self.ax_2 = None, None, None
        self.render_initialized = False  # call plt.show on_render, to not have the shitty plt window al the time


        self.total_steps = 0

        self.colorscheme = main_colorscheme

        ######### on run variables #################
        self.terminated = None
        self.truncated = None
        self.info = {}
        self.label = None  # 2d shape to be reconstructed
        self.outline_img = None  # image of outline
        self.grasp_points = None  # list of grasp points
        self.grasp_point_img = None  # grasp points feed to reconstruction network
        self.reconstruction_img = None  # output of network
        self.observation = None
        self.losses = []
        self.metrics = []
        self.reward = None
        self.step_i = None
        self.info = None
        self.rc_points = []  # raycast points, needed for render
        self.rc_line = None


    def step(self, action):

        occurrences = []
        self.step_i += 1
        self.total_steps += 1

        # 1. Get start and end point of ray
        alpha = action[0]
        beta = action[1]

        r1 = self.c_rr[alpha]
        c1 = self.c_cc[alpha]
        r2 = self.c_rr[beta]
        c2 = self.c_cc[beta]

        self.rc_points.append(np.array([[r1, c1], [r2, c2]]))

        # 2. Cast a ray with anti-aliasing to prevent slip-throughs
        rr, cc, value = ski.draw.line_aa(r1, c1, r2, c2)
        self.rc_line = np.array([rr, cc, value])
        inters_i = np.argmax(self.outline_img[0, rr, cc] * value > 0.01)

        # 3. Check if ray missed
        r_g, c_g, v_g = rr[inters_i], cc[inters_i], value[inters_i]
        if self.outline_img[0, r_g, c_g] * v_g < 0.01:
            occurrences.append('missed')
        elif self.grasp_point_img[0, r_g, c_g] == 1:
            occurrences.append('double')
        else:
            self.grasp_points.append([r_g, c_g])
            self.grasp_point_img[0, r_g, c_g] = 1

        # 4. Infer reconstruction with new grasp point.
        loss, metric, self.reconstruction_img = self.rec_net.infer(self.grasp_point_img, self.label)
        self.losses.append(loss)
        self.metrics.append(metric)

        # 5. Calculate reward (rel/abs decrease in loss)
        self.reward = self.reward_func(self.losses, self.metrics, occurrences)

        # 6. Update observation
        self.observation = self.pack_observation()

        self.terminated = self.step_i == self.max_steps
        # self.terminated = len(self.grasp_points) == self.max_steps

        self.info["losses"] = self.losses
        self.info["metrics"] = self.metrics
        self.info["reconstruction"] = self.reconstruction_img
        self.info['missed'] = 'missed' in occurrences
        self.info['double'] = 'double' in occurrences

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.info = {"losses": [], "rewards": []}

        self.step_i = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        sample = self.new_sample(options)

        self.label = sample['label']
        self.outline_img = self.p_list_to_img_array(sample['outline'].squeeze())
        self.grasp_points = []
        self.grasp_point_img = self.p_list_to_img_array(self.grasp_points)
        self.rc_points = []

        self.losses = []
        self.metrics = []

        self.reconstruction_img = np.zeros((1, self.res, self.res), 'f')

        # grasp_point_image, reconstruction_output, so a two layer image for each grasp points and output
        self.observation = self.pack_observation()

        return self.observation, self.info  # reward, done, info should be included


    def render(self, mode='human', all_rcs=False):
        if mode == 'matrix':
            self.matrix()
            return
        
        if not self.render_initialized:
            plt.ion()
            self.fig, (self.ax_1, self.ax_2) = plt.subplots(1, 2, figsize=(12, 6))
            plt.tight_layout()
            plt.show()
            self.render_initialized = True

        self.ax_1.clear()
        self.ax_1.imshow(convert_for_imshow(
            add_zero_channel(
                two_img_to_one(self.outline_img, self.reconstruction_img > 0.5)), cs = self.colorscheme))

        if len(self.grasp_points) > 0:
            gpa = np.array(self.grasp_points)
            self.ax_1.plot(gpa[-1][0], gpa[-1][1], 'o', color="black", label='last grasp point', zorder=5)
            self.ax_1.scatter(gpa[0:-1, 0], gpa[0:-1, 1], s=10, color=self.colorscheme[2], label= "grasp points", zorder=5)

        # plot circle
        self.ax_1.plot(self.c_rr, self.c_cc, 'o', color=self.colorscheme[3], markersize=1)

        self.ax_1.plot(self.rc_points[-1][0, 0], self.rc_points[-1][0, 1],'o',
                       color=self.colorscheme[2], label='alpha')
        self.ax_1.plot(self.rc_points[-1][1, 0], self.rc_points[-1][1, 1],'o',
                       color=self.colorscheme[3]*0.5, label='beta')
        self.ax_1.scatter(self.rc_line[0], self.rc_line[1], s=.5, color=self.colorscheme[3]*0.8,
                          label='current ray cast',zorder=3)

        if all_rcs:
            for i,[alpha, beta] in enumerate(self.rc_points[:-1]):
                if i == 0:
                    self.ax_1.plot([alpha[0], beta[0]], [alpha[1], beta[1]], color=self.colorscheme[3], linewidth=1,
                               zorder=2,label='ray casts')
                else:
                    self.ax_1.plot([alpha[0], beta[0]], [alpha[1], beta[1]], color=self.colorscheme[3], linewidth=1,
                                   zorder=2)

        self.ax_1.legend(loc="upper right")

        self.ax_2.clear()
        if self.observation_1D:
            self.ax_2.imshow(convert_for_imshow(self.observation, cs = self.colorscheme))
        else:
            self.ax_2.imshow(convert_for_imshow(add_zero_channel(self.observation), cs = self.colorscheme, bin=False))
            #self.ax_2.set_facecolor("w")
        self.fig.canvas.draw()
        plt.pause(.1)

    def close(self):
        '''
        Probably not much to do here, since we dont need to close down any processes or similar
        '''

    ############################# Helpfull Functions ############################

    def new_sample(self, options):
        if not options is None and 'index' in options:
            index = options['index']
        else:
            index = np.random.randint(0, len(self.dataset))
        return self.dataset[index]


    # runs grasp points through reconstruction network and return loss and reconstruction
    def infer_reconstruction(self):
        reconstruction = self.rec_net(self.to_torch(self.grasp_point_img))
        label = self.to_torch(self.label)
        loss = self.loss_func(reconstruction, label)

        rec = (reconstruction >= 0.5)
        lab = (label > 0)

        n = torch.logical_or(rec, lab).float().sum()
        metric = torch.logical_and(rec, lab).float().sum() * 100 / n

        return loss.item(), metric.item(), from_torch(reconstruction)


    # Update observation
    def pack_observation(self):
        if self.observation_1D:
            img = self.reconstruction_img * 255
        else:
            img = two_img_to_one(self.grasp_point_img, self.reconstruction_img) * 255
        return img.astype(np.uint8)

    # converts a list of points to image array, where each point has value one
    def p_list_to_img_array(self, p_list):
        if len(p_list) > 0 and len(p_list[0]) != 2:
            raise ValueError("point list has wrong shape, shape: " + str(p_list.shape))
        a = np.zeros((1, self.res, self.res)).astype(np.float32)
        if len(p_list) > 0:
            a[0, p_list[:, 0], p_list[:, 1]] = 1
        return a


    def to_torch(self, a):
        if len(a.shape) == 2:
            b = add_color_dim(a)
        else:
            b = a.copy()
        return from_numpy(b.reshape(np.concatenate(([1], b.shape)))).to(self.device)

    def num_pgs(self):
        return len(self.grasp_points)

    def matrix(self):
        if not self.render_initialized:
            plt.ion()
            self.fig, self.axs = plt.subplots(2, 5, figsize=(13, 5))
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            self.render_initialized = True
        
        x = (self.step_i-1) % 5

        if self.step_i < 6:
            y = 0
        else:
            y = 1

        self.axs[y, x].clear()
        self.axs[y, x].axis('off')
        self.axs[y, x].title.set_text(self.step_i)
        self.axs[y, x].imshow(convert_for_imshow(
            add_zero_channel(
                two_img_to_one(self.outline_img, self.reconstruction_img > 0.5)), cs = self.colorscheme))

        if len(self.grasp_points) > 0:
            gpa = np.array(self.grasp_points)
            self.axs[y, x].plot(gpa[-1][0], gpa[-1][1], 'o', color="black", label='last grasp point', zorder=5)
            self.axs[y, x].scatter(gpa[0:-1, 0], gpa[0:-1, 1], s=10, color=self.colorscheme[2], label= "grasp points", zorder=5)

        # plot circle
        self.axs[y, x].plot(self.c_rr, self.c_cc, 'o', color=self.colorscheme[3], markersize=1)

        self.axs[y, x].plot(self.rc_points[-1][0, 0], self.rc_points[-1][0, 1],'o',
                       color=self.colorscheme[2], label='alpha')
        self.axs[y, x].plot(self.rc_points[-1][1, 0], self.rc_points[-1][1, 1],'o',
                       color=self.colorscheme[3]*0.5, label='beta')
        self.axs[y, x].scatter(self.rc_line[0], self.rc_line[1], s=.5, color=self.colorscheme[3]*0.8,
                          label='current ray cast',zorder=3)
        self.fig.canvas.draw()
        plt.pause(.1)
            