

import os
import time
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, PPO
from torch import nn
import numpy as np
from neural_nets.models.unet import UNet3
from neural_nets.rec_net import RecNet
from neural_nets.utility_functions import load_rl_data

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

import pickle
import io

from stable_baselines_code.callback import TensorboardCallback
from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import basic_reward, complex_reward
from stable_baselines_code.example_usage_environment import DummyRecNet # importing dummy net for test purposes




tensorboard_path = "./rl_runs/"

gp_terminate = False # if gp_terminate generate plots until 10 gps ahve been reached
version = "random_grasp_points"#punich_miss__free_rays" # name of rl model
#filename to rl agent, if none, random policy will be used
filename = ''#'rl_models/rl_models/punish_miss_free_rays/obs500k7.zip'

name = "version:" + version +"__gp_terminate:" + str(gp_terminate)


os.makedirs("plots_plakat/rl_plots/"+name, exist_ok=True)

# use dummy rec net to save ram, for testing
use_dummy_rec_net = False

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

trial_path = './neural_nets/best_trial.pkl'
rec_net_path = './neural_nets/trained_rec.pkl'

options = {"index" : 0}

rec_net = RecNet(dummy=use_dummy_rec_net)


train_set, eval_set, test_set = load_rl_data(transform=None)

env = ShapeEnv(rec_net, eval_set, complex_reward, smoke=False)
observation, _ = env.reset(options=options)
print("SHAPEW, ", observation.shape)

# example satble baseline model
if not filename == "":
    model = PPO.load(filename, env)

# example run
iter = 0
step = 0
while iter < 15:
    if not filename == "":
        action, _states = model.predict(observation, deterministic=False)  # Sample random action
    else:
        action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(reward)
    env.render()
    #time.sleep(0.5)
    gps = env.num_pgs()
    if (done and gp_terminate) or (gps==10 and not gp_terminate):
        step = 0
        iter += 1
        options = {'index':iter}
        observation, _ = env.reset(options=options)


    plt.savefig("./plots_plakat/rl_plots/"+name + '/iter_%i_step_%i_gp%i.pdf' %(iter,step,gps))
    step += 1

env.close()