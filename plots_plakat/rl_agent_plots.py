

import os
import time
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, PPO
from torch import nn
import numpy as np
from neural_nets.models.unet import UNet3
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
version = "punich_miss__free_rays" # name of rl model
#filename to rl agent, if none, random policy will be used
filename = 'rl_models/rl_models/punish_miss_free_rays/obs500k7.zip'

name = "version:" + version +"__gp_terminate:" + str(gp_terminate)


os.makedirs("plots_plakat/rl_plots/"+name, exist_ok=True)

# use dummy rec net to save ram, for testing
use_dummy_rec_net = True
show_example_run = False

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

trial_path = './neural_nets/best_trial.pkl'
rec_net_path = './neural_nets/trained_rec.pkl'


if not use_dummy_rec_net:
    with open(trial_path, 'rb') as pickle_file:
        best_trial = CPU_Unpickler(pickle_file).load()
    pickle_file.close()

    with open(rec_net_path, 'rb') as pickle_file:
        states = CPU_Unpickler(pickle_file).load()
    pickle_file.close()

    config = best_trial.config
    rec_net = UNet3(config)
    rec_net.load_state_dict(states['net_state_dict'])
else:
    rec_net = DummyRecNet()

train_set, eval_set, test_set = load_rl_data(transform=None)

env = ShapeEnv(rec_net, eval_set, nn.BCELoss(), complex_reward, smoke=False)
observation, _ = env.reset()

# example satble baseline model
if not filename is None:
    model = PPO.load(filename, env)

# example run
iter = 0
step = 0
while True:
    if not filename is None:
        action, _states = model.predict(observation, deterministic=False)  # Sample random action
    else:
        action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(reward)
    env.render()
    time.sleep(0.5)
    if done:
        observation, _ = env.reset()
        step = 0

    plt.savefig("./plots_plakat/rl_plots/"+name + '/iter_%i_step_%i_gp%i.pdf' %(iter,step,2))
    step += 1
    iter += 1
env.close()