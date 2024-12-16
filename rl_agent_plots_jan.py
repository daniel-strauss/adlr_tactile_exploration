import os
import torch
from matplotlib import pyplot as plt
from stable_baselines3 import A2C, PPO
from torch import nn
import numpy as np
from neural_nets.rec_net import RecNet
from neural_nets.utility_functions import load_rl_data

import pickle
import io

from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import basic_reward, complex_reward
from stable_baselines_code.example_usage_environment import DummyRecNet # importing dummy net for test purposes

pol_abs = 'rl_models/obs500k9.zip'
pol_diff = 'rl_models/rew500k9.zip'

tensorboard_path = "./rl_runs/"

gp_terminate = True # if gp_terminate generate plots until 10 gps ahve been reached
version = "Abs" # name of rl model
filename = pol_abs

name = "version:" + version +"_gp_terminate:" + str(gp_terminate)
os.makedirs("plots_plakat/temp/rl_plots/"+name, exist_ok=True)

num_samples = 15

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

trial_path = './neural_nets/best_trial.pkl'
rec_net_path = './neural_nets/trained_rec.pkl'

options = {"index" : 0}

rec_net = RecNet()

train_set, eval_set, test_set = load_rl_data(transform=None)
env = ShapeEnv(rec_net, test_set, complex_reward, smoke=False)
observation, _ = env.reset(options=options)

# example satble baseline model
if not filename == "":
    model = PPO.load(filename, env)

# example run
iter = 0
step = 0
while iter < num_samples:
    if not filename == "":
        action, _states = model.predict(observation, deterministic=False)  # Sample random action
    else:
        action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(reward)
    env.render(all_rcs=True)
    gps = env.num_pgs()
    if (done and gp_terminate) or (gps==10 and not gp_terminate):
        step = 0
        iter += 1
        options = {'index':iter}
        observation, _ = env.reset(options=options)
    plt.savefig("./plots_plakat/temp/rl_plots/"+name + '/iter_%i_step_%i_gp%i.png' %(iter,step,gps), dpi=300)
    step += 1
env.close()