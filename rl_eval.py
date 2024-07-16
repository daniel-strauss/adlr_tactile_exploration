import time
from datetime import datetime

import torch
from stable_baselines3 import A2C, PPO
from torch import nn
import numpy as np
from neural_nets.models.unet import UNet3
from neural_nets.utility_functions import load_data

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure

import pickle
import io

from stable_baselines_code.callback import TensorboardCallback
from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import basic_reward
from stable_baselines_code.example_usage_environment import DummyRecNet # importing dummy net for test purposes


tensorboard_path = "./rl_runs/"


# use dummy rec net to save ram, for testing
use_dummy_rec_net = False
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

train_set, eval_set, test_set = load_data(transform=None)
dataset = torch.utils.data.ConcatDataset([train_set, eval_set])

env = ShapeEnv(rec_net, dataset, nn.BCELoss(), basic_reward, smoke=True)
observation = env.reset()

# example satble baseline model

model = PPO.load('smoke_test', env)

# example run

while True:
    action, _states = model.predict(observation)  # Sample random action
    observation, reward, done, truncated, info = env.step(action)
    print(reward)
    env.render()
    time.sleep(0.5)
    if done:
        break
env.close()