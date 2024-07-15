import time
import torch
from ray.cloudpickle import pickle
from torch import nn
import numpy as np
from neural_nets.models.unet import UNet3
from neural_nets.utility_functions import load_data

from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import basic_reward
from stable_baselines3.common.env_checker import check_env

trial_path = './neural_nets/best_trial.pkl'
model_path = './neural_nets/trained_rec.pkl'

with open(trial_path, 'rb') as pickle_file:
    best_trial = pickle.load(pickle_file)
pickle_file.close()

with open(model_path, 'rb') as pickle_file:
    states = pickle.load(pickle_file)
pickle_file.close()

config = best_trial.config
rec_net = UNet3(config)
rec_net.load_state_dict(states['net_state_dict'])

train_set, eval_set, test_set = load_data(transform=None)
dataset = torch.utils.data.ConcatDataset([train_set, eval_set])

env = ShapeEnv(rec_net, dataset, nn.BCELoss(), basic_reward)
env.reset()

while True:
    action = env.action_space.sample()  # Sample random action
    observation, reward, done, info = env.step(action)
    print(reward)
    env.render()
    time.sleep(0.1)
    if done:
        break
env.close()