import time
import os
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO
from neural_nets.rec_net import RecNet
from neural_nets.utility_functions import load_rl_data

from stable_baselines_code.callback import TensorboardCallback
from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import basic_reward
from torch.utils.data import Subset

model_names = ['rew500k9', 'obs500k9']
model_paths = [os.path.join('rl_models/', name) for name in model_names]

rec_net = RecNet()
train_set, eval_set, test_set = load_rl_data(transform=None)
dummy = Subset(eval_set, range(5))

dataset = dummy
env = ShapeEnv(rec_net, dataset, basic_reward)

data = {}

for name in model_paths:
    model = PPO.load(name, env)
    metrics = np.zeros(10)
    options = {}
    n = len(dataset)
    for i in range(n):
        options['index'] = i
        observation, info = env.reset(options=options)
        for j in range(10):
            action, _states = model.predict(observation, deterministic=False)  # Sample random action
            observation, reward, done, truncated, info = env.step(action)
            metrics[j] += reward
    metrics /= n
    data[name] = metrics
env.close()

print(data)
print(data['rl_models/rew500k9'])