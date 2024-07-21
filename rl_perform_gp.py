import os
import numpy as np
import pickle
import tqdm
from stable_baselines3 import A2C, PPO
from neural_nets.rec_net import RecNet
from neural_nets.utility_functions import load_rl_data

from stable_baselines_code.callback import TensorboardCallback
from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import basic_reward
from torch.utils.data import Subset

skip = False
model_names = ['daniel_o2/complex_after_free', 'daniel_o2/diff_after_free',
               'jan/rew500k9', 'jan/obs500k9']
model_paths = [os.path.join('rl_models/', name) for name in model_names]
save_path = 'rl_models/statistics_gp.pkl'

if os.path.isfile(save_path):
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
        f.close()
else:
    data = {}


rec_net = RecNet(dummy=False)

train_set, eval_set, test_set = load_rl_data(transform=None)
dataset = test_set

env = ShapeEnv(rec_net, dataset, basic_reward)

for name in model_paths:
    print("Testing model: ", name)
    if skip and name in data:
        print(f'Model {name} already evaluted, skipping.')
        continue
    model = PPO.load(name, env)
    options = {}
    n = len(dataset)
    metrics = np.zeros((10, n))
    for i in tqdm.tqdm(range(n), name):
        options['index'] = i
        observation, info = env.reset(options=options)
        while env.num_pgs() < 10:
            action, _states = model.predict(observation, deterministic=False)  # Sample random action
            observation, reward, done, truncated, info = env.step(action)
            if not info['missed']:
                metrics[env.num_pgs()-1,i] += reward
    
    mean = np.mean(metrics, axis=1)
    std = np.std(metrics, axis=1)
    data[name] = np.stack((mean, std), axis=1)


env.close()

name = 'random'
if skip and name in data:
    print(f'Model {name} already evaluted, skipping.')
else:
    options = {}
    n = len(dataset)
    metrics = np.zeros((10, n))
    print(metrics.shape)
    for i in tqdm.tqdm(range(n), name):
        options['index'] = i
        observation, info = env.reset(options=options)
        while env.num_pgs() < 10:
            action, _states = model.predict(observation, deterministic=False)  # Sample random action
            observation, reward, done, truncated, info = env.step(action)
            if not info['missed']:
                metrics[env.num_pgs(), i] += reward

    mean = np.mean(metrics, axis=1)
    std = np.std(metrics, axis=1)
    data[name] = np.stack((mean, std), axis=1)

with open(save_path, 'wb') as f:
    pickle.dump(data, f)
    f.close()