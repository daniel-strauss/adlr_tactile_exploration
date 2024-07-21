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
debug = False


if debug:
    model_names = ['daniel_o2/complex_after_free']#, 'daniel_o2/diff_after_free', 'jan/rew500k9', 'jan/obs500k9']
    max_n = 300
else:
    model_names = ['daniel_o2/complex_after_free', 'daniel_o2/diff_after_free', 'jan/rew500k9', 'jan/obs500k9']
model_paths = [os.path.join('rl_models/', name) for name in model_names]
save_path = 'plots_plakat/plot_data/rl_policies_statistics_gp.pkl'


#if os.path.isfile(save_path):
#    with open(save_path, 'rb') as f:
#        data = pickle.load(f)
#        f.close()
#else:
#    data = {}
data = {}

max_grasp = 15

rec_net = RecNet(dummy=debug, cuda=not debug)
#rec_net.eval()


train_set, eval_set, test_set = load_rl_data(transform=None)
dataset = test_set

env = ShapeEnv(rec_net, dataset, basic_reward)

name = 'random'
if skip and name in data:
    print(f'Model {name} already evaluted, skipping.')
else:
    options = {}

    n = max_n if debug else len(dataset)
    metrics = np.zeros((max_grasp, n))
    for i in tqdm.tqdm(range(n), name):
        options['index'] = i
        observation, info = env.reset(options=options)
        while env.num_pgs() < max_grasp:
            action, _states = model.predict(observation, deterministic=False)  # Sample random action
            observation, reward, done, truncated, info = env.step(action)
            if not info['missed']:
                metrics[env.num_pgs()-1,i] = reward
    
    mean = np.mean(metrics, axis=1)
    std = np.std(metrics, axis=1)
    data[name] = np.stack((mean, std), axis=1)
    data[name + '_metric'] = metrics

env.close()

name = 'random'
if skip and name in data:
    print(f'Model {name} already evaluted, skipping.')
else:
    options = {}
    n = max_n if debug else len(dataset)
    metrics = np.zeros((max_grasp, n))
    print(metrics.shape)

    for i in tqdm.tqdm(range(n), name):
        options['index'] = i
        observation, info = env.reset(options=options)
        while env.num_pgs() < max_grasp:
            action, _states = model.predict(observation, deterministic=False)  # Sample random action
            observation, reward, done, truncated, info = env.step(action)

            if not info['missed']:
                metrics[env.num_pgs()-1, i] = reward
        #print(metrics[-1,i] > metrics[-2,i])


    mean = np.mean(metrics, axis=1)
    std = np.std(metrics, axis=1)
    data[name] = np.stack((mean, std), axis=1)
    data[name + '_metric'] = metrics


with open(save_path, 'wb') as f:
    pickle.dump(data, f)
    f.close()