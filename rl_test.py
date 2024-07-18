import time
from datetime import datetime

import torch
from stable_baselines3 import A2C, PPO
from torch import nn
import numpy as np
from neural_nets.models.unet import UNet3
from neural_nets.utility_functions import load_rl_data
from stable_baselines3.common.evaluation import evaluate_policy

import pickle
import io

from stable_baselines_code.callback import TensorboardCallback
from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import basic_reward, complex_reward
from stable_baselines_code.example_usage_environment import DummyRecNet  # importing dummy net for test purposes

tensorboard_path = "./rl_runs/" + f'RL_{datetime.now().strftime("%Y-%m-%d--%H:%M:%S")}'

debug_mode = True

# use dummy rec net to save ram, for testing
use_dummy_rec_net = debug_mode
show_example_run = debug_mode


def run_example(n):
    print("Example Run")
    for i in range(n):
        action = env.action_space.sample()  # Sample random action
        observation, reward, done, truncated, info = env.step(action)
        print(reward)
        env.render()
        time.sleep(0.5)
        if done:
            env.reset()
    env.close()


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

smoke = False
observation_1D = True
reward = complex_reward

env = ShapeEnv(rec_net, train_set, nn.BCELoss(), reward, smoke=smoke, observation_1D=observation_1D)
env.reset()

eval_env = ShapeEnv(rec_net, eval_set, nn.BCELoss(), reward, smoke=smoke, observation_1D=observation_1D)
eval_env.reset()

if debug_mode:
    n_steps = 2
    learn_steps = 2
    iter = 0
else:
    n_steps = 2000
    learn_steps = 50000
    iter = 10

# example satble baseline model
model = PPO("CnnPolicy", env, verbose=1,
            tensorboard_log=tensorboard_path,
            learning_rate=3e-5,
            ent_coef=0.01,
            n_steps=n_steps,
            batch_size=50)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'Before Training: {mean_reward} +- {std_reward}')

for i in range(iter):
    model.learn(learn_steps, tb_log_name='500K', progress_bar=True, callback=TensorboardCallback())
    model.save('./rl_models/500k' + str(i))

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'After Training: {mean_reward} +- {std_reward}')

# example run
if show_example_run:
    run_example(40)
