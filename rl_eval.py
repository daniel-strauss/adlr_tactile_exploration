import time
from stable_baselines3 import A2C, PPO
from neural_nets.rec_net import RecNet
from neural_nets.utility_functions import load_rl_data

from stable_baselines_code.callback import TensorboardCallback
from stable_baselines_code.environment import ShapeEnv
from stable_baselines_code.reward_functions import basic_reward, complex_reward, improve_reward

tensorboard_path = "./rl_runs/"


# use dummy rec net to save ram, for testing
use_dummy_rec_net = False
show_example_run = False

trial_path = './neural_nets/best_trial.pkl'
rec_net_path = './neural_nets/trained_rec.pkl'

rec_net = RecNet(dummy=use_dummy_rec_net)

train_set, eval_set, test_set = load_rl_data(transform=None)


env = ShapeEnv(rec_net, train_set, complex_reward, smoke=False)
observation, info = env.reset()

#  stable baseline model
#filename = 'daniel/daniel/punish_miss_free_rays/obs500k7.zip'
#model = PPO.load(filename, env)
model = PPO.load('rl_models/rew500k9', env)


# example run
i = 0
while i < 5:
    action, _states = model.predict(observation, deterministic=False)  # Sample random action
    observation, reward, done, truncated, info = env.step(action)
    print(reward)
    env.render()
    time.sleep(0.5)
    if done:
        observation, info = env.reset()
        i+=1
env.close()