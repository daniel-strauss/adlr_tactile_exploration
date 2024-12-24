import time
from datetime import datetime

from stable_baselines3 import PPO
from src.neural_nets.rec_net import RecNet
from src.neural_nets.utility_functions import load_rl_data
from stable_baselines3.common.evaluation import evaluate_policy

from src.stable_baselines_code.callback import TensorboardCallback
from src.stable_baselines_code.environment import ShapeEnv
from src.stable_baselines_code.reward_functions import improve_reward

tensorboard_path = "./rl_runs/" + f'RL_{datetime.now().strftime("%Y-%m-%d--%H:%M:%S")}'

debug_mode = False

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


trial_path = '../../outputs/reconstruction_models/best_trial.pkl'
rec_net_path = '../../outputs/reconstruction_models/trained_rec.pkl'

rec_net = RecNet(dummy=use_dummy_rec_net)

train_set, eval_set, test_set = load_rl_data(transform=None)

smoke = False
observation_1D = False
reward = improve_reward

env = ShapeEnv(rec_net, train_set, reward, smoke=smoke, observation_1D=observation_1D)
obs, info = env.reset()

eval_env = ShapeEnv(rec_net, eval_set, reward, smoke=smoke, observation_1D=observation_1D)
eval_env.reset()

if debug_mode:
    n_steps = 2000
    learn_steps = 20000
    iter = 10
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



if not debug_mode:
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Before Training: {mean_reward} +- {std_reward}')

for i in range(iter):
    model.learn(learn_steps, tb_log_name='rew500k', progress_bar=True, callback=TensorboardCallback())
    model.save('./daniel/rew500k' + str(i))

if not debug_mode:
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'After Training: {mean_reward} +- {std_reward}')

# example run
if show_example_run:
    run_example(40)
