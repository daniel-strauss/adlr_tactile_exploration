


import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from torch.utils.data import Subset

from src.neural_nets.rec_net import RecNet
from src.neural_nets.utility_functions import load_rl_data
from src.stable_baselines_code.environment import ShapeEnv
from src.stable_baselines_code.reward_functions import complex_reward



# --- User-editable variables ---
# Set these variables to configure the demo:
# - use_dummy_recnet: Set to True to use a dummy RecNet (no model files needed, for quick testing)
# - rec_net_trial_path: Path to RecNet trial file
# - rec_net_states_path: Path to RecNet states file
# - policy_path: Path to RL policy

use_dummy_recnet = True  # Set True for dummy RecNet (quick test, no model files needed)
rec_net_trial_path = './outputs/reconstruction_models/best_trial.pkl'
rec_net_states_path = './outputs/reconstruction_models/trained_rec.pkl'
policy_path = './outputs/rl_models/daniel/punish_miss_free_rays/obs500k9'  # adjust if needed

# --- Output dir for video frames ---
video_dir = 'outputs/plots_plakat/temp/rl_demo_video'
os.makedirs(video_dir, exist_ok=True)

# --- Load test set (small subset for demo) ---
_, _, test_set = load_rl_data(transform=None)
test_set = Subset(test_set, [36, 71, 354, 112, 149])
num_samples = len(test_set)


# --- Load RecNet (reconstruction net) ---
if use_dummy_recnet or not (os.path.exists(rec_net_trial_path) and os.path.exists(rec_net_states_path)):
    print("[INFO] Using dummy RecNet (no model files required).")
    rec_net = RecNet(dummy=True, cuda=False)
else:
    rec_net = RecNet(trial_path=rec_net_trial_path, states_path=rec_net_states_path, dummy=False, cuda=False)

# --- Create environment ---
env = ShapeEnv(rec_net, test_set, complex_reward, smoke=False)
observation, _ = env.reset(options = {"index" : 0})

# --- Load PPO policy ---
model = PPO.load(policy_path, env=env)

# --- Run agent and save frames ---
gp_terminate = False # if gp_terminate generate plots until 10 gps ahve been reached
# example run
iter = 0
step = 0
while iter < num_samples:
    
    action, _states = model.predict(observation, deterministic=False)  # Sample action
    observation, reward, done, truncated, info = env.step(action)
    env.render(all_rcs=True)

    gps = env.num_pgs()
    if (done and gp_terminate) or (gps==10 and not gp_terminate):
        step = 0
        iter += 1
        options = {'index':iter}
        observation, _ = env.reset(options=options)
    step += 1
env.close()

print("Demo complete. Frames saved to:", video_dir)
