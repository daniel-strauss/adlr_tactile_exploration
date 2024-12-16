import pickle
import torch
from neural_nets import utility_functions

from ray import tune
from tune.schedulers import ASHAScheduler


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

trial_path='./neural_nets/best_trial.pkl'


with open(trial_path, 'rb') as pickle_file:
    best_trial = CPU_Unpickler(pickle_file).load()
    pickle_file.close()
best_config = best_trial.config

new_config = {
    'epochs': 15,
    'num_workers': 8
}

config = best_config | new_config
scaling_config = ScalingConfig(num_workers=config['num_workers'], use_gpu=True)
run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))

