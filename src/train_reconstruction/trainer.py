import pickle
import torch

from src.neural_nets.utility_functions import standard_config, load_data, train_reconstruction

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            # Neural nets was moved to src, so we need to change the module path
            if module == 'neural_nets.models.unet':
                module = 'src.neural_nets.models.unet'
            # The default behaviour, locate and load the class
            return super().find_class(module, name)

trial_path= './outputs/reconstruction_models/best_trial.pkl'


with open(trial_path, 'rb') as pickle_file:
    best_trial = CPU_Unpickler(pickle_file).load()
    pickle_file.close()
best_config = best_trial.config

new_config = {
    'epochs': 15,
    'num_workers': 8
}

config = best_config | new_config
#scaling_config = ScalingConfig(num_workers=config['num_workers'], use_gpu=True)
#run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))
print(config)
train_set, eval_set, _ = load_data()
train_reconstruction(config, train_set, eval_set, print_num_parameters=True)