import os
from data.reconstruction_dataset import *
from neural_nets.utility_functions import standard_config, load_data, train_reconstruction
import ray
from sys import getsizeof
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ray.cloudpickle as pickle
from pathlib import Path
from neural_nets.models.unet import UNet3
import pickle

data_dir = './datasets/2D_shapes'
cpus = 10
gpus = 1
max_epochs = 10
num_samples = 100

train_set, eval_set, _ = load_data()
image_resolution = train_set[0]['image'].shape[1]
max_unet_depth = int(np.log2(image_resolution))

# set of hyperparameters that will be searched in the given ranges, merged with the standard_config 
config = {
    "lr": tune.loguniform(1e-7, 1e-2),
    "depth": tune.choice([i for i in range(3,max_unet_depth+1-2)]),
    "channels": tune.choice([2 ** i for i in range(4,9)]),
    "batch_size": tune.choice([2 ** i for i in range(2,5)]),
    "epochs": max_epochs,
    "num_workers": cpus,
    "smoke": False
}
config = standard_config | config

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

algo = TuneBOHB()
scheduler = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=max_epochs,
    reduction_factor=4,
    stop_last_trials=False,
)

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(train_reconstruction, train_set=train_set, eval_set=eval_set),
        resources={'cpu': cpus, 'gpu': gpus}
    ),
    tune_config=tune.TuneConfig(
        metric='loss',
        mode='min',
        search_alg=algo,
        scheduler=scheduler,
        num_samples=num_samples,
    ),
    run_config=train.RunConfig(
        name="bohb",
        stop={"training_iteration": max_epochs},
    ),
    param_space=config,
)
results = tuner.fit()

best_trial = results.get_best_result()
print(f"Best trail with a validation loss of {best_trial.metrics['loss']}")
print(f"Config: {best_trial.config}")
print(f"Checkpoint: {best_trial.checkpoint}")

with open('../best_trial.pckl', 'wb') as f:
    pickle.dump(best_trial, f)