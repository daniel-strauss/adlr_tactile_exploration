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

data_dir = './datasets/2D_shapes'
cpus = 2
gpus = 1
max_epochs = 10
num_samples = 10

# set of hyperparameters that will be searched in the given ranges, merged with the standard_config 
config = {
    "lr": tune.loguniform(1e-7, 1e-2),
    "epochs": max_epochs,
    "num_workers": cpus,
    "batch_size": 16,
    "smoke": True
}
config = standard_config | config

dataset, _ = load_data()

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
        tune.with_parameters(train_reconstruction, dataset=dataset),
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
        name="bohb_exp",
        stop={"training_iteration": max_epochs},
    ),
    param_space=config,
)
results = tuner.fit()

best_trial = results.get_best_result()
print(f"Best trial config: {best_trial.config}")