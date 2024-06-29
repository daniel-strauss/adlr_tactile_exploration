import os
from data.reconstruction_dataset import *
from neural_nets.utility_functions import *
import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ray.cloudpickle as pickle
from pathlib import Path
from neural_nets.models.unet import UNet3

data_dir = './datasets/2D_shapes'
cpus = 8
gpus = 1
num_samples = 2
bohb = True


train_set, test_set = load_data(data_dir)

image_resolution = train_set[0]['image'].shape[1]
max_unet_depth = int(np.log2(image_resolution))

# config is the set of params, that will be searched, they got to ghave the same key names, as variables in THparams
config = {
    "lr": tune.loguniform(1e-7, 1e-2),
}

h = Hparams()
h.model = UNet3
h.batch_size = 64
trainer = Trainer(h, train_set)
ray.put(trainer)

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

if bohb:
    algo = TuneBOHB()
    algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=100,
        reduction_factor=4,
        stop_last_trials=False,
    )
    tuner = tune.Tuner(
        trainer.train,
        tune_config=tune.TuneConfig(
            metric='loss',
            mode='min',
            search_alg=algo,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="bohb_exp",
            stop={"training_iteration": 100},
        ),
        param_space=config,
)
results = tuner.fit()

scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=h.epochs,
        grace_period=1,
        reduction_factor=2,
)

result = tune.run(
        trainer.train,
        resources_per_trial={"cpu": cpus, "gpu": gpus},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

best_trial = result.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")