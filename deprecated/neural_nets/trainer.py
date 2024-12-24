'''
Class that provides:
    - instancing neural network, optimizers, dataloaders,
        everything else from hyperparameters in dict config
    - training function
    - progress logging, using tensorboard, automatically opens tap for logging
    - usage of ray, ray is a library that searches for good hparams

    The purpose of this class is to avoid multiple

'''
import time
import warnings
import datetime
import tempfile
from typing import Callable
# import matplotlib.pyplot as plt
import torch.cuda
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle


# Function to count the number of parameters
def count_parameters(model, verbose=False):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if verbose:
                print(f"Layer: {name} | Size: {param.size()} | Parameters: {param.numel()}")
            total_params += param.numel()
    print(f"Total trainable parameters: {total_params}")


# todo think about putting NonTHparams and Hparams into one class, when you are not tired anymore
class NonTHparams():
    # class, that specifies the key, value type pairs for
    # the non-tunable hparams.
    # If a hparam appears here, that you think is tunable, you can move it
    # to THparams

    epochs: int = 50

    # proportion of dataset to be training data_preprocessing
    train_prop: float = 0.8

    #logging
    # amount of val calculations per epoch
    log_val_freq: int = 1
    # setting period to -1 logs every epoch end
    log_train_period: int = 100
    print_log: bool = True
    board_log: bool = True  # log to tensorboard


class THparams():
    # class, that specifies the key, value type pairs for
    # the tunable hparams.
    # If a hparam appears here, that you think is non-tunable, you can move it
    # to NonTHparams
    # todo add beta variable for adams and maybe wheight regulization?

    # optimization hparams,
    lr: float = 1e-5
    optimizer: torch.optim.Optimizer
    loss_func: Callable

    # network hparams
    model: nn.Module
    weight_init: Callable
    depth: int = 5
    channels: int = 64
    first_kernel_size:int = 3

    def __init__(self, params: dict = None):
        if not params is None:
            self.override_from_dict(params)

    def override_from_dict(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Trainer:
    # dictionary containing all non-tunable hyperparameter, e.g. all hyperparameters,
    # that shall not be searched in a hypothetical parameter search
    nt_h: NonTHparams
    t_h: THparams

    train_dataset: Dataset
    val_dataset: Dataset

    def __init__(self, nt_hparams: NonTHparams, t_hparams:THparams, dataset: Dataset):

        # non tunable hyperparameters
        self.nt_h = nt_hparams
        self.t_h = t_hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.init_datasets()

        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available, device used instead: " + str(self.device))

        if self.nt_h.board_log:
            print("During training, progress will be logged to tensorboard. Go to project folder, activate appropriate"
                  "conda environment and run 'tensorboard --logdir runs/' to see the logs.")

    def init_datasets(self):
        # Define the sizes for train, validation, and test
        train_size = int(self.nt_h.train_prop * len(self.dataset))
        val_size = len(self.dataset) - train_size

        # Split the dataset
        self.val_dataset, self.train_dataset = random_split(self.dataset, [val_size, train_size])

    def train_from_dict(self, t_hparams_dict: dict):
        # overrides self.t_h from that dict and then trains
        # todo: find nicer way to do this, restore t_h afterwards maybe?
        self.t_h.override_from_dict(t_hparams_dict)
        self.train()

    def train(self):
        # t_h: class containing all tunable hparams, e.g. learnign rate and shit like that
        # all parameters that should be searched in a hypothetical parameter search are in
        # the class of t_h

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            warnings.warn("Cuda is not available, device used instead: " + str(self.device))
        else:
            print("Hurray! GPU available.")

        # Initialize model, loss function, and optimizer
        model = self.t_h.model(self.t_h).to(self.device)
        if self.nt_h.print_log:
            count_parameters(model)

        criterion = self.t_h.loss_func
        optimizer = self.t_h.optimizer(model.parameters(), lr=self.t_h.lr)

        # initialize model parameters
        model.apply(self.t_h.weight_init)

        # initialize dataloaders
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.t_h.batch_size,
                                  shuffle=True,
                                  num_workers=0)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.t_h.batch_size,
                                shuffle=True,
                                num_workers=0)

        num_t_batches = len(train_loader)

        if self.nt_h.board_log:
            # Set up TensorBoard
            writer = SummaryWriter(f'runs/U-Net_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

        ##############################################################
        # Try to load ray - checkpoint if available
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data_preprocessing.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                model.load_state_dict(checkpoint_state["net_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        else:
            start_epoch = 0

        #######################################################################
        ####################### TRAIN LOOP ####################################
        #######################################################################

        # used to measure time proportion for calculating for val loss
        train_period_stime_val = time.time()
        # used to measure time proportion for adding images to writer
        train_period_stime_train = time.time()

        data_loading_time = 0

        for epoch in range(start_epoch, self.nt_h.num_epochs):
            model.train()  # put model into training mode
            sum_train_loss = 0.0

            for i, batch in enumerate(train_loader):
                # todo: output should be segmentation map, read an article about it

                # measure time to load data_preprocessing
                st_data_load = time.time()
                inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                data_loading_time += time.time() - st_data_load

                # the actual training
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                sum_train_loss += loss.item()
                # log training data_preprocessing every log_train_period batches
                if ((i + epoch * num_t_batches) % self.nt_h.log_train_period == self.nt_h.log_train_period - 1
                        or (self.nt_h.log_train_period == -1 and i == num_t_batches - 1)):

                    train_duration = time.time() - train_period_stime_train
                    log_starttime = time.time()

                    # log training loss
                    if self.nt_h.board_log:
                        writer.add_scalar('Loss/train', sum_train_loss / self.nt_h.log_train_period,
                                          epoch * len(train_loader) + i*self.t_h.batch_size)

                        # Log one reconstructed training image sample
                        img_grid = vutils.make_grid([inputs[-1], labels[-1], outputs[-1]])
                        writer.add_image('reconstructed_training_images', img_grid,
                                         global_step=epoch * len(train_loader) + i*self.t_h.batch_size)

                    log_duration = time.time() - log_starttime
                    log_prop = log_duration / train_duration

                    data_loading_prop = data_loading_time / (train_duration - data_loading_time)

                    data_loading_time = 0
                    train_period_stime_train = time.time()

                    if self.nt_h.print_log:
                        print(
                            f'Epoch [{epoch + 1}/{self.nt_h.num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                            f'Train Loss: {sum_train_loss / self.nt_h.log_train_period:.4f}',
                            f', Logging Time Proportion: {log_prop:.4f}, '
                            f'Data Loading Time Proportion: {data_loading_prop:.4f}')

                    sum_train_loss = 0.0

                # Log val loss every log_val_freq batches
                if (i+1)%int(num_t_batches/self.nt_h.log_val_freq) == 0:

                    train_duration = time.time() - train_period_stime_val

                    val_starttime = time.time()
                    # calculate validation loss
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            inputs = batch['image'].to(self.device)
                            labels = batch['label'].to(self.device)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()

                    val_loss /= len(val_loader)

                    # log validation loss
                    if self.nt_h.board_log:
                        writer.add_scalar('Loss/val', val_loss, epoch * len(train_loader) + i*self.t_h.batch_size)

                        # Log one reconstructed validation image sample
                        # todo: log mutliple images instead, explicitly log images that perform very bad and ones that
                        #  perfoirm very good?
                        batch = next(iter(train_loader))
                        inputs = batch['image'].to(self.device)
                        labels = batch['label'].to(self.device)
                        outputs = model(inputs)

                        img_grid = vutils.make_grid([inputs[-1], labels[-1], outputs[-1]])
                        writer.add_image('reconstructed_validation_images', img_grid,
                                     global_step=epoch * len(train_loader) + i*self.t_h.batch_size)

                    # ploting images like this (they are also plotted on tensorboard, but here they are more
                    # uebersichtlich somehow)
                    # todo: outcommented for now, decide whether to delete or not
                    # print("Example Rectonsturtion from Val Set: ")
                    # plt.figure()
                    # show_datatripple(inputs[-1].cpu().detach().numpy(), labels[-1].cpu().detach().numpy(),
                    #                 outputs[-1].cpu().detach().numpy())
                    # plt.show()

                    model.train()

                    val_duration = time.time() - val_starttime
                    val_t_prop = val_duration / train_duration

                    if self.nt_h.print_log:
                        print('#######################################################################################')
                        print(
                            f'Epoch [{epoch + 1}/{self.nt_h.num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                            f'Val Loss: {val_loss:.4f}',
                            f'Val Log Time Proportion: {val_t_prop:.4f}')
                        print('#######################################################################################')

                    # save model and loss for ray
                    checkpoint_data = {
                        "epoch": epoch,
                        "net_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }
                    with tempfile.TemporaryDirectory() as checkpoint_dir:
                        data_path = Path(checkpoint_dir) / "data_preprocessing.pkl"
                        with open(data_path, "wb") as fp:
                            pickle.dump(checkpoint_data, fp)

                        checkpoint = Checkpoint.from_directory(checkpoint_dir)
                        train.report(
                            {"loss": val_loss},
                            checkpoint=checkpoint,
                        )

                    train_period_stime_val = time.time()
