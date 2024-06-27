import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Callable
from torchvision import transforms
from data.reconstruction_dataset import *
from neural_nets.weight_inits import weight_init_kx
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path

def load_data(dir='./datasets/2D_shapes'):
    transform = transforms.Compose([
        ToTensor()
    ])

    test_path = os.path.join(dir, 'test.csv')
    train_path = os.path.join(dir, 'annotations.csv')
    
    test_set = ReconstructionDataset(test_path, dir, ToTensor())
    train_set = ReconstructionDataset(train_path, dir, transform)

    return train_set, test_set


class Hparams():
    # class, that specifies the key, value type pairs for Hparams

    # train hparams
    batch_size: int = 16
    epochs: int = 10
    train_prop: float = 0.8
    num_workers: int = 8

    # logging
    # amount of val calculations per epoch
    log_val_freq: int = 1
    # setting period to -1 logs every epoch end
    log_train_period: int = 100
    print_log: bool = True
    board_log: bool = True  # log to tensorboard


    # optimization hparams
    lr: float = 1e-5
    weight_decay: float = 0
    optimizer: optim.Optimizer = optim.Adam
    loss_func: Callable = nn.BCELoss()

    # network hparams
    model: nn.Module
    weight_init: Callable = weight_init_kx
    depth: int = 5
    channels: int = 64

    def __init__(self, params: dict = None):
        if not params is None:
            self.override_from_dict(params)

    def override_from_dict(self, params: dict):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Trainer:
    
    h: Hparams
    dataset: Dataset

    def __init__(self, hparams: Hparams, dataset: Dataset):

        # non tunable hyperparameters
        self.h = hparams
        self.dataset = dataset
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        print(f'Training on {self.device}.')

    def train(self, config=None):
        if config is not None:
            self.h.override_from_dict(config)

        device = self.device
        h = self.h

        model = h.model(h)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        criterion = h.loss_func
        optimizer = h.optimizer(model.parameters(), lr=h.lr, weight_decay=h.weight_decay)

        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    checkpoint_state = pickle.load(fp)
                start_epoch = checkpoint_state["epoch"]
                model.load_state_dict(checkpoint_state["net_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        else:
            start_epoch = 0

        torch.manual_seed(0)
        train_size = int(len(self.dataset) * h.train_prop)
        train_subset, val_subset = random_split(
            self.dataset, [train_size, len(self.dataset) - train_size]
        )

        trainloader = DataLoader(
            train_subset, batch_size=h.batch_size, shuffle=True, num_workers=h.num_workers
        )
        valloader = DataLoader(
            val_subset, batch_size=h.batch_size, shuffle=True, num_workers=h.num_workers
        )

        for epoch in range(start_epoch, h.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            for i, batch in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f"
                        % (epoch + 1, i + 1, running_loss / epoch_steps)
                    )
                    running_loss = 0.0

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, batch in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs = batch['image'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)

                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": val_loss / val_steps, "accuracy": correct / total},
                    checkpoint=checkpoint,
                )
            
            tune.report(loss=(val_loss / val_steps), accuracy=(correct / total))

        print("Finished Training")