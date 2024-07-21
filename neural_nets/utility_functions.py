import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import torch.utils
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import transforms
from data.reconstruction_dataset import *
from neural_nets.models.unet import UNet3
from ray import train
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from pathlib import Path

standard_config: dict = {
    'batch_size': 256,
    'epochs': 10,
    'num_workers': 8,
    'smoke': False,

    'lr': 1e-5,
    'weight_decay': 0,
    'optimizer': optim.Adam,
    'loss_func': nn.BCELoss(),

    'model': UNet3,
    'depth': 5,
    'channels': 64
}


def load_data(dir='./datasets/2D_shapes', transform=ToTensor()):
    test_path = os.path.join(dir, 'test.csv')
    eval_path = os.path.join(dir, 'eval.csv')
    train_path = os.path.join(dir, 'train.csv')

    test_set = ReconstructionDataset(test_path, dir, ToTensor())
    eval_set = ReconstructionDataset(eval_path, dir, transform)
    train_set = ReconstructionDataset(train_path, dir, transform)

    return train_set, eval_set, test_set


def load_rl_data(dir='./datasets/2D_shapes', transform=ToTensor()):
    test_path = os.path.join(dir, 'test.csv')
    eval_path = os.path.join(dir, 'eval.csv')
    train_path = os.path.join(dir, 'train.csv')

    test_set = ReinforcementDataset(test_path, dir, transform)
    eval_set = ReinforcementDataset(eval_path, dir, transform)
    train_set = ReinforcementDataset(train_path, dir, transform)

    return train_set, eval_set, test_set


def train_reconstruction(config, train_set: Dataset, eval_set: Dataset):
    model = config['model'](config)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = config['loss_func']
    optimizer = config['optimizer'](model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

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

    if config['smoke']:
        train_set = Subset(train_set, range(min(500, len(train_set))))
        eval_set = Subset(eval_set, range(min(100, len(train_set))))

    torch.manual_seed(0)
    trainloader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']
    )
    valloader = DataLoader(
        eval_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']
    )

    for epoch in range(start_epoch, config['epochs']):  # loop over the dataset multiple times
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
        for i, batch in enumerate(valloader, 0):
            with torch.no_grad():
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(inputs)
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
                {"loss": val_loss / val_steps, "epoch": epoch},
                checkpoint=checkpoint,
            )

    print("Finished Training")
