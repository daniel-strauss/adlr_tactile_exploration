import os
import warnings

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import psutil

# from reconstruction_dataset import *


"""
Helper functions for displaying data_preprocessing
"""


def show_datapair(image, label):
    """Show tactile points with object shape"""
    fig, axs = plt.subplots(1, 2, fig_size=(12, 6))
    axs[0].imshow(image[0])
    axs[0].set_axis_off()
    axs[1].imshow(label[0])
    axs[1].set_axis_off()


def show_datatripple(input, label, output):
    """Show tactile points with object shape"""
    fig, axs = plt.subplots(1, 3, fig_size=(18, 6))
    axs[0].imshow(input[0])
    axs[0].set_axis_off()
    axs[1].imshow(label[0])
    axs[1].set_axis_off()
    axs[2].imshow(output[0])
    axs[2].set_axis_off()


def show_datapair_batch(sample_batch):
    """Show batch of tactile points with object shape"""
    image_batch, label_batch = sample_batch['image'], sample_batch['label']
    batch_size = len(image_batch)

    size = image_batch[0].size()
    print(size)
    plt.figure(figsize=(size[0] / 100, batch_size * (size[1] / 100 + 10) - 10))
    fig, axs = plt.subplots(2, batch_size, sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.tight_layout()

    for i in range(batch_size):
        axs[0, i].imshow(image_batch[i][0])
        axs[0, i].set_axis_off()
        axs[1, i].imshow(label_batch[i][0])
        axs[1, i].set_axis_off()

    plt.show()


class ReconstructionDataset(Dataset):
    """Reconstruction dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.annotation_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.annotation_frame.iloc[idx, 0])
        label_name = os.path.join(self.root_dir,
                                  self.annotation_frame.iloc[idx, 1])

        img = np.load(img_name).astype('f')
        label = np.load(label_name).astype('f')
        outline = np.load(os.path.join(os.path.dirname(label_name), 'outline.npy'))
        sample = {'image': img, 'label': label, 'outline': outline}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ReinforcementDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        annotation_frame = pd.read_csv(csv_file)
        self.labels = annotation_frame.iloc[:,1].drop_duplicates()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label_path = os.path.join(self.root_dir,
                                  self.labels.iloc[idx])
        
        outline_path = os.path.join(os.path.dirname(label_path), 'outline.npy')

        label = np.load(label_path).astype('f')
        outline = np.load(outline_path)
        sample = {'label': label, 'outline': outline}

        if self.transform:
            sample = self.transform(sample)

        return sample




class ToTensor(object):
    """Convert numpy images in sample to Tensors."""

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        return {'image': torch.from_numpy(img),
                'label': torch.from_numpy(label)}
    

class RandomFlip(object):
    """Randomly flips the image and label along the vertical axis"""

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        k = np.random.randint(2)
        
        if k:
            img = np.flip(img, axis=-1)
            label = np.flip(label, axis=-1)
        
        return {'image': img,
                'label': label}


class RandomOrientation(object):
    """Randomly orientates the image and label by a multitude of 90 degrees."""

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        k = np.random.randint(4)

        return {'image': np.rot90(img, k, axes=(1, 2)),
                'label': np.rot90(label, k, axes=(1, 2))}
