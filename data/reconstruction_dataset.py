import os
import warnings

import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import psutil

# from reconstruction_dataset import *


"""
Helper functions for displaying data
"""


def show_datapair(image, label):
    """Show tactile points with object shape"""
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.subplots_adjust(wspace=0)
    axs[0].imshow(image[0])
    axs[0].set_axis_off()
    axs[1].imshow(label[0])
    axs[1].set_axis_off()


def show_datatripple(input, label, output):
    """Show tactile points with object shape"""
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.subplots_adjust(wspace=0)
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
        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MemoryReconstructionDataset(Dataset):
    """Reconstruction dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.images, self.labels = self.create_memory_dataset(csv_file, root_dir)

    def create_memory_dataset(self, csv_file, root_dir):
        
        df = pd.read_csv(csv_file)
        img_paths = df.iloc[:, 0].to_list()
        label_paths = df.iloc[:, 1].to_list()

        imgs = []
        labels = []

        str = ''
        label = None
        for i in tqdm(range(len(img_paths)), 'Data to memory'):
            if(str != label_paths[i]):
                str = label_paths[i]
                label_path = os.path.join(root_dir, str)
                label = np.load(label_path)
            img_path = os.path.join(root_dir, img_paths[i])

            if i % 50000 == 0:
                memory_usage = psutil.virtual_memory()
                print(f"Memory Usage: {memory_usage.percent}%")
                
            img = np.load(img_path)
            imgs.append(img)
            labels.append(label)
        
        return imgs, labels
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        sample = {'image': self.images[idx], 'label': self.labels[idx]}
        if self.transform:
            sample = self.transform(sample)

        return sample


"""
Transformations:
    ToTensor (mandatory): Converts numpy images to torch images
    RandomOrientation (optional): Randomly orientates both image and label.
        Data augmentation technique.
"""


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
