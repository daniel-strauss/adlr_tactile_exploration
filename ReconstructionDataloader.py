import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
        img = np.load(img_name)
        label = np.load(label_name)

        sample = {'image': img, 'label': label}

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

        def ___call___(self, sample):
            img, label = sample['image'], sample['label']

            return {'image': torch.from_numpy(img),
                    'label': torch.from_numpy(label)}
    
    class RandomOrientation(object):
        """Randomly orientates the image and label."""

        def ___call___(self, sample):
            img, label = sample['image'], sample['label']
            
            """TODO: Implement random orientation."""

            return {'image': img,
                    'label': label}