'''
Class that provides:
    - instancing neural network, optimizers, dataloaders,
        everything else from hyperparameters in dict config
    - training function
    - progress logging, using tensoarboard, automatically opens tap for logging
    - usage of ray, ray is a library that searches for good hparams

    The purpose of this class is to avoid multiple

'''
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import torch.cuda

def define_datasets(hparams):
    pass

def define_net(hparams):
    pass

class Trainer:

    # dictionary containing all non-tunable hyperparameter, e.g. all hyperparameters,
    # that shall not be searched in a hypothetical parameter search
    nt_hparams: dict

    def __init__(self, nt_hparams: dict):
        self.nt_hparams = nt_hparams

    def train(self, t_hparams:dict):
        # t_hparams: dictionary containing all tunable hparams, e.g.
        # all parameters that should be searched in a hypothetical parameter search

        # merging the two hparam sets into one hparam set, containing all hparams
        if not set(t_hparams.keys()).isdisjoint(self.nt_hparams.keys()):
            warnings.warn("t_hparams and nt_hparams have overlapping keys. For duplicate keys the key in "
                          "nt_hparams will be discarded. (But rather try to avoid this warning)")

        hparams = t_hparams | self.nt_hparams


        define_datasets(hparams)
        define_net(hparams)
