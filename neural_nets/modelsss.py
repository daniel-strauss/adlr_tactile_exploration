import io
import os
from neural_nets.models.unet import UNet3
import torch
import torch.nn as nn
import pickle

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class RecNet():

    def __init__(self,
                 base_model=UNet3,
                 trial_path='./neural_nets/best_trial.pkl',
                 states_path='./neural_nets/trained_rec.pkl',
                 cuda=True):
        
        with open(trial_path, 'rb') as pickle_file:
            best_trial = CPU_Unpickler(pickle_file).load()
        pickle_file.close()

        with open(states_path, 'rb') as pickle_file:
            states = CPU_Unpickler(pickle_file).load()
        pickle_file.close()

        config = best_trial.config
        self.rec_net = UNet3(config)
        self.rec_net.load_state_dict(states['net_state_dict'])

        if not cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rec_net.to(self.device)
        self.rec_net.eval()
    
    def infer()
        
    