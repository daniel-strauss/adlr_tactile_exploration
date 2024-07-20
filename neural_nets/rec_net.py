import io
import os
from neural_nets.models.unet import UNet3
from skimage.morphology import convex_hull_image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
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
                 loss_func=nn.BCELoss(),
                 trial_path='./neural_nets/best_trial.pkl',
                 states_path='./neural_nets/trained_rec.pkl',
                 dummy=False,
                 cuda=True):

        self.loss_func = loss_func

        if dummy:
            self.rec_net=DummyRecNet()
        else:
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

    def infer(self, input, label):
        with torch.no_grad():
            input = torch.from_numpy(np.expand_dims(input, 0)).to(self.device)
            label = torch.from_numpy(np.expand_dims(label, 0)).to(self.device)
            reconstruction = self.rec_net(input)
            loss = self.loss_func(reconstruction, label)

            rec = (reconstruction >= 0.5)
            lab = (label > 0)
            n = torch.logical_or(rec, lab).float().sum()
            metric = torch.logical_and(rec, lab).float().sum() * 100 / n
        return loss.item(), metric.item(), reconstruction[0].cpu().detach().numpy()
    
    def infer_dataset(self, dataset):
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
        total_loss = 0.0
        steps = 0
        for batch in loader:
            with torch.no_grad():
                inputs = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.rec_net(inputs)
                loss = self.loss_func(outputs, labels)
                total_loss += loss.cpu().numpy()
                steps += 1
        return total_loss / steps
        
class DummyRecNet(torch.nn.Module):
    def forward(self, x):
        # x is assumed to be a tensor of shape (batch_size, channels, res, res)
        # Convert x to numpy array and reshape if necessary
        x_np = x.squeeze().cpu().numpy()  # Convert to numpy array and squeeze out batch and channel dimensions
        hull = convex_hull_image(x_np)
        # Example: Return convex hull vertices as a tensor (you may need to adjust this based on your requirement)
        # Return convex hull vertices as a tensor with singleton dimension
        hull = torch.tensor(hull, dtype=torch.float).reshape((1, 1, 256, 256))
        return hull