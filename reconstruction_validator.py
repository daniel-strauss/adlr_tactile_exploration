import torch
from torch.utils.data import DataLoader, Subset
from neural_nets.utility_functions import load_data
from neural_nets.rec_net import RecNet

def main():
    train_set, eval_set, test_set = load_data()
    smoke_set = Subset(eval_set, range(12))
    rec_net = RecNet()
    loss = rec_net.infer_dataset(smoke_set)
    print(loss.shape)
    return True

if __name__ == '__main__':
    main()