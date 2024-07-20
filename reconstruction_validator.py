import torch
from torch.utils.data import DataLoader, Subset
from neural_nets.utility_functions import load_data
from neural_nets.rec_net import RecNet

def main():
    train_set, eval_set, test_set = load_data()
    smoke_set = Subset(eval_set, range(12))
    rec_net = RecNet()

    train_loss = rec_net.infer_dataset(train_set)
    val_loss = rec_net.infer_dataset(eval_set)
    test_loss = rec_net.infer_dataset(test_set)

    print(f'Train loss: {train_loss}\nValidation loss: {val_loss}\nTest loss: {test_loss}')

if __name__ == '__main__':
    main()