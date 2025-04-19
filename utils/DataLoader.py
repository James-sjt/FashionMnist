import mnist_reader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn


class FashionMNIST(Dataset):
    def __init__(self, dtype, device):
        if dtype == 'train':
            self.data, self.labels = mnist_reader.load_mnist('../data/fashion', kind='train')
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(25, translate=(0.1, 0.1)),
            ])
        elif dtype == 'test':
            self.data, self.labels = mnist_reader.load_mnist('../data/fashion', kind='t10k')
            self.transform = transforms.Compose([
                nn.Identity()
            ])
        else:
            raise ValueError('dtype must be train or test!')

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data = torch.unsqueeze(self.data, 1)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        temp = [0.] * 10
        temp[int(self.labels[index])] = 1.
        label = torch.tensor(temp, dtype=torch.float32)
        return self.transform(self.data[index]).to(self.device), label.to(self.device)



