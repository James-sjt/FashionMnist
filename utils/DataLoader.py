import mnist_reader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import numpy as np


class FashionMNIST(Dataset):
    def __init__(self, dtype, dft, device):
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

        self.dft = dft
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.device = device
        if not self.dft:
            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.data = torch.unsqueeze(self.data, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        temp = [0.] * 10
        temp[int(self.labels[index])] = 1.
        label = torch.tensor(temp, dtype=torch.float32)
        if self.dft:
            inhanceData = gaussian_high_pass_filter(self.data[index])
            inhanceData = torch.tensor(inhanceData, dtype=torch.float32)
            inhanceData = torch.unsqueeze(inhanceData, 0)
            return self.transform(inhanceData).to(self.device), label.to(self.device)
        else:
            return self.transform(self.data[index]).to(self.device), label.to(self.device)


def high_pass_filter(img, radius=4):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


def gaussian_high_pass_filter(img, sigma=5):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)
    H = 1 - np.exp(-(D**2) / (2 * (sigma**2)))
    fshift_filtered = fshift * H
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

