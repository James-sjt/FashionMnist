import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Transformer

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class CLASSIFIER(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class MODEL(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.cnn = CNN()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.classifier = CLASSIFIER()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        cnn_x = self.cnn(x)  # b, 128, 14, 14
        temp_x = cnn_x.view(cnn_x.size(0), cnn_x.size(1), -1)
        flatten_x = temp_x.permute(0, 2, 1)  # b, 196, 128
        cls_x = torch.cat((self.cls_token.expand(flatten_x.shape[0], -1, -1), flatten_x), 1)  # b, 197, 128
        trans_x = self.transformer(cls_x)
        out_x = self.classifier(trans_x[:, 0])
        return out_x
