"""
Train this model.

Usage:
  Train.py [--dim=<int>] [--depth=<int>] [--heads=<int>] [--dim_head=<int>]
           [--mlp_dim=<int>] [--batch_size=<int>] [--lr=<float>] [--pre_train=<bool>] [--pos_emb=<bool>] [--dis=<bool>] [--dft=<bool>]
  Train.py (-h | --help)

Options:
  -h --help                 Show this screen.
  --dim=<int>              Input feature dimension of Transformer [default: 128]
  --depth=<int>            Number of Transformer blocks [default: 4]
  --heads=<int>            Number of heads in Transformer [default: 4]
  --dim_head=<int>         Dimension of each head [default: 32]
  --mlp_dim=<int>          MLP layer dimension [default: 256]
  --batch_size=<int>       Batch size [default: 2]
  --lr=<float>             Learning rate [default: 1e-3]
  --pre_train=<bool>        Use pre-trained weights [default: False]
  --pos_emb=<bool>         Use position embedding [default: False]
  --dis=<bool>             Use self-distillation [default: False]
  --dft=<bool>             Use high-pass filter to enhance images [default: False]
"""

from Model import MODEL
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from DataLoader import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm
from docopt import docopt
from SelfDistillation import SelfDistillation
import numpy as np


class CONFIG:
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, batch_size, lr, pre_train, pos_emb, dis, dft):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_heads
        self.mlp_dim = mlp_dim
        self.batch_size = batch_size
        self.lr = lr
        self.pre_train = pre_train
        self.pos_emb = pos_emb
        self.dis = dis
        self.dft = dft

def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False


def main(model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = MODEL(model_config.dim,
                  model_config.depth,
                  model_config.heads,
                  model_config.dim_head,
                  model_config.mlp_dim,
                  model_config.pos_emb).to(device)

    if model_config.pre_train:
        if model_config.dis and model_config.dft:
            Model.load_state_dict(torch.load('Model_dis_dft.pth', map_location=device))
        elif model_config.dis and not model_config.dft:
            Model.load_state_dict(torch.load('Model_dis.pth', map_location=device))
        elif not model_config.dis and not model_config.dft:
            Model.load_state_dict(torch.load('Model_simple.pth', map_location=device))
        else:
            raise ValueError('No pre-trained weights, set pre_train to False!')


    print('Total parameters of MyModel: ', sum(p.numel() for p in Model.parameters() if p.requires_grad))

    if not model_config.dis:
        Loss = nn.CrossEntropyLoss()
    else:
        Loss = SelfDistillation(model_config.dim, model_config.depth, 10, T=1, alpha=0.5, beta=0.5).to(device)

    optimizer = torch.optim.AdamW(Model.parameters(), lr=model_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    data_train = DataLoader(FashionMNIST('train', model_config.dft, device), batch_size=model_config.batch_size, shuffle=True)
    data_test = DataLoader(FashionMNIST('test', model_config.dft, device), batch_size=1, shuffle=False)

    best_acc = 0.

    for epoch in range(100):
        pre_train, true_train, total_loss = [], [], 0.
        for batch in tqdm(data_train):
            outputs, steps = Model(batch[0])
            if not model_config.dis:
                loss = Loss(outputs, batch[1])
            else:
                loss = Loss(outputs, steps, batch[1])

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for i in range(outputs.shape[0]):
                pre_train.append(outputs.cpu().detach().numpy().argmax(axis=1)[i])
                true_train.append(batch[1].cpu().detach().numpy().argmax(axis=1)[i])

        print('*' * 50)
        print(f'EPOCH: {epoch + 1}, result on training set:')
        print(classification_report(true_train, pre_train))
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f'Mean of Loss: {total_loss / 60000.:.4f}, current lr: {current_lr:.4f}')
        print('*' * 50)

        report = classification_report(true_train, pre_train, output_dict=True)
        resultofTrain = np.array([epoch + 1, report['accuracy'], total_loss / 60000.])
        with open('trainMyModelDepth3.txt', 'a') as f:
            np.savetxt(f, resultofTrain.reshape(1, -1), fmt='%.4f')

        Model.eval()

        pre_test, true_test, total_loss = [], [], 0.
        for batch in tqdm(data_test):
            outputs, steps = Model(batch[0])
            if not model_config.dis:
                loss = Loss(outputs, batch[1])
            else:
                loss = Loss(outputs, steps, batch[1])

            total_loss += loss.item()
            for i in range(outputs.shape[0]):
                pre_test.append(outputs.cpu().detach().numpy().argmax(axis=1)[i])
                true_test.append(batch[1].cpu().detach().numpy().argmax(axis=1)[i])

        print('*' * 50)
        print(f'EPOCH: {epoch + 1}, result on testing set:')
        report = classification_report(true_test, pre_test, output_dict=True)
        print(classification_report(true_test, pre_test))
        print(f'Mean of Loss: {total_loss / 10000.:.4f}')

        if report['accuracy'] > best_acc:
            torch.save(Model.state_dict(), 'Model_' + str(epoch + 1) + '.pth')
            best_acc = report['accuracy']
            print(f'Best Accuracy: {best_acc:.4f}, parameters saved!')
        resultofVal = np.array([epoch + 1, report['accuracy'], total_loss / 10000.])
        with open('validationMyModelDepth3.txt', 'a') as f:
            np.savetxt(f, resultofVal.reshape(1, -1), fmt='%.4f')
        print('*' * 50)

        Model.train()

if __name__ == '__main__':
    arguments = docopt(__doc__)

    model_config = CONFIG(
        dim=int(arguments["--dim"]),
        depth = int(arguments["--depth"]),
        heads = int(arguments["--heads"]),
        dim_heads = int(arguments["--dim_head"]),
        mlp_dim = int(arguments["--mlp_dim"]),
        batch_size = int(arguments["--batch_size"]),
        lr = float(arguments["--lr"]),
        pre_train = str2bool(arguments["--pre_train"]),
        pos_emb = str2bool(arguments["--pos_emb"]),
        dis = str2bool(arguments["--dis"]),
        dft=str2bool(arguments["--dft"]),
    )

    main(model_config)
