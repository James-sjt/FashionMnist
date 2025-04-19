"""
Train this model.

Usage:
  Train.py [--dim=<int>] [--depth=<int>] [--heads=<int>] [--dim_head=<int>]
           [--mlp_dim=<int>] [--batch_size=<int>] [--lr=<float>] [--pre_train=<bool>]
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

class CONFIG:
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, batch_size, lr, pre_train):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_heads
        self.mlp_dim = mlp_dim
        self.batch_size = batch_size
        self.lr = lr
        self.pre_train = pre_train


def main(model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = MODEL(model_config.dim,
                  model_config.depth,
                  model_config.heads,
                  model_config.dim_head,
                  model_config.mlp_dim).to(device)

    if model_config.pre_train:
        Model.load_state_dict(torch.load('Model_97.pth', map_location=device))

    print('Total parameters of MyModel: ', sum(p.numel() for p in Model.parameters() if p.requires_grad))

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(Model.parameters(), lr=model_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    data_train = DataLoader(FashionMNIST('train', device), batch_size=model_config.batch_size, shuffle=True)
    data_test = DataLoader(FashionMNIST('test', device), batch_size=1, shuffle=False)

    best_acc = 0.

    for epoch in range(100):
        pre_train, true_train, total_loss = [], [], 0.
        for batch in tqdm(data_train):
            outputs = Model(batch[0])

            loss = Loss(outputs, batch[1])
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

        Model.eval()

        pre_test, true_test, total_loss = [], [], 0.
        for batch in tqdm(data_test):
            outputs = Model(batch[0])
            loss = Loss(outputs, batch[1])
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
        pre_train = bool(arguments["--pre_train"]),
    )

    main(model_config)
