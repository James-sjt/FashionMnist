"""
Fine-tuning VGG16.

Usage:
  Train.py [--batch_size=<int>] [--lr=<float>] [--pre_train=<bool>]
  Train.py (-h | --help)

Options:
  -h --help                 Show this screen.
  --batch_size=<int>       Batch size [default: 2]
  --lr=<float>             Learning rate [default: 1e-3]
  --pre_train=<bool>        Use pre-trained weights [default: True]
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from DataLoader import FashionMNIST
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from docopt import docopt


class CONFIG:
    def __init__(self, batch_size, lr, pre_train):
        self.batch_size = batch_size
        self.lr = lr
        self.pre_train = pre_train

def main(model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Model = models.vgg16(weights=model_config.pre_train).to(device)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    Model.classifier[6] = nn.Linear(4096, 10, device=device)

    for name, p in Model.named_parameters():  # fine-tuning the last ffl of the classifier.
        if name != 'classifier.6.weight':
            p.requires_grad = False

    print('Total parameters which require trace gradient: ', sum(p.numel() for p in Model.parameters() if p.requires_grad))

    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(Model.parameters(), lr=model_config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    data_train = DataLoader(FashionMNIST('train', device), batch_size=model_config.batch_size, shuffle=True)
    data_test = DataLoader(FashionMNIST('test', device), batch_size=1, shuffle=False)

    best_acc = 0.

    for epoch in range(100):
        pre_train, true_train, total_loss = [], [], 0.
        for batch in tqdm(data_train):
            input_data = torch.concatenate([preprocess(batch[0]), preprocess(batch[0]), preprocess(batch[0])], dim=1)
            outputs = Model(input_data)

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
            input_data = torch.concatenate([preprocess(batch[0]), preprocess(batch[0]), preprocess(batch[0])], dim=1)
            outputs = Model(input_data)
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
            torch.save(Model.state_dict(), 'VGG16_' + str(epoch + 1) + '.pth')
            best_acc = report['accuracy']
            print(f'Best Accuracy: {best_acc:.4f}, parameters saved!')

        print('*' * 50)

        for name, p in Model.named_parameters():  # fine-tuning the last ffl of the classifier.
            if name != 'classifier.6.weight':
                p.requires_grad = False

if __name__ == '__main__':
    arguments = docopt(__doc__)

    model_config = CONFIG(
        batch_size = int(arguments["--batch_size"]),
        lr = float(arguments["--lr"]),
        pre_train = bool(arguments["--pre_train"]),
    )

    main(model_config)
