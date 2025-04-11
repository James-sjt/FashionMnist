from Model import MODEL
import torch
from DataLoader import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dim, depth, heads, dim_head, mlp_dim, batch_size = 128, 4, 4, 32, 256, 2

Model = MODEL(dim, depth, heads, dim_head, mlp_dim).to(device)

print('Total parameters: ', sum(p.numel() for p in Model.parameters() if p.requires_grad))

Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(Model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

data_train = DataLoader(FashionMNIST('train', device), batch_size=batch_size, shuffle=True)
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

