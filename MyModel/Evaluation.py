from matplotlib import pyplot as plt
import numpy as np

data_train = ['trainMyModel.txt', 'trainMyModelDepth3.txt', 'trainMyModelHeads.txt', 'trainMyModelLR.txt']
data_valid = ['validationMyModel.txt', 'validationMyModelDepth3.txt', 'validationMyModelHeads.txt', 'validationMyModelLR.txt']

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Model Training and Validation Comparison', fontsize=16)

for i in range(len(data_train)):
    tempTrain = []
    with open(data_train[i], 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            tempTrain.append(np.array(row))
    dataTrainNp = np.array(tempTrain)

    tempValid = []
    with open(data_valid[i], 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split()]
            tempValid.append(np.array(row))
    dataValidNp = np.array(tempValid)

    labels = [
        'Depth=4', 'Depth=3', 'Heads=2', 'LR=1e-4'
    ]

    ax[0].plot(dataTrainNp[:, 0], dataTrainNp[:, 1], label=f'Train Acc ({labels[i]})')
    ax[0].plot(dataValidNp[:, 0], dataValidNp[:, 1], label=f'Valid Acc ({labels[i]})')

    ax[1].plot(dataTrainNp[:, 0], dataTrainNp[:, 2], label=f'Train Loss ({labels[i]})')
    ax[1].plot(dataValidNp[:, 0], dataValidNp[:, 2], label=f'Valid Loss ({labels[i]})')

ax[0].set_title('Accuracy over Epochs', fontsize=14)
ax[0].set_ylabel('Accuracy')
ax[0].grid(True)
ax[0].legend(loc='lower right', fontsize=9)

ax[1].set_title('Loss over Epochs', fontsize=14)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].grid(True)
ax[1].legend(loc='upper right', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
plt.close()
