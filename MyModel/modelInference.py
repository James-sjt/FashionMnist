import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from DataLoader import FashionMNIST
from Model import MODEL

# Configuration
class CONFIG:
    def __init__(self):
        self.dim = 128
        self.depth = 4
        self.heads = 4
        self.dim_head = 32
        self.mlp_dim = 256
        self.pos_emb = False
        self.dis = True
        self.dft = 'True'
        self.batch_size = 1

# FashionMNIST class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Plot ROC curves for each class
def plot_roc(y_true, y_probs, num_classes):
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve by Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    config = CONFIG()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = MODEL(config.dim, config.depth, config.heads, config.dim_head, config.mlp_dim, config.pos_emb).to(device)
    model.load_state_dict(torch.load('Model_dis_dft.pth', map_location=device))  # Your trained model
    model.eval()

    data_test = DataLoader(FashionMNIST('test', config.dft, device), batch_size=1, shuffle=False)

    y_true, y_pred = [], []
    y_probs = []

    with torch.no_grad():
        for batch in tqdm(data_test):
            logits, _ = model(batch[0])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels = torch.argmax(batch[1], dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels)
            y_probs.extend(probs)

    y_probs = np.array(y_probs)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, "MyModel")

    # Classification Report
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ROC Curve
    plot_roc(y_true, y_probs, num_classes=len(class_names))

if __name__ == "__main__":
    main()
