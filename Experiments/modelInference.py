import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from DataLoader import FashionMNIST

# Class names for FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocessing for input resizing and cropping
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

# Load test data
def get_test_loader(device):
    return DataLoader(FashionMNIST('test', 'False', device), batch_size=1, shuffle=False)

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

# Plot ROC curve for each class
def plot_roc_curve(y_true, y_probs, model_name):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Inference and show confusion matrix + ROC
def inference_and_show_confmat(model, model_name, device):
    model.eval()
    data_test = get_test_loader(device)
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(data_test, desc=f"Inferencing {model_name}"):
            input_data = torch.cat([preprocess(batch[0]), preprocess(batch[0]), preprocess(batch[0])], dim=1)
            input_data = input_data.to(device)
            outputs = model(input_data)
            prob = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = np.argmax(prob)
            true = torch.argmax(batch[1], dim=1).cpu().numpy()[0]

            y_pred.append(pred)
            y_true.append(true)
            y_probs.append(prob)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, model_name)

    # ROC Curve
    plot_roc_curve(y_true, np.array(y_probs), model_name)

# Main entry point
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model weights from Hugging Face
    url_mobile = "https://huggingface.co/James0323/FashionMnist/resolve/main/MobileNet_45.pth"
    state_dict_mobile = torch.hub.load_state_dict_from_url(url_mobile, map_location=device)

    url_res = "https://huggingface.co/James0323/FashionMnist/resolve/main/ResNet50_91.pth"
    state_dict_res = torch.hub.load_state_dict_from_url(url_res, map_location=device)

    url_vgg = "https://huggingface.co/James0323/FashionMnist/resolve/main/VGG16_73.pth"
    state_dict_vgg = torch.hub.load_state_dict_from_url(url_vgg, map_location=device)

    # MobileNet
    mobilenet = models.mobilenet_v2(pretrained=False).to(device)
    mobilenet.classifier[1] = nn.Linear(1280, 10, device=device)
    mobilenet.load_state_dict(state_dict_mobile)
    inference_and_show_confmat(mobilenet, 'MobileNet', device)

    # VGG16
    vgg16 = models.vgg16(weights=None).to(device)
    vgg16.classifier[6] = nn.Linear(4096, 10, device=device)
    vgg16.load_state_dict(state_dict_vgg)
    inference_and_show_confmat(vgg16, 'VGG16', device)

    # ResNet50
    resnet50 = models.resnet50(pretrained=False).to(device)
    resnet50.fc = nn.Linear(2048, 10, device=device)
    resnet50.load_state_dict(state_dict_res)
    inference_and_show_confmat(resnet50, 'ResNet50', device)

if __name__ == '__main__':
    main()
