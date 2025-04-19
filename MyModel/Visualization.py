from Model import MODEL
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from DataLoader import FashionMNIST
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def transform_img(features, height=14, wide=14):
    b, l, d = features.shape

    if height * wide != l - 1:
        raise ValueError('height * wide != l - 1')

    features = features[:, 1:]
    temp = features.view(b, height, wide, d)
    temp = temp.transpose(2, 3).transpose(1, 2)
    return temp # b, c, h, w

device = 'cpu'

dim, depth, heads, dim_head, mlp_dim, batch_size = 128, 4, 4, 32, 256, 2

Model = MODEL(dim, depth, heads, dim_head, mlp_dim).to(device)

Model.load_state_dict(torch.load('Model_97.pth', map_location=device))

cam = GradCAM(model=Model, target_layers=[Model.transformer.layers[1][-1].fn.net[3]], reshape_transform=transform_img)

Model.eval()

data_test = DataLoader(FashionMNIST('test', device), batch_size=1, shuffle=False)
count = 0

for batch in tqdm(data_test):
    grayscale_cam = cam(input_tensor=batch[0], targets=None)[0]
    rgb_image = np.repeat(batch[0][0].numpy(), 3, axis=0)  # shape: (3, 28, 28)
    rgb_image = np.transpose(rgb_image, (1, 2, 0))  # to HWC
    rgb_image = rgb_image / rgb_image.max()  # normalize to [0, 1]

    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite('HeapImg/cam' + str(count) + '.jpg', visualization_bgr)
    count += 1