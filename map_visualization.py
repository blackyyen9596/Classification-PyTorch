import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

path_test = r'D:\training_dataset\simpsons\test'

# 讀取模型
device = torch.device("cuda")
model = torch.hub.load('facebookresearch/WSL-Images',
                       'resnext101_32x8d_wsl',
                       force_reload=True)
model.to(device)

# for name, para in model.named_parameters():
#     print('{}: {}'.format(name, para.shape))

# 輸出模型可用函式
# print(dir(model))

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def viz(module, input):
    plt.figure(figsize=(10, 10))
    x = input[0][0]
    min_num = np.minimum(256, x.size()[0])
    for i in range(min_num):
        plt.subplot(8, 8, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x[i].cpu())
    plt.show()


for name, m in model.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        m.register_forward_pre_hook(viz)
img = cv2.imread(
    r'D:\training_dataset\simpsons\test\abraham_grampa_simpson\pic_0008.jpg')
img = test_transforms(img).unsqueeze(0).to(device)
with torch.no_grad():
    model(img)