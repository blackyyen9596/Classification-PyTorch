import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim

# 顯示之層數
layer = 'conv1.weight'

# 讀取模型
device = torch.device("cuda")
model = torch.hub.load('facebookresearch/WSL-Images',
                       'resnext101_32x8d_wsl',
                       force_reload=True)
weights = dict()
# 顯示每一層之名稱與shape
for name, para in model.named_parameters():
    print('{}: {}'.format(name, para.shape))
    if name == layer:
        weights[layer] = para
print(weights[layer].size())

plt.figure(figsize=(10, 10))
for idx, filt in enumerate(weights[layer].detach().numpy()):
    # print(filt[:, :])
    # print(idx)
    plt.subplot(8, 8, idx + 1)
    plt.imshow(filt[1, :, :])
    plt.axis('off')
plt.show()

# input()

# 輸出模型可用函式
# print(dir(model))
# print(model._modules.size())
# print(model.layer1[0].conv1._parameters)