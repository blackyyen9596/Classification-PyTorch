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
from cv2 import cv2

from torch.utils.data import DataLoader
from utils.dataloader_bg import DataLoaderX

path_train = r'D:\training_dataset\simpsons\train'
path_val = r'D:\training_dataset\simpsons\val'

# path_train = r'/home/blackyyen/dataset/simpsons/train'
# path_val = r'/home/blackyyen/dataset/simpsons/val'
# data_dir = r'/simpsons'

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
train_batch_size = 150
val_batch_size = 100
# learning rate
lr = 1e-3

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# choose the training and test datasets
train_data = datasets.ImageFolder(path_train, transform=train_transforms)
valid_data = datasets.ImageFolder(path_val, transform=valid_transforms)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(train_data,
                          batch_size=train_batch_size,
                          num_workers=num_workers,
                          shuffle=True,
                          pin_memory=True)
valid_loader = DataLoader(valid_data,
                          batch_size=val_batch_size,
                          num_workers=num_workers,
                          shuffle=False,
                          pin_memory=True)

train_data, train_target, val_data, val_target = [], [], [], []
for data, target in tqdm(train_loader):
    train_data.append(data.unsqueeze(0).half())
    train_target.append(target)
# for data, target in tqdm(valid_loader):
#     val_data.append(data)
#     val_target.append(target)

# 讀取模型
device = torch.device("cuda")
model = torch.load('./classification/resnext101_32x8d.pth')
# 查看可用的模型
# print(torch.hub.list('pytorch/vision'))
# print(torch.hub.list('facebookresearch/WSL-Images))
# 將模型儲存起來
# model = torch.hub.load('facebookresearch/WSL-Images',
#                        'resnext101_32x8d_wsl',
#                        force_reload=False,
#                        verbose=0)
# torch.save(model, './classification/resnext101_32x8d.pth')
# 輸出模型可用函式
# print(dir(model))
# print(model.layer1[0].conv1._parameters)

# 提取參數fc的輸入參數
fc_features = model.fc.in_features
# 將最後輸出類別改為20
model.fc = nn.Linear(fc_features, 20)
for name, param in model.named_parameters():
    if name == 'layer4.0.conv1.weight':
        break
    param.requires_grad = False
# 將所有層設定為不可訓練
# for param in model.parameters():
#     param.requires_grad = False
# 將最後一層fc設定為可以訓練
# for param in model.fc.parameters():
#     param.requires_grad = True
# 輸出模型參數
# from torchsummary import summary
# summary(model.to(device), (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                           T_max=100,
                                           eta_min=1e-4,
                                           last_epoch=-1)
#%%
model.to(device)
# number of epochs to train the model
n_epochs = 50

valid_loss_min = np.Inf  # track change in validation loss

train_losses, valid_losses = [], []

for epoch in range(1, n_epochs + 1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print('running epoch: {}'.format(epoch))
    # 訓練模型
    model.train()
    for train_number in tqdm(range(len(train_data))):
        # move tensors to GPU if CUDA is available
        train_data_batch = train_data[train_number].float()
        train_target_batch = train_target[train_number]
        train_data_batch, train_target_batch = train_data_batch.to(
            device), train_target_batch.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        train_data_reshape = train_data_batch.reshape(
            train_data_batch.shape[1], train_data_batch.shape[2],
            train_data_batch.shape[3], train_data_batch.shape[4])
        output = model(train_data_reshape)
        # calculate the batch loss
        loss = criterion(output, train_target_batch)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        scheduler.step()
        # update training loss
        train_loss += loss.item() * data.size(0)
    # 驗證模型
    model.eval()
    with torch.no_grad():
        accuracy = 0.0
        for data, target in tqdm(valid_loader):
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)
            # accuracy
            predict_y = torch.max(output, dim=1)[1]
            accuracy = accuracy + (predict_y == target.to(device)).sum().item()

    accuracy = accuracy / len(valid_loader.dataset)

    # calculate average losses
    train_losses.append(train_loss / len(train_loader.dataset))
    valid_losses.append(valid_loss / len(valid_loader.dataset))
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)

    # print training/validation statistics
    print('\ttrain Loss: {:.6f} \tval Loss: {:.6f}'.format(
        train_loss, valid_loss))
    print('\taccuracy: {:.6f}'.format(accuracy))
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('valid loss decreased ({:.6f} --> {:.6f}).  saving model ...'.
              format(valid_loss_min, valid_loss))
        torch.save(model.state_dict(), './classification/logs/model.pth')
        valid_loss_min = valid_loss