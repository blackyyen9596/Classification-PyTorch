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
from tools.dataloader_bg import DataLoaderX

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

path_train = r'D:\training_dataset\simpsons\train'
path_val = r'D:\training_dataset\simpsons\val'

# path_train = r'/media/blackyyen/新增磁碟區1/Dataset/simpsons/train'
# path_val = r'/media/blackyyen/新增磁碟區1/Dataset/simpsons/val'
# data_dir = r'/simpsons'

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
train_batch_size = 20
val_batch_size = 20
# learning rate
lr = 1e-3

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# choose the training and test datasets
train_data = datasets.ImageFolder(path_train, transform=train_transforms)
valid_data = datasets.ImageFolder(path_val, transform=valid_transforms)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoaderX(train_data,
                           batch_size=train_batch_size,
                           num_workers=num_workers,
                           shuffle=True,
                           pin_memory=True)
val_loader = DataLoaderX(valid_data,
                         batch_size=val_batch_size,
                         num_workers=num_workers,
                         shuffle=False,
                         pin_memory=True)
# 讀取模型
device = torch.device("cuda")
model = torch.load('./classification/pretrained_model/resnext101_32x8d.pth')
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
# for name, param in model.named_parameters():
#     if name == 'layer4.0.conv1.weight':
#         break
#     param.requires_grad = False
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
                                           T_max=50,
                                           eta_min=1e-5,
                                           last_epoch=-1)

model.to(device)
# number of epochs to train the model
n_epochs = 50

val_loss_min = np.Inf  # track change in validation loss

train_losses, val_acc, val_losses, lr = [], [], [], []

for epoch in range(1, n_epochs + 1):
    # keep track of training and validation loss
    train_loss = 0.0
    val_loss = 0.0
    print('running epoch: {}'.format(epoch))
    # 訓練模型
    model.train()
    with tqdm(train_loader) as pbar:
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)
            pbar.update(1)
            pbar.set_description('train')
            pbar.set_postfix(
                **{
                    'loss': loss.item(),
                    'lr': optimizer.state_dict()['param_groups'][0]['lr']
                })
    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
    scheduler.step()
    # 驗證模型
    model.eval()
    with tqdm(val_loader) as pbar:
        with torch.no_grad():
            accuracy = 0.0
            for data, target in val_loader:
                # move tensors to GPU if CUDA is available
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                val_loss += loss.item() * data.size(0)
                # accuracy
                predict_y = torch.max(output, dim=1)[1]
                accuracy = accuracy + (predict_y
                                       == target.to(device)).sum().item()
                pbar.update(1)
                pbar.set_description('val')
                pbar.set_postfix(**{
                    'loss': loss.item(),
                })
    accuracy = accuracy / len(val_loader.dataset)
    val_acc.append(accuracy)

    # calculate average losses
    train_losses.append(train_loss / len(train_loader.dataset))
    val_losses.append(val_loss / len(val_loader.dataset))
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)

    # print training/validation statistics
    print('\ttrain Loss: {:.6f} \tval Loss: {:.6f}'.format(
        train_loss, val_loss))
    print('\taccuracy: {:.6f}'.format(accuracy))
    # save model if validation loss has decreased
    if val_loss <= val_loss_min:
        print(
            'val loss decreased ({:.6f} --> {:.6f}).  saving model ...'.format(
                val_loss_min, val_loss))
        torch.save(model.state_dict(), './classification/logs/model.pth')
        val_loss_min = val_loss

# 繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.legend(loc='best')
plt.savefig('./classification/image/acc.jpg')
plt.show()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend(loc='best')
plt.savefig('./classification/image/loss.jpg')
plt.show()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.plot(lr, label='Learning Rate')
plt.legend(loc='best')
plt.savefig('./classification/image/lr.jpg')
plt.show()