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

from tools.dataloader_dali import get_imagenet_iter_dali

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

# torch.set_num_threads(12)

path_train = r'D:\training_dataset\simpsons\train'
path_val = r'D:\training_dataset\simpsons\val'

path_train = r'/media/blackyyen/新增磁碟區1/training_dataset/simpsons/train'
path_val = r'/media/blackyyen/新增磁碟區1/training_dataset/simpsons/val'
data_dir = r'/simpsons'

train_loader = get_imagenet_iter_dali(
    type='train',
    image_dir='/media/blackyyen/新增磁碟區1/training_dataset/simpsons',
    batch_size=250,
    num_threads=6,
    crop=224,
    device_id=0,
    num_gpus=1)
val_loader = get_imagenet_iter_dali(
    type='val',
    image_dir='/media/blackyyen/新增磁碟區1/training_dataset/simpsons',
    batch_size=250,
    num_threads=6,
    crop=224,
    device_id=0,
    num_gpus=1)
print('start iterate')

# learning rate
lr = 1e-3

# 讀取模型
device = torch.device("cuda")
model = torch.load('./classification/pretrained_model/resnext101_32x8d.pth')

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
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                           T_max=100,
                                           eta_min=1e-4,
                                           last_epoch=-1)
model.to(device, non_blocking=True)
train_scaler = torch.cuda.amp.GradScaler()
val_scaler = torch.cuda.amp.GradScaler()
# number of epochs to train the model
n_epochs = 100

val_loss_min = np.Inf  # track change in validation loss

train_losses, val_losses, lr = [], [], []

for epoch in range(1, n_epochs + 1):
    # keep track of training and validation loss
    train_loss = 0.0
    val_loss = 0.0
    print('running epoch: {}'.format(epoch))
    # 訓練模型
    model.train()
    with tqdm(train_loader) as pbar:
        for data in train_loader:
            # move tensors to GPU if CUDA is available
            data, target = data[0]['data'].to(
                device,
                non_blocking=True), data[0]['label'].squeeze().long().to(
                    device, non_blocking=True)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            # calculate the batch loss
            with torch.cuda.amp.autocast():
                y_pred = model(data).squeeze()
                loss = criterion(y_pred, target)
                # backward pass: compute gradient of the loss with respect to model parameters
            train_scaler.scale(loss).backward()
            loss = loss.detach().cpu().numpy()
            train_scaler.step(optimizer)
            train_scaler.update()
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
            correct = 0.0
            for data, target in val_loader:
                # move tensors to GPU if CUDA is available
                data, target = data.to(device), target.to(device)
                # forward pass: compute predicted outputs by passing inputs to the model
                with torch.cuda.amp.autocast():
                    y_pred = model(data).squeeze()
                    loss = criterion(y_pred, target)
                loss = loss.detach().cpu().numpy()
                # update average validation loss
                val_loss += loss.item() * data.size(0)
                # accuracy
                predict_y = torch.max(y_pred, dim=1)[1]
                correct = correct + (predict_y
                                     == target.to(device)).sum().item()
                pbar.update(1)
                pbar.set_description('val')
                pbar.set_postfix(**{
                    'loss': loss.item(),
                })
    accuracy = correct / len(val_loader.dataset)

    # calculate average losses
    train_losses.append(train_loss / len(train_loader.dataset))
    val_losses.append(val_loss / len(val_loader.dataset))
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)

    # print training/validation statistics
    print('\ttrain loss: {:.6f} \tval loss: {:.6f}'.format(
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