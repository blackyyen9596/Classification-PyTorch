import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import datasets
import os
from PIL import Image
from tools.to_csv import to_csv
from PIL import Image

path_test = r'D:\Github\BlackyYen\BlackyYen-public\machine_learning\Classification\ntut-ml-2020-classification\test\test'
class_path = r'D:\training_dataset\simpsons\test'
weights_path = r'.\classification\logs\best_weight.pth'
classes = os.listdir(class_path)
# print(classes)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 讀取模型
device = torch.device("cuda")
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 20)
model.to(device)
model.load_state_dict(torch.load(weights_path))
model.eval()

result = []
for i in tqdm(range(1, 991)):
    drain = os.path.join(path_test, '%s.jpg' % i)
    fopen = Image.open(drain).convert('RGB')
    data = test_transforms(fopen).to(device)
    output = model(data[None, ...])
    predict_y = torch.max(output, dim=1)[1]
    result.append(classes[int(predict_y)])

# 將結果寫入csv檔中
to_csv(id_read_path=
       r'../Classification/ntut-ml-2020-classification/sampleSubmission.csv',
       save_path=r'../Classification/results/best_weight.csv',
       y_pred=result)