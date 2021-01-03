import torch

# 查看可用的模型
print(torch.hub.list('pytorch/vision'))
print(torch.hub.list('facebookresearch/WSL-Images'))
# 將模型儲存起來
model = torch.hub.load('facebookresearch/WSL-Images',
                       'resnext101_32x8d_wsl',
                       force_reload=True,
                       verbose=1)
torch.save(model, './classification/pretrained_model/resnext101_32x8d.pth')