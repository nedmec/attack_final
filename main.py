from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.models as models
from PIL import Image
from transformers import get_cosine_schedule_with_warmup
from fgsm import fgsm_attack
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 这里的扰动量先设定为几个值，后面可视化展示不同的扰动量影响以及成像效果
di = ['飞机','汽车','鸟','猫','鹿','狗','青蛙','马','船','卡车']


# 看看我们有没有配置GPU，没有就是使用cpu
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# 实例化模型并移动到GPU
pretrained_model = "./data/model.pt"
model = torch.load(pretrained_model)
# 设置为验证模式. 
model.eval()

image_path = "./data/1.jpg"  
image = Image.open(image_path).convert("RGB")  # 加载图片并转换为RGB格式

# 定义预处理转换
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片尺寸为模型输入大小
    transforms.ToTensor(),  # 转换为张量
])

input_tensor = preprocess(image)  # 应用预处理转换
input_tensor = input_tensor.to(device)

input_batch = input_tensor.unsqueeze(0)
input_batch.requires_grad = True
out = model(input_batch)
pred = out.max(1, keepdim=True)[1]
print(di[pred.item()])
output = model(input_batch)
pre= pred.flatten()
loss = F.nll_loss(output, pre)  # 计算损失函数
loss.backward()  # 计算梯度
data_grad = input_batch.grad
# 调用FGSM攻击


perturbed_data = fgsm_attack(input_batch, 10, data_grad)
out = model(perturbed_data )
pred = out.max(1, keepdim=True)[1]
print(di[pred.item()])

perturbed_data = perturbed_data.squeeze().detach().cpu().numpy()
perturbed_data = np.transpose(perturbed_data, (1, 2, 0))
plt.imshow(image)
