import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.models as models
from torch.autograd import Variable
from torchvision import datasets
from torchvision import utils as vutils
from torch.autograd.gradcheck import zero_gradients

import math
import copy
from PIL import Image
from deepfool_fashion import deepfool
from pathlib import Path

from model import utils
import os
import time 
import datetime

from Architectures import MNISTNet_ori

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

testset = datasets.MNIST(
    root='./mnist/',
    train=False,
    download=True,
    transform=transform)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=1,
    shuffle=True,
    )


net = MNISTNet_ori().to(device)
net.load_state_dict(torch.load('./model/MNIST/Tclassifier.pth', map_location=device))
net.eval()


# 前向传播，输出网络预测值
def output(sample, net):
    sample = sample.float().to(device)
    output = net(sample).cpu()
    _, pre = torch.max(output.data, 1)
    return pre


false = 0        # 原网络的分类错误数
correct = 0      # Deepfool攻击后，分类正确的数
total = 0
per_num = 0      #对抗样本个数
advlist = []
for images, labels in test_loader:
    total += labels.size(0)
    predicted = output(images, net)
    # 只对分类正确的样本，生成对抗样本
    if predicted==labels:
        r, loop_i, label_orig, label_pert, pert_image = deepfool(images, net, device, overshoot=0.05, max_iter=100)
        if label_pert == label_orig:
            correct += 1
        else:
            advlist.append(pert_image.reshape(1, 28, 28).numpy())  # 对抗样本
            per_num += 1
    else:
        false += 1
    if total%1000==0:
        print('准确率: %.4f %%' % (100 * correct / total))

np.save('./dataset/deepfool/deepfool1.npy', advlist)
print('准确率: %.4f %%' % (100 * correct / total)) 
print('原始准确率: %.4f %%' % (100 * (1 - false / total)))