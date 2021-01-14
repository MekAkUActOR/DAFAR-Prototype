#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:23:41 2020

@author: hongxing
"""

import torch
import os
import torchvision as tv
import pandas as pd
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import scipy.misc
import imageio
from PIL import Image
import matplotlib.pyplot as plt

from Architectures import CIFDtcAnom
from mydataloader import GrayDataset, MyDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''define hyperparameter'''
BATCH_SIZE = 50         # 批的大小
LR = 0.01
EPOCH = 400              # 遍历训练集的次数
lr_decay = 0.5          # 学习率衰减率
lr_destep = 70          # 学习率衰减阶梯

'''dataset'''

ciftrainset = MyDataset('./dataset/CIFAR10/normal/trainre.npy')
ciftrainloader = torch.utils.data.DataLoader(ciftrainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

ciftestset = MyDataset('./dataset/CIFAR10/normal/testre.npy')
ciftestloader = torch.utils.data.DataLoader(ciftestset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


net = CIFDtcAnom().to(device)

optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

criterion = nn.MSELoss()
#optimizer = optim.Adam(decoder.parameters(), lr=LR)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_destep, gamma=lr_decay)

if __name__ == "__main__":

    for epoch in range(EPOCH):
        sum_loss = 0.0
        for i, data in enumerate(ciftrainloader):
            inputs = torch.reshape(data, (BATCH_SIZE, 3*32*32))
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            # print(inputs)
            recons = net(inputs)
            
            loss = criterion(recons, inputs)

            loss.backward()

            optimizer.step()

            sum_loss = sum_loss + loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.05f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        scheduler.step()

        with torch.no_grad():
            net.eval()
            l2sum = 0.0
            i = 0
            for data in ciftestloader:
                inputs = torch.reshape(data, (BATCH_SIZE, 3*32*32))
                inputs = inputs.to(device)
                outputs= net(inputs)
                l2sum = l2sum + criterion(outputs, inputs).item()
                i += 1
            print('Avg L2 distance in testset:', l2sum / i)
        torch.cuda.empty_cache()
            
    torch.save(net.state_dict(), './model/DETECTOR/CIFDtcAnomL2.pth')
    
