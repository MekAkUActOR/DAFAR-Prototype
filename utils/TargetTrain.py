#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:57:54 2020

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''define hyperparameter'''
BATCH_SIZE = 50         # 批的大小
LR1 = 0.005
LR2 = 0.005             # 初始学习率
EPOCH = 50              # 遍历训练集的次数
lr_decay = 0.5          # 学习率衰减率
lr_destep = 10          # 学习率衰减阶梯

'''dataset'''
transform = transforms.ToTensor()

trainset = tv.datasets.MNIST(
    root='./mnist/',
    train=True,
    download=True,
    transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

testset = tv.datasets.MNIST(
    root='./mnist/',
    train=False,
    download=True,
    transform=transform)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

'''network'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(     
            nn.Conv2d(1, 32, 3, padding = 1), 
            nn.ReLU(),      
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)

        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 200),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(200, 10)
        
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x, indices1 = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x, indices2 = self.maxpool(x)
        encoded = x

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return encoded, indices1, indices2, x
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_t1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, padding = 1),
            nn.ReLU(),
            )

        self.conv_t2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding = 1),
            nn.ReLU(),
            )
        self.conv_t3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, padding = 1),
            nn.ReLU(),
            )
        self.conv_t4 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 3, padding = 1),
            nn.Tanh(),
            )
        self.unpool = nn.MaxUnpool2d(2,2)

    def forward(self, x, indices1, indices2):
        
        #print(x.shape)
        x = self.unpool(x, indices2)
        #print(x.shape)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.unpool(x, indices1)
        x = self.conv_t3(x)
        x = self.conv_t4(x)
        decoded = x
        return decoded
    
    
'''

        |\        |=|
        | \       | |
 x ---->|  |--|-->| |---->f(x)
        | /   |   | |
        |/    |   |=|
        `     |
          /|  |
         / |  |
ae(x)<--|  |--|
         \ |   
          \|    

'''

net = Net().to(device)
decoder = Decoder().to(device)

optimizer1 = optim.SGD(net.parameters(), lr=LR1, momentum=0.9)
optimizer2 = optim.SGD(decoder.parameters(), lr=LR2, momentum=0.9)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
#optimizer = optim.Adam(decoder.parameters(), lr=LR)

scheduler = optim.lr_scheduler.StepLR(optimizer2, step_size=lr_destep, gamma=lr_decay)

if __name__ == "__main__":

    for epoch in range(EPOCH):
        sum_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # print(inputs)
            encoded, indices1, indices2, outputs = net(inputs)
            decoded = decoder(encoded, indices1, indices2)

            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(decoded, inputs)
            loss = loss2 + loss1

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            sum_loss = sum_loss + loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.05f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        scheduler.step()

        with torch.no_grad():
            net.eval()
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                encoded, indices1, indices2, outputs= net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('%d epoch: acc %d%%  correct:%d  total:%d' % (epoch + 1, (100 * correct / total),correct,total))
        torch.cuda.empty_cache()

        with torch.no_grad():
            net.eval()
            decoder.eval()
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                encoded, indices1, indices2, outputs= net(images)
                decoded = decoder(encoded, indices1, indices2)
                temp, temp1, temp2, outputs = net(decoded)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('%d epoch: cir1acc %d%%  correct:%d  total:%d' % (epoch + 1, (100 * correct / total),correct,total))
        torch.cuda.empty_cache()
            
    torch.save(net.state_dict(), './model/MNIST/Tclassifier.pth')
    torch.save(decoder.state_dict(), './model/MNIST/Decoder.pth')
    
