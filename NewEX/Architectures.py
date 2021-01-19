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


'''network'''
class MNISTNet_ori(nn.Module):
    def __init__(self):
        super(MNISTNet_ori, self).__init__()
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

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

'''------------------------------------------'''

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
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


class MNISTDecoder(nn.Module):
    def __init__(self):
        super(MNISTDecoder, self).__init__()
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


class MSTreAE(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
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

        x = self.unpool(encoded, indices2)
        #print(x.shape)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.unpool(x, indices1)
        x = self.conv_t3(x)
        x = self.conv_t4(x)
        decoded = x

        x = encoded.view(encoded.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return decoded, x


'''------------------------------------------'''
'''------------------------------------------'''
'''------------------------------------------'''

class CIFAR10Net_ori(nn.Module):
    def __init__(self):
        super(CIFAR10Net_ori, self).__init__()
        self.conv1 = nn.Sequential(     
            nn.Conv2d(3, 96, 3, padding = 1), 
            nn.ReLU(),      
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(192, 192, 1),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(192, 10, 1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 10, 200),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(200, 10),
        )

    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x, indices1 = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x, indices2 = self.maxpool(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

'''------------------------------------------'''

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Sequential(     
            nn.Conv2d(3, 96, 3, padding = 1), 
            nn.ReLU(),      
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(192, 192, 3, padding = 1),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(192, 192, 1),
            nn.ReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(192, 10, 1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 8 * 10, 200),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(200, 200),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(200, 10),
        )

    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x, indices1 = self.maxpool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x, indices2 = self.maxpool(x)
        encoded = x

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return encoded, indices1, indices2, x
    

class CIFAR10Decoder(nn.Module):
    def __init__(self):
        super(CIFAR10Decoder, self).__init__()
        self.conv_t1 = nn.Sequential(
            nn.ConvTranspose2d(192, 192, 3, padding = 1),
            nn.ReLU(),
            )

        self.conv_t2 = nn.Sequential(
            nn.ConvTranspose2d(192, 192, 3, padding = 1),
            nn.ReLU(),
            )
        self.conv_t3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 3, padding = 1),
            nn.ReLU(),
            )
        self.conv_t4 = nn.Sequential(
            nn.ConvTranspose2d(96, 96, 3, padding = 1),
            nn.ReLU(),
            )
        self.conv_t5 = nn.Sequential(
            nn.ConvTranspose2d(96, 96, 3, padding = 1),
            nn.ReLU(),
            )
        self.conv_t6 = nn.Sequential(
            nn.ConvTranspose2d(96, 3, 3, padding = 1),
            nn.Tanh(),
            )
        self.unpool = nn.MaxUnpool2d(2,2)

    def forward(self, x, indices1, indices2):
        
        x = self.unpool(x, indices2)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.conv_t3(x)
        x = self.unpool(x, indices1)
        x = self.conv_t4(x)
        x = self.conv_t5(x)
        x = self.conv_t6(x)
        decoded = x
        return decoded
    
'''------------------------------------------'''
'''------------------------------------------'''
'''------------------------------------------'''

class MSTDtcAnom(nn.Module):
    def __init__(self):
        super(MSTDtcAnom, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            )
        self.fc_t1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            )
        self.fc_t2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            )
        self.fc_t3 = nn.Sequential(
            nn.Linear(256, 28 * 28),
            nn.Tanh(),
            )

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc_t1(x)
        x = self.fc_t2(x)
        x = self.fc_t3(x)
        decoded = x
        return decoded
      
        
class CIFDtcAnomGray(nn.Module):
    def __init__(self):
        super(CIFDtcAnomGray, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 32, 512),
            nn.ReLU(),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            )
        self.fc_t1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            )
        self.fc_t2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            )
        self.fc_t3 = nn.Sequential(
            nn.Linear(512, 32 * 32),
            nn.Tanh(),
            )

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc_t1(x)
        x = self.fc_t2(x)
        x = self.fc_t3(x)
        decoded = x
        return decoded


class CIFDtcAnom(nn.Module):
    def __init__(self):
        super(CIFDtcAnom, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            )
        self.fc_t1 = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            )
        self.fc_t2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            )
        self.fc_t3 = nn.Sequential(
            nn.Linear(512, 3 * 32 * 32),
            nn.Tanh(),
            )

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc_t1(x)
        x = self.fc_t2(x)
        x = self.fc_t3(x)
        decoded = x
        return decoded