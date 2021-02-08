#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:49:18 2020

@author: hongxing
"""

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)  # reproducible

def Graying(img):
    i = 0
    grayed = 0
    for channel in img:
        grayed = grayed + abs(channel)
        i += 1
    grayed = grayed/i
    return grayed[np.newaxis,:]


'''MyDataset'''
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = np.load(data)
    def __getitem__(self, index):
        hdct = self.data[index, :, :, :]
        hdct = torch.from_numpy(hdct)
        return hdct
    def __len__(self):
        return self.data.shape[0]


class GrayDataset(Dataset):
    def __init__(self, data):
        self.data = np.load(data)
    def __getitem__(self, index):
        hdct = self.data[index, :, :, :]
        hdct = Graying(hdct)
        hdct = torch.from_numpy(hdct)
        return hdct
    def __len__(self):
        return self.data.shape[0]

'''
dataset=MyDataset('./dataset/MNIST/fgsm/fgsm1.npy')
mstfgsmloader= DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

for inputs in mstfgsmloader:
    print(inputs.shape
    break
'''