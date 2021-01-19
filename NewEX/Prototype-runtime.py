#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:39:06 2020

@author: hongxing
"""

import sys
import argparse

import torch
import os
import torchvision as tv
import pandas as pd
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.misc
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from layers import SinkhornDistance

from Architectures import CIFDtcAnom, MSTDtcAnom
from mydataloader import MyDataset, GrayDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.description='configuration'
parser.add_argument("-i", "--input", help="path of input picture", required=True)
parser.add_argument("-t", "--threshold", help="anomaly score threshold", type=float, required=True)
args = parser.parse_args()


print(args)



def AnomScore(inputs, detector):

    inputs = inputs.to(device)
    outputs = detector(inputs)
    '''
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
    dis, P, C = sinkhorn(outputs, inputs)
    wdistances.append(dis.data.numpy().squeeze(0))
    '''
    substract = (inputs - outputs).cpu().detach().numpy().squeeze(0)
    l2 = 0
    for channel in substract:
        l2 += np.linalg.norm(channel)
    return l2

def AnomScoreSet(dataset, detector, c, h, w):
    wdistances = []
    for data in dataset:
        inputs = torch.reshape(data, (1, c*h*w))
        inputs = inputs.to(device)
        wdistances.append(AnomScore(inputs, detector))
    wdistances = np.array(wdistances)
    return wdistances
'''
msttestset = MyDataset('./dataset/MNIST/normal/testre.npy')
msttestloader = torch.utils.data.DataLoader(msttestset, batch_size=1, shuffle=True, pin_memory=True)

mdetector = MSTDtcAnom().to(device)
mdetector.load_state_dict(torch.load('./model/DETECTOR/MSTDtcAnomL2.pth', map_location=torch.device('cpu')))
mdetector.eval()
'''

ciftestset = MyDataset('./dataset/CIFAR10/normal/testre.npy')
ciftestloader = torch.utils.data.DataLoader(ciftestset, batch_size=1, shuffle=True, pin_memory=True)

cdetector = CIFDtcAnom().to(device)
cdetector.load_state_dict(torch.load('./model/DETECTOR/CIFDtcAnomL2.pth', map_location=torch.device('cpu')))
cdetector.eval()

'''
wdistances = AnomScoreSet(msttestloader, mdetector, 1, 28, 28)
np.save('./distance/l2/MNIST/normal/testrere.npy', wdistances)
print('Max',np.max(wdistances))
print('Min',np.min(wdistances))
print('Mean',np.mean(wdistances))
print('Mid',np.median(wdistances))
print('------------------------------------')


wdistances = AnomScoreSet(ciftestloader, cdetector, 3, 32, 32)
np.save('./distance/l2/CIFAR10/normal/testrere.npy', wdistances)
print('Max',np.max(wdistances))
print('Min',np.min(wdistances))
print('Mean',np.mean(wdistances))
print('Mid',np.median(wdistances))
print('------------------------------------')
'''


i = 1
while i <= 8:
    '''
    mstfgsmset = MyDataset('./dataset/MNIST/fgsm/fgsm'+str(i)+'re.npy')
    mstfgsmloader = torch.utils.data.DataLoader(mstfgsmset, batch_size=1, shuffle=True, pin_memory=True)
    
    mstpgdset = MyDataset('./dataset/MNIST/pgd/pgd'+str(i)+'re.npy')
    mstpgdloader = torch.utils.data.DataLoader(mstpgdset, batch_size=1, shuffle=True, pin_memory=True)
      
    ciffgsmset = MyDataset('./dataset/CIFAR10/fgsm/fgsm'+str(i)+'re.npy')
    ciffgsmloader = torch.utils.data.DataLoader(ciffgsmset, batch_size=1, shuffle=True, pin_memory=True)
    '''
    cifpgdset = MyDataset('./dataset/CIFAR10/pgd/pgd'+str(i)+'re.npy')
    cifpgdloader = torch.utils.data.DataLoader(cifpgdset, batch_size=1, shuffle=True, pin_memory=True)
    
    '''
    wdistances = AnomScoreSet(mstfgsmloader, mdetector, 1, 28, 28)
    np.save('./distance/l2/MNIST/fgsm/fgsm'+str(i)+'rere.npy', wdistances)
    print('MNIST fgsm', i)
    print('Max',np.max(wdistances))
    print('Min',np.min(wdistances))
    print('Mean',np.mean(wdistances))
    print('Mid',np.median(wdistances))
    print('------------------------------------')
    
    wdistances = AnomScoreSet(mstpgdloader, mdetector, 1, 28, 28)
    np.save('./distance/l2/MNIST/pgd/pgd'+str(i)+'rere.npy', wdistances)
    print('MNIST pgd', i)
    print('Max',np.max(wdistances))
    print('Min',np.min(wdistances))
    print('Mean',np.mean(wdistances))
    print('Mid',np.median(wdistances))
    print('------------------------------------')
    
    
    wdistances = AnomScoreSet(ciffgsmloader, cdetector, 3, 32, 32)
    np.save('./distance/l2/CIFAR10/fgsm/fgsm'+str(i)+'rere.npy', wdistances)
    print('CIFAR10 fgsm', i)
    print('Max',np.max(wdistances))
    print('Min',np.min(wdistances))
    print('Mean',np.mean(wdistances))
    print('Mid',np.median(wdistances))
    print('------------------------------------')
    '''
    wdistances = AnomScoreSet(cifpgdloader, cdetector, 3, 32, 32)
    np.save('./distance/l2/CIFAR10/pgd/pgd'+str(i)+'rere.npy', wdistances)
    print('CIFAR10 pgd', i)
    print('Max',np.max(wdistances))
    print('Min',np.min(wdistances))
    print('Mean',np.mean(wdistances))
    print('Mid',np.median(wdistances))
    print('------------------------------------')
    i += 1





