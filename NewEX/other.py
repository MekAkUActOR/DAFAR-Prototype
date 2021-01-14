#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:34:14 2020

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
from Architectures import CIFAR10Net_ori, MNISTNet_ori, CIFAR10Net, CIFAR10Decoder, MNISTNet, MNISTDecoder, CIFDtcAnom, MSTDtcAnom
from mydataloader import MyDataset, GrayDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#orilist, advlist, oriftlist, advftlist, orireclist, advreclist = np.load('./examples/MNIST/ori.npy'), np.load('./examples/MNIST/adv.npy'), np.load('./examples/MNIST/orift.npy'), np.load('./examples/MNIST/advft.npy'), np.load('./examples/MNIST/orirec.npy'), np.load('./examples/MNIST/advrec.npy')

'''

net = MNISTNet().to(device)
net.load_state_dict(torch.load('./model/MNIST/Tclassifier.pth', map_location=torch.device('cpu')))
net.eval()


i = 0
avgnoi = 0
avgadv = 0
while i<=9:
    ori = torch.from_numpy(orilist[i]).to(device)
    ori_noi = ori + 0.2 + 0.2 * torch.randn(ori.shape).to(device)
    ori_adv = torch.from_numpy(advlist[i]).to(device)
    
    encoded, indices1, indices2, outputs = net(ori)
    encodednoi, indices1, indices2, outputs = net(ori_noi)
    encodedadv, indices1, indices2, outputs = net(ori_adv)
    
    noil2 = np.linalg.norm(encoded.detach().numpy().reshape(56,56)-encodednoi.detach().numpy().reshape(56,56))
    advl2 = np.linalg.norm(encoded.detach().numpy().reshape(56,56)-encodedadv.detach().numpy().reshape(56,56))
    print(noil2, advl2)
    avgnoi += noil2
    avgadv += advl2
    
    i += 1
print(avgnoi, avgadv)
'''


def maxinterval(a, b, c, d, e):
    wide = []
    wide.append(np.max(a) - np.min(a))
    wide.append(np.max(b) - np.min(b))
    wide.append(np.max(c) - np.min(c))
    wide.append(np.max(d) - np.min(d))
    wide.append(np.max(e) - np.min(e))
    maxint = np.max(np.array(wide))/50
    return maxint
    

def defaxis(arry):
    wide = np.max(arry) - np.min(arry)
    interval = wide/50
    block = []
    for i in range(0, 51):
        block.append(np.min(arry) + i*interval)
    return np.min(arry), np.max(arry), np.mean(arry), np.median(arry), interval, block

def calnums(arry):
    Min, Max, Avg, Mid, interval, block = defaxis(arry)
    print(Min, Max, Avg, Mid, interval)
    nums = np.zeros((50,))
    for i in range(0, 50):
        for x in arry:
            if x >= block[i] and x < block[i + 1]:
                nums[i] += 1
        nums[i] = nums[i]/len(arry) /interval
    xaxis = []
    for i in range(0, 50):
        xaxis.append(block[i]+interval/2)
    return xaxis, nums

def setthres(arry):
    var = np.std(arry)
    thres = np.mean(arry) + 2*var
    return thres
      

def calportionADV(arry, thres):
    right = 0
    for i in arry:
        if i>=thres:
            right += 1
    return right/len(arry)
    
def calportionNOR(arry, thres):
    right = 0
    for i in arry:
        if i<=thres:
            right += 1
    return right/len(arry)

[0.9984]
'''
intensity = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


mstdarfa = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mstonlyclass = [76.37, 99.45, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mstmagnet = [69.46, 88.33, 98.79, 99.16, 99.86, 100.0, 100.0, 100.0]
mstfeatsq = [74.68, 92.12, 97.46, 99.69, 100.0, 100.0, 100.0, 100.0]



mstdarfa1 = [67.59, 99.28, 99.96, 100.0, 100.0, 100.0, 100.0, 100.0]
mstonlyclass1 = [58.44, 95.45, 96.02, 97.13, 97.95, 98.43, 99.12, 99.24]
mstmagnet1 = [47.46, 73.33, 95.79, 99.16, 100.0, 100.0, 100.0, 100.0]
mstfeatsq1 = [60.68, 79.12, 94.46, 97.69, 99.12, 100.00, 100.0, 100.0]
'''

'''
mstdarfa = [64.66, 85.79, 86.02, 86.03, 85.79, 85.94, 86.00, 86.04]
mstonlyclass = [84.92, 80.45, 66.02, 55.13, 50.95, 47.43, 45.12, 40.24]
mstmagnet = [84.46, 85.87, 86.02, 86.10, 85.84, 85.98, 86.00, 86.02]
#mstfeatsq = [60.68, 79.12, 94.46, 97.69, 99.12, 100.00, 100.0, 100.0]
'''

'''
intensity = ['FGSM', 'JSMA', 'CW', 'PGD']

mstdarfa = [100.0, 100.0, 100.0, 100.0]
mstonlyclass = [100.0, 63.36, 45.12, 73.98]
mstmagnet = [100.0, 100.0, 100.0, 100.0]
mstfeatsq = [99.69, 98.34, 100.0, 100.0]

mstdarfa1 = [100.0, 100.0, 100.0, 100.0]
mstonlyclass1 = [97.13, 32.8, 17.20, 50.07]
mstmagnet1 = [99.16, 98.56, 98.01, 100.0]
mstfeatsq1 = [97.69, 98.34, 96.10, 99.45]
‘’‘

’‘’
plt.figure(figsize=(8,4))
plt.subplots_adjust(left=0.1, top= 0.93, right = 0.95, bottom = 0.15, hspace = 0.25)

plt.subplot(1,2,1)
plt.title('(a) MNIST', fontsize=15)
plt.plot(intensity, mstdarfa, 'r-s', label='DAFAR', linewidth=1.5)
plt.plot(intensity, mstonlyclass, 'y-v', label='Only binary classifier', linewidth=1.5)
plt.plot(intensity, mstmagnet, 'c-o', label='Detector of MagNet', linewidth=1.5)
plt.plot(intensity, mstfeatsq, 'c-d', label='Feature Squeezing', linewidth=1.5)
plt.legend(loc=3, fontsize = 12)
plt.xlabel('Attack method', fontsize=15)
plt.ylabel('Detection accuracy/%', fontsize=15)
plt.ylim(0, 110)
plt.tick_params(labelsize=12)
plt.grid(ls='--')

plt.subplot(1,2,2)
plt.title('(b) CIFAR-10', fontsize=15)
plt.plot(intensity, mstdarfa1, 'r-s', label='DAFAR', linewidth=1.5)
plt.plot(intensity, mstonlyclass1, 'y-v', label='Only binary classifier', linewidth=1.5)
plt.plot(intensity, mstmagnet1, 'c-o', label='Detector of MagNet', linewidth=1.5)
plt.plot(intensity, mstfeatsq1, 'c-d', label='Feature Squeezing', linewidth=1.5)
plt.legend(loc=6, fontsize = 12)
plt.xlabel('Attack method', fontsize=15)
#plt.ylabel('Detection accuracy/%', fontsize=15)
plt.ylim(0, 110)
plt.tick_params(labelsize=12)
plt.grid(ls='--')

plt.savefig('./figures/dtcaccmed.eps')
#plt.show()
plt.close()
'''


arrytype = 're'
xname = '$L_2$ Distance'
filename = 'reconserr.eps'
#'reconserr.eps'
#'AnomScore.eps'
thres1 = 23.333
thres2 = 230.143

mst = np.load('./distance/l2/MNIST/normal/test'+arrytype+'.npy')

mstfgsm1 = np.load('./distance/l2/MNIST/fgsm/fgsm2'+arrytype+'.npy')
mstfgsm2 = np.load('./distance/l2/MNIST/fgsm/fgsm4'+arrytype+'.npy')
mstfgsm3 = np.load('./distance/l2/MNIST/fgsm/fgsm6'+arrytype+'.npy')
mstfgsm4 = np.load('./distance/l2/MNIST/fgsm/fgsm8'+arrytype+'.npy')

mstpgd1 = np.load('./distance/l2/MNIST/pgd/pgd2'+arrytype+'.npy')
mstpgd2 = np.load('./distance/l2/MNIST/pgd/pgd4'+arrytype+'.npy')
mstpgd3 = np.load('./distance/l2/MNIST/pgd/pgd6'+arrytype+'.npy')
mstpgd4 = np.load('./distance/l2/MNIST/pgd/pgd8'+arrytype+'.npy')


cif = np.load('./distance/l2/CIFAR10/normal/test'+arrytype+'.npy')

ciffgsm1 = np.load('./distance/l2/CIFAR10/fgsm/fgsm2'+arrytype+'.npy')
ciffgsm2 = np.load('./distance/l2/CIFAR10/fgsm/fgsm4'+arrytype+'.npy')
ciffgsm3 = np.load('./distance/l2/CIFAR10/fgsm/fgsm6'+arrytype+'.npy')
ciffgsm4 = np.load('./distance/l2/CIFAR10/fgsm/fgsm8'+arrytype+'.npy')

cifpgd1 = np.load('./distance/l2/CIFAR10/pgd/pgd2'+arrytype+'.npy')
cifpgd2 = np.load('./distance/l2/CIFAR10/pgd/pgd4'+arrytype+'.npy')
cifpgd3 = np.load('./distance/l2/CIFAR10/pgd/pgd6'+arrytype+'.npy')
cifpgd4 = np.load('./distance/l2/CIFAR10/pgd/pgd8'+arrytype+'.npy')


plt.figure(figsize=(17,4))
plt.subplots_adjust(left=0.05, top= 0.93, right = 0.95, bottom = 0.12, hspace = 0.25)
plt.subplot(1, 4, 1)
plt.title('(a) MNIST-FGSM', fontsize=12)
x0, y0 = calnums(mst)
x1, y1 = calnums(mstfgsm1)
x2, y2 = calnums(mstfgsm2)
x3, y3 = calnums(mstfgsm3)
x4, y4 = calnums(mstfgsm4)
plt.plot(x0, y0, color='b', label='Normal', marker='o', markersize=3, linewidth=1)
plt.plot(x1, y1, color='c', label='FGSM 0.1', marker='v', markersize=3, linewidth=1)
plt.plot(x2, y2, color='g', label='FGSM 0.2', marker='s', markersize=3, linewidth=1)
plt.plot(x3, y3, color='y', label='FGSM 0.3', marker='d', markersize=3, linewidth=1)
plt.plot(x4, y4, color='red', label='FGSM 0.4', marker='3', markersize=4, linewidth=1)
#plt.vlines(23.333,0,0.14, colors = "violet", linestyles = "dashed", linewidth=0.8)
plt.legend(loc=1, fontsize = 8)
plt.xlabel(xname, fontsize=12)
plt.ylabel('Proportion Density of Examples', fontsize=12)
plt.tick_params(labelsize=8)

plt.subplot(1, 4, 2)
plt.title('(b) MNIST-PGD', fontsize=12)
x0, y0 = calnums(mst)
x1, y1 = calnums(mstpgd1)
x2, y2 = calnums(mstpgd2)
x3, y3 = calnums(mstpgd3)
x4, y4 = calnums(mstpgd4)
plt.plot(x0, y0, color='b', label='Normal', marker='o', markersize=3, linewidth=1)
plt.plot(x1, y1, color='c', label='PGD 0.1', marker='v', markersize=3, linewidth=1)
plt.plot(x2, y2, color='g', label='PGD 0.2', marker='s', markersize=3, linewidth=1)
plt.plot(x3, y3, color='y', label='PGD 0.3', marker='d', markersize=3, linewidth=1)
plt.plot(x4, y4, color='red', label='PGD 0.4', marker='3', markersize=4, linewidth=1)
#plt.vlines(23.333,0,0.14, colors = "violet", linestyles = "dashed", linewidth=0.8)
plt.legend(loc=1, fontsize = 8)
plt.xlabel(xname, fontsize=12)
plt.tick_params(labelsize=8)

plt.subplot(1, 4, 3)
plt.title('(c) CIFAR10-FGSM', fontsize=12)
x0, y0 = calnums(cif)
x1, y1 = calnums(ciffgsm1)
x2, y2 = calnums(ciffgsm2)
x3, y3 = calnums(ciffgsm3)
x4, y4 = calnums(ciffgsm4)
plt.plot(x0, y0, color='b', label='Normal', marker='o', markersize=3, linewidth=1)
plt.plot(x1, y1, color='c', label='FGSM 0.1', marker='v', markersize=3, linewidth=1)
plt.plot(x2, y2, color='g', label='FGSM 0.2', marker='s', markersize=3, linewidth=1)
plt.plot(x3, y3, color='y', label='FGSM 0.3', marker='d', markersize=3, linewidth=1)
plt.plot(x4, y4, color='red', label='FGSM 0.4', marker='3', markersize=4, linewidth=1)
#plt.vlines(230.143,0,0.0175, colors = "violet", linestyles = "dashed", linewidth=0.8)
plt.legend(loc=1, fontsize = 8)
plt.xlabel(xname, fontsize=12)
#plt.ylabel('Proportion of Examples', fontsize=8)
plt.tick_params(labelsize=8)

plt.subplot(1, 4, 4)
plt.title('(d) CIFAR10-PGD', fontsize=12)
x0, y0 = calnums(cif)
x1, y1 = calnums(cifpgd1)
x2, y2 = calnums(cifpgd2)
x3, y3 = calnums(cifpgd3)
x4, y4 = calnums(cifpgd4)
plt.plot(x0, y0, color='b', label='Normal', marker='o', markersize=3, linewidth=1)
plt.plot(x1, y1, color='c', label='PGD 0.1', marker='v', markersize=3, linewidth=1)
plt.plot(x2, y2, color='g', label='PGD 0.2', marker='s', markersize=3, linewidth=1)
plt.plot(x3, y3, color='y', label='PGD 0.3', marker='d', markersize=3, linewidth=1)
plt.plot(x4, y4, color='red', label='PGD 0.4', marker='3', markersize=4, linewidth=1)
#plt.vlines(230.143,0,0.028, colors = "violet", linestyles = "dashed", linewidth=0.8)
plt.legend(loc=1, fontsize = 8)
plt.xlabel(xname, fontsize=12)
#plt.ylabel('Proportion of Examples', fontsize=8)
plt.tick_params(labelsize=8)

plt.savefig('./figures/'+filename)
#plt.show()
plt.close()















