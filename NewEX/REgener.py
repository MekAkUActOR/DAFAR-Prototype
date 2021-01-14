#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:39:06 2020

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
from layers import SinkhornDistance

from Architectures import CIFAR10Net, CIFAR10Decoder, MNISTNet, MNISTDecoder
from mydataloader import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def REgener(name, dataset, tnet, decoder):
    
    if name == 'normal':
        recons = []
        wdistances = []
        for data in dataset:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            encoded, indices1, indices2, outputs = tnet(inputs)
            decoded = decoder(encoded, indices1, indices2)
            '''
            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
            dis, P, C = sinkhorn(decoded, inputs)
            wdistances.append(dis.data.numpy().squeeze(0))
            '''
            substract = (inputs - decoded).cpu().detach().numpy().squeeze(0)
            l2 = 0
            for channel in substract:
                l2 += np.linalg.norm(channel)
            wdistances.append(l2)
            recons.append(substract)
        return recons, wdistances
    elif name == 'adversary':
        recons = []
        wdistances = []
        for data in dataset:
            inputs = data
            inputs = inputs.to(device)
            encoded, indices1, indices2, outputs = tnet(inputs)
            decoded = decoder(encoded, indices1, indices2)
            '''
            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
            dis, P, C = sinkhorn(decoded, inputs)
            wdistances.append(dis.data.numpy().squeeze(0))
            '''
            substract = (inputs - decoded).cpu().detach().numpy().squeeze(0)
            l2 = 0
            for channel in substract:
                l2 += np.linalg.norm(channel)
            wdistances.append(l2)
            recons.append(substract)
        return recons, wdistances



transform1 = transforms.ToTensor()

transform2 = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



msttrainset = tv.datasets.MNIST(
    root='./mnist/',
    train=True,
    download=True,
    transform=transform1)

msttrainloader = torch.utils.data.DataLoader(
    msttrainset,
    batch_size=1,
    shuffle=True,
    )

msttestset = tv.datasets.MNIST(
    root='./mnist/',
    train=False,
    download=True,
    transform=transform1)

msttestloader = torch.utils.data.DataLoader(
    msttestset,
    batch_size=1,
    shuffle=False,
    )

ciftrainset = tv.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform2)
ciftrainloader = torch.utils.data.DataLoader(ciftrainset, batch_size=1,
                                          shuffle=True, num_workers=2)

ciftestset = tv.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform2)
ciftestloader = torch.utils.data.DataLoader(ciftestset, batch_size=1,
                                         shuffle=False, num_workers=2)


i = 1
mstfgsmset = []
mstfgsmloader = []
mstpgdset = []
mstpgdloader = []
ciffgsmset = []
ciffgsmloader = []
cifpgdset = []
cifpgdloader = []
while i <= 8:
    '''
    mstfgsmset.append(MyDataset('./dataset/MNIST/fgsm/fgsm'+str(i)+'.npy'))
    mstfgsmloader.append(torch.utils.data.DataLoader(mstfgsmset[i - 1], batch_size=1, shuffle=True, pin_memory=True))
    
    mstpgdset.append(MyDataset('./dataset/MNIST/pgd/pgd'+str(i)+'.npy'))
    mstpgdloader.append(torch.utils.data.DataLoader(mstpgdset[i - 1], batch_size=1, shuffle=True, pin_memory=True))
    
    ciffgsmset.append(MyDataset('./dataset/CIFAR10/fgsm/fgsm'+str(i)+'.npy'))
    ciffgsmloader.append(torch.utils.data.DataLoader(ciffgsmset[i - 1], batch_size=1, shuffle=True, pin_memory=True))
    '''
    cifpgdset.append(MyDataset('./dataset/CIFAR10/pgd/pgd'+str(i)+'.npy'))
    cifpgdloader.append(torch.utils.data.DataLoader(cifpgdset[i - 1], batch_size=1, shuffle=True, pin_memory=True))
    i += 1


'''
net = MNISTNet().to(device)
net.load_state_dict(torch.load('./model/MNIST/Tclassifier.pth', map_location=torch.device('cpu')))
net.eval()

decoder = MNISTDecoder().to(device)
decoder.load_state_dict(torch.load('./model/MNIST/Decoder.pth', map_location=torch.device('cpu')))
decoder.eval()
'''
netc = CIFAR10Net().to(device)
netc.load_state_dict(torch.load('./model/CIFAR10/Tclassifier.pth', map_location=torch.device('cpu')))
netc.eval()

decoderc = CIFAR10Decoder().to(device)
decoderc.load_state_dict(torch.load('./model/CIFAR10/Decoder.pth', map_location=torch.device('cpu')))
decoderc.eval()

'''
recons, wdistances = REgener('normal', msttrainloader, net, decoder)
#np.save('./dataset/MNIST/normal/trainre.npy', np.array(recons))
np.save('./distance/l2/MNIST/normal/trainre.npy', np.array(wdistances))
print('Max',np.max(np.array(wdistances)))
print('Min',np.min(np.array(wdistances)))
print('Mean',np.mean(np.array(wdistances)))
print('Mid',np.median(np.array(wdistances)))
print('------------------------------------')

recons, wdistances = REgener('normal', msttestloader, net, decoder)
#np.save('./dataset/MNIST/normal/testre.npy', np.array(recons))
np.save('./distance/l2/MNIST/normal/testre.npy', np.array(wdistances))
print('Max',np.max(np.array(wdistances)))
print('Min',np.min(np.array(wdistances)))
print('Mean',np.mean(np.array(wdistances)))
print('Mid',np.median(np.array(wdistances)))
print('------------------------------------')

recons, wdistances = REgener('adversary', mstfgsmloader, net, decoder)
#np.save('./dataset/MNIST/fgsm/fgsm4re.npy', np.array(recons))
np.save('./distance/l2/MNIST/fgsm/fgsm4re.npy', np.array(wdistances))
print('Max',np.max(np.array(wdistances)))
print('Min',np.min(np.array(wdistances)))
print('Mean',np.mean(np.array(wdistances)))
print('Mid',np.median(np.array(wdistances)))
print('------------------------------------')

recons, wdistances = REgener('adversary', mstpgdloader, net, decoder)
#np.save('./dataset/MNIST/pgd/pgd4re.npy', np.array(recons))
np.save('./distance/l2/MNIST/pgd/pgd4re.npy', np.array(wdistances))
print('Max',np.max(np.array(wdistances)))
print('Min',np.min(np.array(wdistances)))
print('Mean',np.mean(np.array(wdistances)))
print('Mid',np.median(np.array(wdistances)))
print('------------------------------------')



recons, wdistances = REgener('normal', ciftrainloader, netc, decoderc)
#np.save('./dataset/CIFAR10/normal/trainre.npy', np.array(recons))
np.save('./distance/l2/CIFAR10/normal/trainre.npy', np.array(wdistances))
print('Max',np.max(np.array(wdistances)))
print('Min',np.min(np.array(wdistances)))
print('Mean',np.mean(np.array(wdistances)))
print('Mid',np.median(np.array(wdistances)))
print('------------------------------------')

recons, wdistances = REgener('normal', ciftestloader, netc, decoderc)
#np.save('./dataset/CIFAR10/normal/testre.npy', np.array(recons))
np.save('./distance/l2/CIFAR10/normal/testre.npy', np.array(wdistances))
print('Max',np.max(np.array(wdistances)))
print('Min',np.min(np.array(wdistances)))
print('Mean',np.mean(np.array(wdistances)))
print('Mid',np.median(np.array(wdistances)))
print('------------------------------------')
'''

i = 1
while i <= 8:
    '''
    recons, wdistances = REgener('adversary', mstfgsmloader[i - 1], net, decoder)
    np.save('./dataset/MNIST/fgsm/fgsm'+str(i)+'re.npy', np.array(recons))
    np.save('./distance/l2/MNIST/fgsm/fgsm'+str(i)+'re.npy', np.array(wdistances))
    print('MNIST FGSM', i)
    print('Max',np.max(np.array(wdistances)))
    print('Min',np.min(np.array(wdistances)))
    print('Mean',np.mean(np.array(wdistances)))
    print('Mid',np.median(np.array(wdistances)))
    print('------------------------------------')
    
    recons, wdistances = REgener('adversary', mstpgdloader[i - 1], net, decoder)
    np.save('./dataset/MNIST/pgd/pgd'+str(i)+'re.npy', np.array(recons))
    np.save('./distance/l2/MNIST/pgd/pgd'+str(i)+'re.npy', np.array(wdistances))
    print('MNIST PGD', i)
    print('Max',np.max(np.array(wdistances)))
    print('Min',np.min(np.array(wdistances)))
    print('Mean',np.mean(np.array(wdistances)))
    print('Mid',np.median(np.array(wdistances)))
    print('------------------------------------')
    
    recons, wdistances = REgener('adversary', ciffgsmloader[i - 1], netc, decoderc)
    np.save('./dataset/CIFAR10/fgsm/fgsm'+str(i)+'re.npy', np.array(recons))
    np.save('./distance/l2/CIFAR10/fgsm/fgsm'+str(i)+'re.npy', np.array(wdistances))
    print('CIFAR FGSM', i)
    print('Max',np.max(np.array(wdistances)))
    print('Min',np.min(np.array(wdistances)))
    print('Mean',np.mean(np.array(wdistances)))
    print('Mid',np.median(np.array(wdistances)))
    print('------------------------------------')
    '''
    
    recons, wdistances = REgener('adversary', cifpgdloader[i - 1], netc, decoderc)
    np.save('./dataset/CIFAR10/pgd/pgd'+str(i)+'re.npy', np.array(recons))
    np.save('./distance/l2/CIFAR10/pgd/pgd'+str(i)+'re.npy', np.array(wdistances))
    print('CIFAR PGD', i)
    print('Max',np.max(np.array(wdistances)))
    print('Min',np.min(np.array(wdistances)))
    print('Mean',np.mean(np.array(wdistances)))
    print('Mid',np.median(np.array(wdistances)))
    print('------------------------------------')
    i += 1




