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

from Architectures import MSTreAE, MSTDtcAnom
from mydataloader import MyDataset, GrayDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generate the restruction error of a sample
def REgener(inputs, reAE):
    inputs = inputs.to(device)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    decoded, outputs = reAE(inputs)
    substract = (inputs - decoded).cpu().detach().numpy().squeeze(0)
    return outputs, substract

# calculate the anomly score of a sample
def AnomScore(inputs, detector):
    inputs = torch.from_numpy(inputs).to(device)
    inputs = inputs.view(inputs.size()[0], -1)
    outputs = detector(inputs)
    substract = (inputs - outputs).cpu().detach().numpy().squeeze(0)
    l2 = 0
    for channel in substract:
        l2 += np.linalg.norm(channel)
    return l2


# load parameters of models
reAE = MSTreAE().to(device)
model_dict = reAE.state_dict()
pretrained_dict = torch.load('./model/MNIST/Tclassifier.pth', map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
reAE.load_state_dict(model_dict)
pretrained_dict = torch.load('./model/MNIST/Decoder.pth', map_location=device)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
reAE.load_state_dict(model_dict)
reAE.eval()

detector = MSTDtcAnom().to(device)
detector.load_state_dict(torch.load('./model/DETECTOR/MSTDtcAnomL2.pth', map_location=device))
detector.eval()

    
setpath = "./dataset/MNIST/fgsm/fgsm1.npy"
threshold = 23.333
settype = "normal"

if settype == "adversarial":
    mstattackset = MyDataset(setpath)
    mstattackloader = torch.utils.data.DataLoader(mstattackset, batch_size=1, shuffle=True, pin_memory=True)
    i = 0
    a = 0
    for data in mstattackloader:
        i += 1
        outputs, substract = REgener(data, reAE)
        score = AnomScore(substract, detector)
        if score >= threshold:
            a += 1
    print(a/i)

elif settype == "normal":
    transform1 = transforms.ToTensor()
    msttestset = tv.datasets.MNIST(root='./mnist/', train=False, download=True, transform=transform1)
    msttestloader = torch.utils.data.DataLoader(msttestset, batch_size=1, shuffle=False)
    i = 0
    a = 0
    for data in msttestloader:
        i += 1
        inputs, label = data
        outputs, substract = REgener(inputs, reAE)
        score = AnomScore(substract, detector)
        if score < threshold:
            a += 1
    print(a/i)



