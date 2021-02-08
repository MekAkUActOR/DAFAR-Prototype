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

from Architectures import MSTreAE, MSTDtcAnom
from mydataloader import MyDataset, GrayDataset


parser = argparse.ArgumentParser()
parser.description='configuration'
parser.add_argument("-i", "--input", help="path of input picture", required=True)
parser.add_argument("-t", "--threshold", help="anomaly score threshold", type=float, required=True)
#parser.add_argument("-m", "--model", help="path of model parameter", required=True)
#parser.add_argument("-n", "--network", help="path of network file", required=True)
args = parser.parse_args()

print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# generate the restruction error of a sample
def REgener(inputs, reAE):
    inputs = inputs.to(device)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    decoded, outputs = reAE(inputs)
    substract = (inputs - decoded).cpu().detach().numpy().squeeze(0)
    return outputs, substract

# generate the restruction error set of a train set
def ReSet(dataset, reAE):
    recons = []
    for data in dataset:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        decoded, outputs = reAE(inputs)
        substract = (inputs - decoded).cpu().detach().numpy().squeeze(0)
        recons.append(substract)
    return recons

# calculate the anomly score of a sample
def AnomScore(inputs, detector):
    inputs = inputs.to(device)
    outputs = detector(inputs)
    substract = (inputs - outputs).cpu().detach().numpy().squeeze(0)
    l2 = 0
    for channel in substract:
        l2 += np.linalg.norm(channel)
    return l2

# generate the anomly score set of a train set
def AnomScoreSet(dataset, detector):
    wdistances = []
    for data in dataset:
        inputs = data.reshape((1, -1))
        inputs = inputs.to(device)
        wdistances.append(AnomScore(inputs, detector))
    wdistances = np.array(wdistances)
    return wdistances

# get the threshold
def setThres(arry):
    var = np.std(arry)
    thres = np.mean(arry) + 2*var
    return thres


# load parameters of models
reAE = MSTreAE().to(device)
model_dict = reAE.state_dict()
if device == "cpu":
    pretrained_dict = torch.load('./model/MNIST/Tclassifier.pth', map_location=torch.device('cpu'))
else:
    pretrained_dict = torch.load('./model/MNIST/Tclassifier.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
reAE.load_state_dict(model_dict)
if device == "cpu":
    pretrained_dict = torch.load('./model/MNIST/Decoder.pth', map_location=torch.device('cpu'))
else:
    pretrained_dict = torch.load('./model/MNIST/Decoder.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
reAE.load_state_dict(model_dict)
reAE.eval()

detector = MSTDtcAnom().to(device)
if device == "cpu":
    detector.load_state_dict(torch.load('./model/DETECTOR/MSTDtcAnomL2.pth', map_location=torch.device('cpu')))
else:
    detector.load_state_dict(torch.load('./model/DETECTOR/MSTDtcAnomL2.pth'))
detector.eval()


# load dataset
transform1 = transforms.ToTensor()

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


# main program
recons = ReSet(msttrainloader,reAE)
wdistances = AnomScoreSet(recons, detector)
threshold = setThres(wdistances)
print(threshold)


