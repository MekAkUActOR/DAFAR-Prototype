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


def REgener(inputs, reAE):
    inputs = inputs.to(device)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    decoded, outputs = reAE(inputs)
    substract = (inputs - decoded).cpu().detach().numpy().squeeze(0)
    return outputs, substract

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

    
sample = args.input #"./6a.jpg"
threshold = args.threshold #23.333

inputs = torch.from_numpy(np.array(imageio.imread(sample)).astype(float).reshape(1,1,28,28)/255).to(device)
outputs, substract = REgener(inputs, reAE)
output=np.argmax(outputs.data.cpu().numpy())
score = AnomScore(substract, detector)
if score > threshold:
    print("\033[37;41mATTACK!\033[0m")
else:
    print(output)



