#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:14:14 2020

@author: hongxing
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import torchvision as tv
import torchvision.transforms as transforms

import scipy.misc
import imageio
from PIL import Image
import matplotlib.pyplot as plt

from cleverhans.attacks import FastGradientMethod, DeepFool, CarliniWagnerL2, ProjectedGradientDescent
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

from Architectures import CIFAR10Net_ori, MNISTNet_ori, CIFAR10Net, CIFAR10Decoder, MNISTNet, MNISTDecoder, CIFDtcAnom, MSTDtcAnom
from mydataloader import MyDataset, GrayDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def singleFGSM(torch_model, xs, ys, eps, c, h, w, clip_min, clip_max):
    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, c, h, w,))
    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(torch_model)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
    
    # Create an FGSM attack
    atk_op = FastGradientMethod(cleverhans_model, sess=sess)
    atk_params = {'eps': eps,
                   'clip_min': clip_min,
                   'clip_max': clip_max}
    adv_x_op = atk_op.generate(x_op, **atk_params)
    
    # Run an evaluation of our model against fgsm
    xs, ys = xs.to(device), ys.to(device)
    adv = torch.from_numpy(sess.run(adv_x_op, feed_dict={x_op: xs}))
    pred = np.argmax(torch_model(adv).data.cpu().numpy())
    if ys != pred:
        return adv
    else:
        return []
    
def singlePGD(torch_model, xs, ys, eps, c, h, w, clip_min, clip_max):
    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, c, h, w,))
    # Convert pytorch model to a tf_model and wrap it in cleverhans
    tf_model_fn = convert_pytorch_model_to_tf(torch_model)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
    
    # Create an FGSM attack
    atk_op = ProjectedGradientDescent(cleverhans_model, sess=sess)
    atk_params = {'eps': eps,
                   'clip_min': clip_min,
                   'clip_max': clip_max}
    adv_x_op = atk_op.generate(x_op, **atk_params)
    
    # Run an evaluation of our model against fgsm
    xs, ys = xs.to(device), ys.to(device)
    adv = torch.from_numpy(sess.run(adv_x_op, feed_dict={x_op: xs}))
    pred = np.argmax(torch_model(adv).data.cpu().numpy())
    if ys != pred:
        return adv
    else:
        return []
    
def findADVt(net, dataset, labellist, eps, c, h, w, clip_min, clip_max):
    orilist = []
    advlist = []
    for label in labellist:
        for data in dataset:
            xs, ys = data
            if ys.numpy() == label:
                xs, ys = xs.to(device), ys.to(device)
                if ys == np.argmax(net(xs).data.cpu().numpy()):
                    adv = singlePGD(net, xs, ys, eps, c, h, w, clip_min, clip_max)
                    if len(adv):
                        orilist.append(xs.numpy())
                        advlist.append(adv.numpy())
                        break
                    else:
                        continue
                else:
                    continue
            else:
                continue
    return orilist, advlist



EVALSIZE = 50
GENESIZE = 1

transform1 = transforms.ToTensor()

transform2 = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


mnisttestset = tv.datasets.MNIST(
    root='./mnist/',
    train=False,
    download=True,
    transform=transform1)
mnistevalloader = torch.utils.data.DataLoader(
    mnisttestset,
    batch_size=EVALSIZE,
    shuffle=True,
    )
mnistgeneloader = torch.utils.data.DataLoader(
    mnisttestset,
    batch_size=GENESIZE,
    shuffle=True,
    )


cifartestset = tv.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform2)
cifarevalloader = torch.utils.data.DataLoader(cifartestset, batch_size=EVALSIZE,
                                         shuffle=True, num_workers=2)
cifargeneloader = torch.utils.data.DataLoader(cifartestset, batch_size=GENESIZE,
                                         shuffle=True, num_workers=2)

tnet = MNISTNet_ori().to(device)
tnet.load_state_dict(torch.load('./model/MNIST/Tclassifier.pth', map_location=torch.device('cpu')))
tnet.eval()

net = MNISTNet().to(device)
net.load_state_dict(torch.load('./model/MNIST/Tclassifier.pth', map_location=torch.device('cpu')))
net.eval()

decoder = MNISTDecoder().to(device)
decoder.load_state_dict(torch.load('./model/MNIST/Decoder.pth', map_location=torch.device('cpu')))
decoder.eval()

netc = CIFAR10Net().to(device)
netc.load_state_dict(torch.load('./model/CIFAR10/Tclassifier.pth', map_location=torch.device('cpu')))
netc.eval()

decoderc = CIFAR10Decoder().to(device)
decoderc.load_state_dict(torch.load('./model/CIFAR10/Decoder.pth', map_location=torch.device('cpu')))
decoderc.eval()


orilist, advlist = findADVt(tnet, mnistgeneloader, [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]], 0.3, 1, 28, 28, 0, 1)
np.save('./examples/MNIST/ori.npy', orilist)
np.save('./examples/MNIST/adv.npy', advlist)

orilist, advlist = np.load('./examples/MNIST/ori.npy'), np.load('./examples/MNIST/adv.npy')

oriftlist = []
advftlist = []
for ori in orilist:
    ori = torch.from_numpy(ori).to(device)
    encoded, indices1, indices2, outputs = net(ori)
    oriftlist.append(encoded.detach().numpy())
    
for adv in advlist:
    adv = torch.from_numpy(adv).to(device)
    encoded, indices1, indices2, outputs = net(adv)
    advftlist.append(encoded.detach().numpy())
    
np.save('./examples/MNIST/orift.npy', oriftlist)
np.save('./examples/MNIST/advft.npy', advftlist)

orireclist = []
advreclist = []
for ori in orilist:
    ori = torch.from_numpy(ori).to(device)
    encoded, indices1, indices2, outputs = net(ori)
    decoded = decoder(encoded, indices1, indices2)
    orireclist.append(decoded.detach().numpy())
    
for adv in advlist:
    adv = torch.from_numpy(adv).to(device)
    encoded, indices1, indices2, outputs = net(adv)
    decoded = decoder(encoded, indices1, indices2)
    advreclist.append(decoded.detach().numpy())
    
np.save('./examples/MNIST/orirec.npy', orireclist)
np.save('./examples/MNIST/advrec.npy', advreclist)






    
