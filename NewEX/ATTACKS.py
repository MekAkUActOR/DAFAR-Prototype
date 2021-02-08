#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 01:00:26 2020

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

from Architectures import CIFAR10Net_ori, MNISTNet_ori

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
parameters:
torch_model --- the target model
dataset --- original dataset
eps_list --- the list of attack intencities
opt --- 'generate' adversarial examples or just 'evaluate' attack success rate
clip_min/clip_max --- the minimum/maxmum of pixel values
'''
def FGSM(torch_model, dataset, eps_list, opt, c, h, w, clip_min, clip_max):

    if opt == 'evaluate':
        acclist = []
        for eps in eps_list:
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
            adv_preds_op = tf_model_fn(adv_x_op)
            
            # Run an evaluation of our model against fgsm
            total = 0
            correct = 0
            for xs, ys in dataset:
                xs, ys = xs.to(device), ys.to(device)
                adv_preds = sess.run(adv_preds_op, feed_dict={x_op: xs})
                correct += (np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy()).sum()
                total += dataset.batch_size
            
            acc = float(correct) / total
            print('Adv accuracy: {:.3f}'.format(acc * 100))
            acclist.append(acc)
        return acclist
        
    elif opt == 'generate':
        advpacklist = []
        for eps in eps_list:
            advlist = []
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
            for xs, ys in dataset:
                xs, ys = xs.to(device), ys.to(device)
                adv = torch.from_numpy(sess.run(adv_x_op, feed_dict={x_op: xs}))
                if ys == np.argmax(torch_model(xs).data.cpu().numpy()):
                    pred = np.argmax(torch_model(adv).data.cpu().numpy())
                    if ys != pred:
                        adv = adv.numpy()
                        advlist.append(adv)
            print(len(advlist))
            advpacklist.append(advlist)
        return advpacklist

def CW2(torch_model, dataset, eps_list, opt, c, h, w, clip_min, clip_max):

    if opt == 'evaluate':
        acclist = []
        for eps in eps_list:
            sess = tf.Session()
            x_op = tf.placeholder(tf.float32, shape=(None, c, h, w,))
            # Convert pytorch model to a tf_model and wrap it in cleverhans
            tf_model_fn = convert_pytorch_model_to_tf(torch_model)
            cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
            
            # Create an FGSM attack
            atk_op = CarliniWagnerL2(cleverhans_model, sess=sess)
            atk_params = {
                           'clip_min': clip_min,
                           'clip_max': clip_max}
            adv_x_op = atk_op.generate(x_op, **atk_params)
            adv_preds_op = tf_model_fn(adv_x_op)
            
            # Run an evaluation of our model against fgsm
            total = 0
            correct = 0
            for xs, ys in dataset:
                xs, ys = xs.to(device), ys.to(device)
                adv_preds = sess.run(adv_preds_op, feed_dict={x_op: xs})
                correct += (np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy()).sum()
                total += dataset.batch_size
            
            acc = float(correct) / total
            print('Adv accuracy: {:.3f}'.format(acc * 100))
            acclist.append(acc)
        return acclist
        
    elif opt == 'generate':
        advpacklist = []
        for eps in eps_list:
            advlist = []
            sess = tf.Session()
            x_op = tf.placeholder(tf.float32, shape=(None, c, h, w,))
            # Convert pytorch model to a tf_model and wrap it in cleverhans
            tf_model_fn = convert_pytorch_model_to_tf(torch_model)
            cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
            
            # Create an FGSM attack
            atk_op = CarliniWagnerL2(cleverhans_model, sess=sess)
            atk_params = {
                           'clip_min': clip_min,
                           'clip_max': clip_max}
            adv_x_op = atk_op.generate(x_op, **atk_params)
            
            total = 0
            # Run an evaluation of our model against fgsm
            for xs, ys in dataset:
                xs, ys = xs.to(device), ys.to(device)
                adv = torch.from_numpy(sess.run(adv_x_op, feed_dict={x_op: xs}))
                if ys == np.argmax(torch_model(xs).data.cpu().numpy()):
                    pred = np.argmax(torch_model(adv).data.cpu().numpy())
                    if ys != pred:
                        print('OK')
                        total += 1
                        print(total)
                        adv = adv.numpy()
                        advlist.append(adv)
                if total == 500:
                    break
            print(len(advlist))
            advpacklist.append(advlist)
        return advpacklist

def PGD(torch_model, dataset, eps_list, opt, c, h, w, clip_min, clip_max):

    if opt == 'evaluate':
        acclist = []
        for eps in eps_list:
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
            adv_preds_op = tf_model_fn(adv_x_op)
            
            # Run an evaluation of our model against fgsm
            total = 0
            correct = 0
            for xs, ys in dataset:
                xs, ys = xs.to(device), ys.to(device)
                adv_preds = sess.run(adv_preds_op, feed_dict={x_op: xs})
                correct += (np.argmax(adv_preds, axis=1) == ys.cpu().detach().numpy()).sum()
                total += dataset.batch_size
            
            acc = float(correct) / total
            print('Adv accuracy: {:.3f}'.format(acc * 100))
            acclist.append(acc)
        return acclist
        
    elif opt == 'generate':
        advpacklist = []
        for eps in eps_list:
            advlist = []
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
            
            total = 0
            # Run an evaluation of our model against fgsm
            for xs, ys in dataset:
                xs, ys = xs.to(device), ys.to(device)
                adv = torch.from_numpy(sess.run(adv_x_op, feed_dict={x_op: xs}))
                if ys == np.argmax(torch_model(xs).data.cpu().numpy()):
                    pred = np.argmax(torch_model(adv).data.cpu().numpy())
                    if ys != pred:
                        print('OK')
                        total += 1
                        print(total)
                        adv = adv.numpy()
                        advlist.append(adv)
                if total == 500:
                    break
            print(len(advlist))
            advpacklist.append(advlist)
        return advpacklist

# save the dataset in the form of npy
def savedataset(advpacklist, path):
    advset = []

    for advlist in advpacklist:
        advs = []
        for adv in advlist:
            advs.append(adv.squeeze(0))
        advset.append(advs)
    
    advsets = []
    
    for advs in advset:
        advsets.append(np.array(advs))
    
    i = 1
    for advs in advsets:
        np.save(path + str(i)+'.npy', advs)
        i+=1

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
        adv = adv.numpy()
        return adv
    else:
        return []


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


net = MNISTNet_ori().to(device)
net.load_state_dict(torch.load('./model/MNIST/Tclassifier.pth', map_location=device))
net.eval()

netc = CIFAR10Net_ori().to(device)
netc.load_state_dict(torch.load('./model/CIFAR10/Tclassifier.pth', device))
netc.eval()


'''
# generate adversarial example sets of epsilons in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] and save in the file './dataset/CIFAR10/pgd/pgd'
advpacklist = PGD(netc, cifargeneloader, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], 'generate', 3, 32, 32, -1, 1)      
savedataset(advpacklist, './dataset/CIFAR10/pgd/pgd')  
'''
