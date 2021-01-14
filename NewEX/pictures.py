#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 16:19:42 2020

@author: hongxing
"""

import numpy as np
import scipy.misc
import imageio
from PIL import Image
import matplotlib.pyplot as plt

#orilist, advlist, oriftlist, advftlist, orireclist, advreclist = np.load('./examples/MNIST/ori.npy'), np.load('./examples/MNIST/adv.npy'), np.load('./examples/MNIST/orift.npy'), np.load('./examples/MNIST/advft.npy'), np.load('./examples/MNIST/orirec.npy'), np.load('./examples/MNIST/advrec.npy')

'''
i = 0
while i <= 9:
    ori = orilist[i].reshape(28,28)
    plt.figure()
    plt.imshow(ori, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/ori'+str(i)+'.jpg')
    plt.close()
    
    orirec = orireclist[i].reshape(28,28)
    plt.figure()
    plt.imshow(orirec, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/orirec'+str(i)+'.jpg')
    plt.close()
    
    adv = advlist[i].reshape(28,28)
    plt.figure()
    plt.imshow(adv, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/adv'+str(i)+'.jpg')
    plt.close()
    
    advrec = advreclist[i].reshape(28,28)
    plt.figure()
    plt.imshow(advrec, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/advrec'+str(i)+'.jpg')
    plt.close()
    
    orift = oriftlist[i].reshape(56,56)
    plt.figure()
    plt.imshow(orift, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/orift'+str(i)+'.jpg')
    plt.close()
    
    advft = advftlist[i].reshape(56,56)
    plt.figure()
    plt.imshow(advft, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/advft'+str(i)+'.jpg')
    plt.close()
    
    i += 1
''' 
'''
plt.figure()
plt.subplots_adjust(left=0.04, top= 0.96, right = 0.96, bottom = 0.04, wspace = 0.01, hspace = 0.01)
for i in range(1, 11):
    plt.subplot(6, 10, i)
    plt.imshow(orilist[i - 1].reshape(28,28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
for i in range(1, 11):
    plt.subplot(6, 10, i+10)
    plt.imshow(orireclist[i - 1].reshape(28,28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
for i in range(1, 11):
    plt.subplot(6, 10, i+20)
    plt.imshow(abs((orireclist[i - 1]-orilist[i - 1]).reshape(28,28)), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
for i in range(1, 11):
    plt.subplot(6, 10, i+30)
    plt.imshow(advlist[i - 1].reshape(28,28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
for i in range(1, 11):
    plt.subplot(6, 10, i+40)
    plt.imshow(advreclist[i - 1].reshape(28,28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
for i in range(1, 11):
    plt.subplot(6, 10, i+50)
    plt.imshow(abs((advreclist[i - 1]-advlist[i - 1]).reshape(28,28)), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
plt.savefig('./figures/whole.eps')
plt.close()
'''


    
    
    
    
    
    
    
    