import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import random
import copy
from layers import SinkhornDistance
from torch.autograd.gradcheck import zero_gradients


#最大迭代次数
epochs=100
#used as a termination criterion to prevent vanishing updates
overshoot=0.02
#类别数
num_classes=10

def deepfool(epochs, overshoot, num_classes, xs, ys, torch_model):
	output = torch_model(xs)
	label1 = np.argmax(output.data.cpu().numpy())
	predicted = torch.max(output.data,1)[1] #outputs含有梯度值，其处理方式与之前有所不同
	if ys==label1:
		prob = F.softmax(output)[0][label1.item()]*100
		_, predicted = torch.max(output.data, 1)
		_,output = torch_model(img)
		input_shape = xs.cpu().detach().numpy().shape
		w = np.zeros(input_shape)
		r_tot = np.zeros(input_shape)
		
		for epoch in range(epochs):
			output = torch_model(xs)
			scores=output.data.cpu().numpy()[0]
			label=np.argmax(scores)
			if label != label1:
				break
			pert = np.inf
			output[0, label1].backward(retain_graph=True)
			grad_orig = img.grad.data.cpu().numpy().copy()
			for k in range(1, num_classes):
				if k == label:
					continue
				zero_gradients(xs)
				output[0, k].backward(retain_graph=True)
				cur_grad = xs.grad.data.cpu().numpy().copy()
				w_k = cur_grad - grad_orig
				f_k = (output[0, k] - output[0, label1]).data.cpu().numpy()
				pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
				if pert_k < pert:
					pert = pert_k
					w = w_k
			r_i =  (pert+1e-8) * w / np.linalg.norm(w)
			r_tot = np.float32(r_tot + r_i)
			xs.data=xs.data + (1+overshoot)*torch.from_numpy(r_tot).to(device)
		if epoch < 200:
			return xs
		else:
			return 1
	else:
		return 0


def DF(torch_model, dataset, eps_list, opt, c, h, w, clip_min, clip_max):
	if opt == 'evaluate':
		acclist = []
		for eps in eps_list:
			total = 0
            correct = 0
			for xs, ys in dataset:
				total += dataset.batch_size
				xs, ys = xs.to(device), ys.to(device)
				adv = deepfool(100, 0.02, 10, xs, ys, torch_model)





