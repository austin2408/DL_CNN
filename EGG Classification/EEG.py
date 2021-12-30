import torch
import time
import torch.nn as nn
from torchvision import datasets ,transforms
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import os

class EEG_ELU(nn.Module):
	def __init__(self):
		super(EEG_ELU, self).__init__()
		self.firstconv = nn.Sequential(              
			nn.Conv2d(
				in_channels=1,              
				out_channels=16,            
				kernel_size=(1,51),              
				stride=(1,1),                   
				padding=(0,25),
				bias=False,                  
			),
			nn.BatchNorm2d(
				16,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			)                                                     
		)

		self.depthwiseConv = nn.Sequential(         
			nn.Conv2d(
				in_channels=16,
				out_channels=32,
				kernel_size=(2,1),
				stride=(1,1),
				groups=16,
				bias=False,
			),
			nn.BatchNorm2d(
				32,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			),
			nn.ELU(alpha=1.0),
			nn.AvgPool2d(
			   kernel_size=(1,4),
			   stride=(1,4),
			   padding=0, 
			),
			nn.Dropout(p=0.25),                                           
		)

		self.separableConv = nn.Sequential(         
			nn.Conv2d(
				in_channels=32,
				out_channels=32,
				kernel_size=(1,15),
				stride=(1,1),
				padding=(0,7),
				bias=False,
			),
			nn.BatchNorm2d(
				32,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			),
			nn.ELU(alpha=1.0),
			nn.AvgPool2d(
			   kernel_size=(1,8),
			   stride=(1,8),
			   padding=0,
			),
			nn.Dropout(p=0.25),                                          
		)
		self.classify = nn.Linear(736,2,bias=True)

	def forward(self, x):
		x = self.firstconv(x)
		x = self.depthwiseConv(x)
		x = self.separableConv(x)
		x = x.view(x.size(0), -1)
		x = self.classify(x)
		return x

class EEG_ReLU(nn.Module):
	def __init__(self):
		super(EEG_ReLU, self).__init__()
		self.firstconv = nn.Sequential(              
			nn.Conv2d(
				in_channels=1,              
				out_channels=16,            
				kernel_size=(1,51),              
				stride=(1,1),                   
				padding=(0,25),
				bias=False,                  
			),
			nn.BatchNorm2d(
				16,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			)                                                     
		)

		self.depthwiseConv = nn.Sequential(         
			nn.Conv2d(
				in_channels=16,
				out_channels=32,
				kernel_size=(2,1),
				stride=(1,1),
				groups=16,
				bias=False,
			),
			nn.BatchNorm2d(
				32,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			),
			nn.ReLU(),
			nn.AvgPool2d(
			   kernel_size=(1,4),
			   stride=(1,4),
			   padding=0, 
			),
			nn.Dropout(p=0.25),                                           
		)

		self.separableConv = nn.Sequential(         
			nn.Conv2d(
				in_channels=32,
				out_channels=32,
				kernel_size=(1,15),
				stride=(1,1),
				padding=(0,7),
				bias=False,
			),
			nn.BatchNorm2d(
				32,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			),
			nn.ReLU(),
			nn.AvgPool2d(
			   kernel_size=(1,8),
			   stride=(1,8),
			   padding=0,
			),
			nn.Dropout(p=0.25),                                          
		)
		self.classify = nn.Linear(736,2,bias=True)
	

	def forward(self, x):
		x = self.firstconv(x)
		x = self.depthwiseConv(x)
		x = self.separableConv(x)
		x = x.view(x.size(0), -1)
		x = self.classify(x)
		return x

class EEG_LeakyReLU(nn.Module):
	def __init__(self):
		super(EEG_LeakyReLU, self).__init__()
		self.firstconv = nn.Sequential(              
			nn.Conv2d(
				in_channels=1,              
				out_channels=16,            
				kernel_size=(1,51),              
				stride=(1,1),                   
				padding=(0,25),
				bias=False,                  
			),
			nn.BatchNorm2d(
			16,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			)                                                     
		)

		self.depthwiseConv = nn.Sequential(         
			nn.Conv2d(
				in_channels=16,
				out_channels=32,
				kernel_size=(2,1),
				stride=(1,1),
				groups=16,
				bias=False,
			),
			nn.BatchNorm2d(
				32,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			),
			nn.LeakyReLU(),
			nn.AvgPool2d(
			kernel_size=(1,4),
			stride=(1,4),
			padding=0, 
			),
			nn.Dropout(p=0.25),                                           
		)

		self.separableConv = nn.Sequential(         
			nn.Conv2d(
				in_channels=32,
				out_channels=32,
				kernel_size=(1,15),
				stride=(1,1),
				padding=(0,7),
				bias=False,
			),
			nn.BatchNorm2d(
				32,
				eps=1e-05,
				momentum=0.1,
				affine=True,
				track_running_stats=True,
			),
			nn.LeakyReLU(),
			nn.AvgPool2d(
			kernel_size=(1,8),
			stride=(1,8),
			padding=0,
			),
			nn.Dropout(p=0.25),                                          
		)
		self.classify = nn.Linear(736,2,bias=True)
	

	def forward(self, x):
		x = self.firstconv(x)
		x = self.depthwiseConv(x)
		x = self.separableConv(x)
		x = x.view(x.size(0), -1)
		x = self.classify(x)
		return x