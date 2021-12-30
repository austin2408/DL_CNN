import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from torch.utils import model_zoo
import random
import sys
import math
from sklearn.metrics import confusion_matrix

import torchvision.transforms as transforms
from matplotlib.font_manager import FontProperties

import cv2
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from matplotlib import pyplot as plt
import numpy as np
import time
import os
from dataLoader import *
from model import *
import argparse


parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--weight', type=str)
parser.add_argument('--model', type=int, default = 50)
args = parser.parse_args()

test_datasets = dataloader('/home/austin/nctu_hw/DL/DL_hw4','/home/austin/Downloads/eyes_data','test')
test_dataloader = DataLoader(test_datasets, batch_size = 1, shuffle = True, num_workers = 4)

net = resnet(args.model,pre=False)
net = net.cuda()
net.eval()
net.load_state_dict(torch.load(args.weight))

list_pred = []
list_label = []
total = 0
correct = 0
for i_batch, sampled_batched in enumerate(test_dataloader):
    print(i_batch,end='\r')
    inputs = sampled_batched['Image'].cuda()
    labels = sampled_batched['label'].cuda()
    with torch.no_grad():
        outputs = net(inputs)
        _,predicted = torch.max(outputs.data,1)
        list_pred.append(predicted.cpu().numpy().tolist()[0])
        list_label.append(labels.cpu().numpy().tolist()[0])
        
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
t = int(correct.cpu())
total = float(total)
acc = 100.0*(t / total)
print("Accuracy : {:.2f} %".format(acc))
maxtrix = confusion_matrix(list_label, list_pred)
plt.matshow(maxtrix, cmap=plt.cm.Blues, alpha=0.7)
for i in range(maxtrix.shape[0]):
    for j in range(maxtrix.shape[1]):
        g = round(float(maxtrix[i,j]/np.sum(maxtrix[i][:])),2)
        plt.text(x=j, y=i, s = g, va='center', ha='center')
    # plt.colorbar()
plt.title('Confusion Matrix'+'\n')
plt.xlabel('pred')
plt.ylabel('true')
plt.xticks(np.arange(maxtrix.shape[1]),['0','1','2','3','4'])
plt.yticks(np.arange(maxtrix.shape[1]),['0','1','2','3','4'])
# plt.savefig('/home/austin/nctu_hw/DL/DL_hw4/try50_3/confusion_'+str(11)+'.png',dpi=300)




