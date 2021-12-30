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

import torchvision.transforms as transforms

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
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import argparse


parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--model', type=int, default = 50)
parser.add_argument('--pre', type=bool, default = True)
parser.add_argument('--frez', type=bool, default = False)
parser.add_argument('--batch', type=int, default = 32)
parser.add_argument('--epochs', type=int, default = 10)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--decay', type=float, default = 0.001)
parser.add_argument('--Loss', type=bool, default = False)
args = parser.parse_args()
print(args)


train_datasets = dataloader('/home/austin/nctu_hw/DL/DL_hw4','/home/austin/Downloads/eyes_data')
train_dataloader = DataLoader(train_datasets, batch_size = args.batch, shuffle = True, num_workers = 4)
test_datasets = dataloader('/home/austin/nctu_hw/DL/DL_hw4','/home/austin/Downloads/eyes_data','test')
test_dataloader = DataLoader(test_datasets, batch_size = 1, shuffle = True)

net = resnet(args.model, args.pre, args.frez)
net = net.cuda()


if args.Loss:
    weights = [1.0, 11, 5, 30, 36]
    class_weights = torch.FloatTensor(weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
else:
    criterion = nn.CrossEntropyLoss().cuda()

optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=0.9,weight_decay=args.decay)

def Acc(data,net,epoch,mode='test'):
    list_pred = []
    list_label = []
    total = 0
    correct = 0
    net.eval()
    for i_batch, sampled_batched in enumerate(data):
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
    
    print(len(list_pred))
    print(len(list_label))
    t = int(correct.cpu())
    total = float(total)
    acc = 100.0*(t / total)
    print("Accuracy : {:.2f} %".format(acc))
    if mode == 'test':
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
        plt.savefig('/home/austin/nctu_hw/DL/DL_hw4/try50_3_2/confusion_'+str(epoch+1)+'.png',dpi=300)
    net.train()

    return acc

for epoch in range(args.epochs):
    train_loss = []
    for i_batch, sampled_batched in enumerate(train_dataloader):
        print('Epoch: ',epoch+1 ,'/',args.epochs, ' Progress rate:' ,round(100*(i_batch/(len(train_datasets)//args.batch)),2),end='\r')
        inputs = sampled_batched['Image'].cuda()
        labels = sampled_batched['label'].cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    print('Test acc ..')
    test_acc = Acc(test_dataloader,net,epoch)
    print('Train acc ..')
    train_acc = Acc(train_dataloader,net,epoch,'train')

    file = open('/home/austin/nctu_hw/DL/DL_hw4/try50_3_2/loss_3_2.txt',"a+")
    file.write(str(sum(train_loss)/len(train_loss))+'\n')
    file = open('/home/austin/nctu_hw/DL/DL_hw4/try50_3_2/acc_3_2.txt',"a+")
    file.write(str(train_acc)+'/'+str(test_acc)+'\n')

    torch.save(net.state_dict(), '/home/austin/nctu_hw/DL/DL_hw4/try50_3_2/res50_3_2_'+str(epoch+1)+'.pth')

    print("Epoch: {:03d}/{:03d}, Training: Loss: {:.4f}".format(
            epoch+1, args.epochs, sum(train_loss)/len(train_loss)))
    print('\n'+'============================'+'\n')



