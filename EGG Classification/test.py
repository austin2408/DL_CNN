import torch
import time
import torch.nn as nn
from torchvision import datasets ,transforms
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import os
from EEG import *
from DeepConv import *
import argparse

parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--weight', type=str, default = None)
args = parser.parse_args()

def read_bci_data():
    S4b_train = np.load('/home/austin/nctu_hw/DL/DL_hw3/datasets/S4b_test.npz')
    X11b_train = np.load('/home/austin/nctu_hw/DL/DL_hw3/datasets/X11b_train.npz')
    S4b_test = np.load('/home/austin/nctu_hw/DL/DL_hw3/datasets/S4b_test.npz')
    X11b_test = np.load('/home/austin/nctu_hw/DL/DL_hw3/datasets/X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    # print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

train_data, train_label, test_data, test_label = read_bci_data()
name = args.weight.split('_')

if name[2].split('/')[2]=='Epoch':
    model = name[4]
    act = name[8].split('.')[0]
else:
    model = name[2].split('/')[2]
    act = name[6].split('.')[0]

if act == "ELU":
    if model == "EEG":
        net = EEG_ELU().cuda()
    else:
        net = Deep_ELU().cuda()
if act == "ReLU":
    if model == "EEG":
        net = EEG_ReLU().cuda()
    else:
        net = Deep_ReLU().cuda()
if act == "LeakyReLU":
    if model == "EEG":
        net = EEG_LeakyReLU().cuda()
    else:
        net = Deep_LeakyReLU().cuda()

net.load_state_dict(torch.load(args.weight))

print(net)

def Acc(data1,data2):
    list_pred = []
    list_label = []
    correct = 0
    total = 0
    net.eval()
    inputs_test = Variable(torch.from_numpy(data1)).cuda()
    inputs_test = inputs_test.to(dtype=torch.float)

    labels_test = Variable(torch.from_numpy(data2)).cuda()
    labels_test = labels_test.to(dtype=torch.long)
    with torch.no_grad():
        outputs = net(inputs_test)
        _,predicted = torch.max(outputs.data,1)
        list_pred.append(predicted.cpu().numpy().tolist()[0])
        list_label.append(labels_test.cpu().numpy().tolist()[0])

    total += labels_test.size(0)
    correct += (predicted == labels_test).sum()
    t = int(correct.cpu())
    total = float(total)
    print('Accuracy : ',format(float((100.0*(t / total))),'.2f'),'%')

Acc(test_data, test_label)