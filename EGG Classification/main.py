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
parser.add_argument('--model', type=str, default = 'EEG')
parser.add_argument('--act', type=str, default = 'LeakyReLU')
parser.add_argument('--batch', type=int, default = 120)
parser.add_argument('--epochs', type=int, default = 100)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--shuffle', type=bool, default = False)
args = parser.parse_args()

print(args)

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

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

train_data, train_label, test_data, test_label = read_bci_data()



if args.act == "ELU":
    if args.model == "EEG":
        net = EEG_ELU().cuda()
    else:
        net = Deep_ELU().cuda()
if args.act == "ReLU":
    if args.model == "EEG":
        net = EEG_ReLU().cuda()
    else:
        net = Deep_ReLU().cuda()
if args.act == "LeakyReLU":
    if args.model == "EEG":
        net = EEG_LeakyReLU().cuda()
    else:
        net = Deep_LeakyReLU().cuda()

print(net)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

def Acc(data1,data2):
    list_pred = []
    list_label = []
    correct = 0
    total = 0
    net.eval()
    for i in range(1080):
            test = np.expand_dims(data1[i], axis=0)
            inputs_test = Variable(torch.from_numpy(test)).cuda()
            inputs_test = inputs_test.to(dtype=torch.float)

            labels_test = Variable(torch.from_numpy(data2.reshape(1080,1)[i])).cuda()
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
    net.train()

    return float((100.0*(t / total)))

batch = args.batch
epochs = args.epochs
run = []
loss_train = []
acc_train = []
acc_valid = []
history = []

for epoch in range(epochs):
    epoch_start = time.time()
    if args.shuffle:
        permutation = np.random.permutation(train_data.shape[0])
        train_data = train_data[permutation]
        train_label = train_label[permutation]
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0

    # training
    for i in range(int(1080/batch)):
        inputs = Variable(torch.from_numpy(train_data[i*batch:(i+1)*batch])).cuda()
        inputs = inputs.to(dtype=torch.float)

        labels = Variable(torch.from_numpy(train_label[i*batch:(i+1)*batch])).cuda()
        labels = labels.to(dtype=torch.long)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
    

    # calculate acc
    avg_train_acc = Acc(train_data,train_label)
    avg_valid_acc = Acc(test_data,test_label)
    acc_train.append(avg_train_acc)
    acc_valid.append(avg_valid_acc)

    # calculate loss
    avg_train_loss = train_loss/1080
    loss_train.append(avg_train_loss)
    
    history.append([avg_train_loss, avg_train_acc, avg_valid_acc])
    
    epoch_end = time.time()
    run.append(epoch+1)

    print("Epoch: {:03d}/{:03d}, Training: Loss: {:.4f}, Train_Accuracy: {:.4f}%, Test_Accuracy: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, epochs, avg_train_loss, avg_train_acc, avg_valid_acc, epoch_end-epoch_start
        ))
    # if avg_valid_acc >= 88.0:
    #     print('Save !!')
    #     torch.save(net.state_dict(), '/home/austin/nctu_hw/DL/DL_hw3/weight/87_'+str(epoch+1)+'_'+args.model+'_'+str(args.batch)+'_'+str(args.epochs)+'_'+str(args.lr)+'_'+args.act+'.pth')

torch.save(net.state_dict(), '/home/austin/nctu_hw/DL/DL_hw3/weight/'+args.model+'_'+str(args.batch)+'_'+str(args.epochs)+'_'+str(args.lr)+'shuffle'+'_'+args.act+'.pth')
file = open('/home/austin/nctu_hw/DL/DL_hw3/loss_record/Deep/Train_loss_'+args.model+'shuffle_'+str(args.batch)+'_'+str(args.epochs)+'_'+str(args.lr)+'_'+args.act+'.txt', "a+")
for Loss in loss_train:
    file.write(str(Loss)+'\n')
file.close()

file = open('/home/austin/nctu_hw/DL/DL_hw3/acc_record/Deep/Train_acc_'+args.model+'shuffle_'+str(args.batch)+'_'+str(args.epochs)+'_'+str(args.lr)+'_'+args.act+'.txt', "a+")
for acc in acc_train:
    file.write(str(acc)+'\n')
file.close()

file = open('/home/austin/nctu_hw/DL/DL_hw3/acc_record/Deep/Test_acc_'+args.model+'shuffle_'+str(args.batch)+'_'+str(args.epochs)+'_'+str(args.lr)+'_'+args.act+'.txt', "a+")
for acc in acc_valid:
    file.write(str(acc)+'\n')
file.close()