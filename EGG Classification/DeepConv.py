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

class Deep_ReLU(nn.Module):
    def __init__(self):
        super(Deep_ReLU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1,5),
                stride=(1,1),
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(2,1),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                25,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(1,2),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                50,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=50,
                out_channels=100,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                100,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=100,
                out_channels=200,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                200,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.classify = nn.Linear(146000,2,bias=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classify(x)
        return x

class Deep_ELU(nn.Module):
    def __init__(self):
        super(Deep_ELU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1,5),
                stride=(1,1),
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(2,1),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                25,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(
                kernel_size=(1,2),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                50,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=50,
                out_channels=100,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                100,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=100,
                out_channels=200,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                200,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.classify = nn.Linear(146000,2,bias=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        # x = torch.sigmoid(x)
        return x

class Deep_LeakyReLU(nn.Module):
    def __init__(self):
        super(Deep_LeakyReLU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1,5),
                stride=(1,1),
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(2,1),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                25,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=(1,2),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                50,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=50,
                out_channels=100,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                100,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=100,
                out_channels=200,
                kernel_size=(1,2),
                stride=(1,1),
            ),
            nn.BatchNorm2d(
                200,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=(1,5),
                stride=(2,1),
            ),
            nn.Dropout(p=0.5),
        )

        self.classify = nn.Linear(146000,2,bias=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x


