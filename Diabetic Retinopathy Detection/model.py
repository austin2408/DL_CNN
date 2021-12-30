from torchvision import models
import torch
import torch.nn as nn

def resnet(ver,pre=True,frez=False):
    if ver == 50:
        resnet = models.resnet50(pretrained=pre)
    else:
        resnet = models.resnet18(pretrained=pre)
    
    if frez == True:
        for param in resnet.parameters():
            param.requires_grad = False
    
    fc_inputs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Linear(fc_inputs, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 5)
    )

    return resnet



