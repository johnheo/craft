import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


__all__ = [
    "resnet50_cifar100",
    "resnet101_cifar100"
                         
]

def resnet50_cifar100():
    model = timm.create_model('resnet50', pretrained = True)
    model.fc = nn.Linear(2048, NUM_CLASSES, True)
    return model

def resnet101_cifar100():

    model = timm.create_model('resnet101', pretrained = True)
    model.fc = nn.Linear(2048, NUM_CLASSES, True)
    return model
