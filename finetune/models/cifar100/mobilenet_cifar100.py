import timm
import torch
import torch.nn as nn



__all__ = ["mobilenetv2_cifar100"]



def mobilenetv2_cifar100():
    model = timm.create_model('mobilenetv2_100', pretrained = True)
    model.classifier = nn.Linear(1280, 100, True)
    return model

