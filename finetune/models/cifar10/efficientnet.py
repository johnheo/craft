import torch
import timm
import torch.nn as nn

__all__ =["efficientnet_b0_cifar10"]


def efficientnet_b0_cifar10():

    model = timm.create_model('efficientnet_b0', pretrained = True)
    model.classifier = nn.Linear(1280, 10, True)
    return model
