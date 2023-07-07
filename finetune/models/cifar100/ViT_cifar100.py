import torch
import torch.nn as nn
import timm

__all__ = ['vit_base_cifar100']


def vit_base_cifar100():
   model =  timm.create_model('vit_base_patch16_224', pretrained = True)
   model.head = nn.Linear(768, 100, bias = True)
                                                                                         
   return model
