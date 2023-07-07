import torch
import torch.nn as nn
import timm


__all__ = ['vit_cifar10']


def vit_cifar10():
  
   model =  timm.create_model('vit_base_patch16_224_in21k', pretrained = True)
   model.head = nn.Linear(768, 10, bias = True)
                                                                                         
   return model
