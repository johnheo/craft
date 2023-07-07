import timm
import torch
import torch.nn as nn

__all__ = ['vit_flowers102']


def vit_flowers102():

   model =  timm.create_model('vit_base_patch16_224_in21k', pretrained = True)
   model.head = nn.Linear(768, 102, bias = True)
                                                                                         
   return model
