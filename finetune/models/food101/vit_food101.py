import timm
import torch
import torch.nn as nn

__all__ = ['vit_food101']


def vit_food101():

   model =  timm.create_model('vit_base_patch16_224_in21k', pretrained = True)
   model.head = nn.Linear(768, 101, bias = True)

   return model
