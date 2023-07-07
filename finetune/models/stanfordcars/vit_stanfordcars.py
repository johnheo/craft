import timm
import torch
import torch.nn as nn

__all__ = ['vit_stanfordcars']

def vit_stanfordcars():

   model =  timm.create_model('vit_base_patch16_224_in21k', pretrained = True)
   model.head = nn.Linear(768, 196, bias = True)

   return model
