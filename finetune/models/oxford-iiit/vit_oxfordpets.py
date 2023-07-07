import timm
import torch
import torch.nn as nn

__all__ = ['vit_oxfordpets']


def vit_oxfordpets():

   model =  timm.create_model('vit_base_patch16_224_in21k', pretrained = True)
   model.head = nn.Linear(768, 37, bias = True)

   return model
