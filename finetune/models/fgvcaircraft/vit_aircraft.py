import timm
import torch
import torch.nn as nn

__all__ = ['vit_aircraft']


def vit_aircraft():

   model =  timm.create_model('vit_base_patch16_224_in21k', pretrained = True)
   model.head = nn.Linear(768, 100, bias = True)

   return model
