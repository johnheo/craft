from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import vision_transformer
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention

def attention_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def window_attention_forward(self, x, mask = None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B
    
def prep_ckpt(ckpt):
    if 'state_dict' in ckpt.keys():
        ckpt = ckpt['state_dict']
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            new_ckpt[k[7:]] = v
        else:
            new_ckpt[k] = v
    return new_ckpt
    
def get_net(args, name):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.

    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384

    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    DIR='/mnt/hdd-nfs/johnheo/wacv/wanda/image_classifiers/ckpts/'
    if 'vit' in args.model:
        model = 'vit-B' # 'effnet-B0'
    elif 'eff' in args.model:
        model = 'effnet-B0'
    else:
        raise NotImplementedError
    
    path = DIR + args.resume
    print('read ckpt from: ', path)

    ckpt = torch.load(path, map_location='cpu')
    if args.dataset == 'imagenet':
        nb_classes = 1000
    elif args.dataset == 'cifar100':
        nb_classes = 100
    elif args.dataset == 'cifar10':
        nb_classes = 10
    elif args.dataset == 'flowers-102':
        nb_classes = 102
    elif args.dataset == 'stanford-cars':
        nb_classes = 196
    elif args.dataset == 'oxford-pets':
        nb_classes = 37
    elif args.dataset == 'aircraft':
        nb_classes = 100
    elif args.dataset == 'food-101':
        nb_classes = 101
    else:
        raise NotImplementedError

    net = timm.create_model(name, pretrained=False, num_classes=nb_classes)
    net.load_state_dict(prep_ckpt(ckpt))
    # net = timm.create_model(name, pretrained=True)


    for name, module in net.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(window_attention_forward, module)

    net.cuda()
    net.eval()
    return net
