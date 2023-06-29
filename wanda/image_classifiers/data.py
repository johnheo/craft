import os
import json

import torch
from torchvision import transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
# from autoaugment import CIFAR10Policy

def load_cifar(data, batch_size=256, num_workers=2, autoaugment=False, data_path = '/mnt/hdd-nfs/johnheo/data'):
    if data == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        data_loader = CIFAR10
    elif data == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        data_loader = CIFAR100
    else:
        raise NotImplementedError('Only CIFAR10 and CIFAR100 are supported')

    # Transforms
    if autoaugment:
        train_transform = transforms.Compose([
                                              transforms.Resize(224),
                                              transforms.RandomCrop(224, padding=4, fill=0),
                                              transforms.RandomHorizontalFlip(),
                                            #   CIFAR10Policy(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose([
                                              transforms.Resize(224),
                                              transforms.RandomCrop(224, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # DataLoader
    train_set = data_loader(root=data_path, train=True, download=True, transform=train_transform)
    test_set = data_loader(root=data_path, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    
    # return train_loader, test_loader
    return train_set, test_set