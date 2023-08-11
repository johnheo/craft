import copy
import torch
import torchvision.models as torch_models
from . import cifar10 as cifar10_models
from . import cifar100 as cifar100_models
from . import flowers102 as flowers102_models
from . import stanfordcars as stanfordcars_models
from . import oxfordiiit as oxfordpets_models
from . import fgvcaircraft as aircraft_models
from . import food101 as food101_models

from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torch.nn as nn
import logging
msglogger = logging.getLogger()


SUPPORTED_DATASETS = ('cifar10', 'cifar100', 'flowers102', 'stanfordcars', 'oxfordpets', 'fgvc-aircraft', 'food101')

TORCHVISION_MODEL_NAMES = sorted(
                            name for name in torch_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torch_models.__dict__[name]))

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))

CIFAR100_MODEL_NAMES = sorted(name for name in cifar100_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar100_models.__dict__[name]))

STANFORDCARS_MODEL_NAMES = sorted(name for name in stanfordcars_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(stanfordcars_models.__dict__[name]))


FLOWERS102_MODEL_NAMES = sorted(name for name in flowers102_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(flowers102_models.__dict__[name]))

OXFORD_PETS_MODEL_NAMES = sorted(name for name in oxfordpets_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(oxfordpets_models.__dict__[name]))

AIRCRAFT_MODEL_NAMES = sorted(name for name in aircraft_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(aircraft_models.__dict__[name]))

FOOD101_MODEL_NAMES = sorted(name for name in food101_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(food101_models.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(),
                            set(FOOD101_MODEL_NAMES + AIRCRAFT_MODEL_NAMES+ OXFORD_PETS_MODEL_NAMES + STANFORDCARS_MODEL_NAMES + FLOWERS102_MODEL_NAMES + CIFAR10_MODEL_NAMES + CIFAR100_MODEL_NAMES)))


def create_model(pretrained, dataset, arch, process_group_params=None, parallel='DP', device_ids=None):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained [boolean]: True is you wish to load a pretrained model.
            Some models do not have a pretrained version.
        dataset: dataset name (only 'imagenet' and 'cifar10' are supported)
        arch: architecture name
        parallel [boolean]: if set, use torch.nn.DataParallel
        device_ids: Devices on which model should be created -
            None - GPU if available, otherwise CPU
            -1 - CPU
            >=0 - GPU device IDs
    """
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset {} is not supported'.format(dataset))

    model = None
    cadene = False
    try:
        if dataset == 'cifar10':

                   model = _create_cifar10_model(arch, pretrained)

        elif dataset == 'mnist':
                   model = _create_mnist_model(arch, pretrained)

        elif dataset == 'flowers102':

                   model = _create_flowers102_model(arch, pretrained)

        elif dataset == 'stanfordcars':
                   model = _create_stanfordcars_model(arch, pretrained)

        elif dataset == 'oxford-iiit':

                   model = _create_oxfordpets_model(arch, pretrained)

        elif dataset == 'food101':

                   model = _create_food101_model(arch, pretrained)

        elif dataset == 'fgvc-aircraft':
    
                   model = _create_aircraft_model(arch, pretrained)

        elif dataset == 'cifar100':
                  model = _create_cifar100_model(arch, pretrained)
    
    except ValueError:

        raise ValueError('Could not recognize dataset {} and model {} pair'.format(dataset, arch))



    if process_group_params is None:
        msglogger.info("=> created a %s%s model with the %s dataset" % ('pretrained ' if pretrained else '',
                                                                        arch, dataset))
    elif process_group_params['rank'] == 0:
        msglogger.info("=> created a %s%s model with the %s dataset" % ('pretrained ' if pretrained else '',
                                                                        arch, dataset))

    if torch.cuda.is_available() and device_ids != -1:
        if parallel == 'DP':
            device = 'cuda'
        else:
            rank = process_group_params['rank']
            device = torch.device(rank)

        if parallel is not None:
            if (arch.startswith('alexnet') or arch.startswith('vgg')):
                if parallel == 'DP':
                    model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
                else:
                    model.features = model.features.to(device)
                    model.features = DDP(model.features, device_ids=[device])
            else:
                if parallel == 'DP':
                    model = torch.nn.DataParallel(model, device_ids=device_ids)
                else:
                    model = model.to(device)
                    model = DDP(model, device_ids=[device])
    else:
        device = 'cpu'

    _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene)

    if parallel == 'DP':
        return model.to(device)
    else:
        return model

def _create_cifar10_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = cifar10_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def _create_stanfordcars_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = stanfordcars_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def _create_food101_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = food101_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def _create_oxfordpets_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = oxfordpets_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def _create_aircraft_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = aircraft_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model

def _create_flowers102_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = flowers102_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model

def _create_cifar100_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR100) does not have a pretrained model".format(arch))
    try:
        model = cifar100_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR100".format(arch))
    return model
