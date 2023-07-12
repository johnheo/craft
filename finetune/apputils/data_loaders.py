#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Helper code for data loading.

This code will help with the image classification datasets: ImageNet and `CIFAR10

"""
import os
from urllib.request import urlretrieve
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split
from torch.utils.data import Dataset
import numpy as np
import h5py
import pandas as pd
from autoaugment import CIFAR10Policy, Cutout
import distiller




DATASETS_NAMES = ['imagenet', 'cifar10', 'mnist', 
                  'jsc', 'tiny',
                  'cifar10contrastive', 
                  'cifar10autoaugment', 'aircraft',
                  'cifar100', 'flowers102', 'stanfordcars', 'oxfordpets', 'food101'
                 ]


def classification_dataset_str_from_arch(arch):
    if 'cifar100' in arch:
        dataset = 'cifar100'
    elif 'cifar10contrastive' in arch:
        dataset = 'cifar10contrastive'
    elif 'cifar10autoaugment' in arch:
        dataset = 'cifar10autoaugment'
    elif 'cifar' in arch:
        dataset = 'cifar10'
    elif 'flowers102' in arch:
        dataset = 'flowers102'
    elif 'stanfordcars' in arch:
        dataset = 'stanfordcars'
    elif 'oxfordpets' in arch:
        dataset = 'oxfordpets'
    elif 'aircraft' in arch: 
        dataset = 'aircraft'
    elif 'food101' in arch:
        dataset = 'food101'
    elif 'mnist' in arch:
        dataset = 'mnist'
    elif 'jsc' in arch:
        dataset = 'jsc'
    elif 'tiny' in arch:
        dataset = 'tiny'
    else:
        dataset = 'imagenet'
    return dataset


def classification_num_classes(dataset):
    return {'cifar10contrastive': 10,
            'cifar10autoaugment': 10,
            'cifar10': 10,
            'cifar100': 100,
            'mnist': 10,
            'jsc': 5,
            'oxfordpets': 37,
            'aircraft': 100,
            'food101': 101,
            'flowers102': 102,
            'tiny': 200,
            'imagenet': 1000}.get(dataset, None)


def classification_get_input_shape(dataset):
    if dataset == 'imagenet':
        return 1, 3, 224, 224
    elif dataset == 'cifar10':
        return 1, 3, 32, 32
    elif dataset == 'cifar10contrastive':
        return 1, 3, 32, 32
    elif dataset == 'aircraft':
        return 1, 3, 224, 224
    elif dataset == 'food101':
        return 1, 3, 224, 224
    elif dataset == 'cifar10autoaugment':
        return 1, 3, 224, 224
    elif dataset == 'cifar100':
        return 1, 3, 32, 32
    elif dataset == 'mnist':
        return 1, 1, 28, 28
    elif dataset == 'jsc':
        return 1, 16
    elif dataset == 'flowers102':
        return 1, 3, 224, 224
    elif dataset == 'oxfordpets':
        return 1, 3, 224, 224
    elif dataset == 'stanfordcars':
        return 1, 3, 224, 224
    elif dataset == 'tiny':
        return 1, 3, 224, 224
    else:
        raise ValueError("dataset %s is not supported" % dataset)


def __dataset_factory(dataset):
    return {'cifar10contrastive': cifar10contrastive_get_datasets,
            'cifar10autoaugment': cifar10autoaugment_get_datasets,
            'cifar10': cifar10_get_datasets,
            'cifar100': cifar100_get_datasets,
            'mnist': mnist_get_datasets,
            'food101': food101_get_datasets,
            'flowers102': flowers102_get_datasets,
            'jsc': jsc_get_datasets,
            'aircraft': aircraft_get_datasets,
            'tiny': tiny_get_datasets,
            'oxfordpets': oxfordpets_get_datasets,
            'stanfordcars': stanfordcars_get_datasets,
            'imagenet': imagenet_get_datasets}.get(dataset, None)


def load_data(dataset, data_dir, batch_size, workers, parallel='DP', validation_split=0.1, deterministic=False,
              effective_train_size=1., effective_valid_size=1., effective_test_size=1.,
              fixed_subset=True, sequential=False):
    """Load a dataset.

    Args:
        dataset: a string with the name of the dataset to load (cifar10/imagenet)
        data_dir: the directory where the datset resides
        batch_size: the batch size
        workers: the number of worker threads to use for loading the data
        validation_split: portion of training dataset to set aside for validation
        deterministic: set to True if you want the data loading process to be deterministic.
          Note that deterministic data loading suffers from poor performance.
        effective_train/valid/test_size: portion of the datasets to load on each epoch.
          The subset is chosen randomly each time. For the training and validation sets,
          this is applied AFTER the split to those sets according to the validation_split parameter
        fixed_subset: set to True to keep the same subset of data throughout the run
          (the size of the subset is still determined according to the effective_train/valid/test
          size args)
    """
    if dataset not in DATASETS_NAMES:
        raise ValueError('load_data does not support dataset %s" % dataset')
    datasets_fn = __dataset_factory(dataset)
    return get_data_loaders(datasets_fn, data_dir, batch_size, workers, parallel,
                            validation_split=validation_split,
                            deterministic=deterministic,
                            effective_train_size=effective_train_size,
                            effective_valid_size=effective_valid_size,
                            effective_test_size=effective_test_size,
                            fixed_subset=fixed_subset,
                            sequential=sequential)


def mnist_get_datasets(data_dir):
    """Load the MNIST dataset."""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    ########## Mahdi Nazemi <mnazemi@usc.edu> ##########
    class MNISTNullaNetDataset(Dataset):
        def __init__(self, dataset_file):
            self.train = torch.load(dataset_file)

        def __len__(self):
            return len(self.train)

        def __getitem__(self, idx):
            img, target = self.train[idx]
            return img, target

    #train_dataset = MNISTNullaNetDataset('/home/nazemi/university/neural-networks/primitive/mnist_sparse/design-127/data/train.pth.tar')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset





def jsc_get_datasets(data_dir):
    """Load the JSC dataset."""
    class JSC(Dataset):
        resources = [
            ("https://cernbox.cern.ch/index.php/s/AgzB93y3ac0yuId/download?path=%2F&x-access-token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkcm9wX29ubHkiOmZhbHNlLCJleHAiOiIyMDIwLTA2LTExVDA0OjAxOjA0Ljk2MjA2MTI0NyswMjowMCIsImV4cGlyZXMiOjAsImlkIjoiODY0MDYiLCJpdGVtX3R5cGUiOjEsIm10aW1lIjoxNTEwODQ5MDI5LCJvd25lciI6Indvb2Rzb24iLCJwYXRoIjoiZW9zaG9tZS13OjE0OTk5NjQiLCJwcm90ZWN0ZWQiOmZhbHNlLCJyZWFkX29ubHkiOnRydWUsInNoYXJlX25hbWUiOiJzYW1wbGVzIiwidG9rZW4iOiJBZ3pCOTN5M2FjMHl1SWQifQ.jSu5szYRU1-h1CsMcb0VIAbUhxW14DNPsEnJm-fnxNA&files=processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z"),
                ]

        training_file = 'training.pt'
        test_file = 'test.pt'
        classes = list(range(5))

        def __init__(self, root, train=True, download=False):
            super(JSC, self).__init__()
            self.root = root
            self.train = train

            if not os.path.exists(root):
                self.download()
                
            if self.train:
                data_file = self.training_file
            else:
                data_file = self.test_file
            self.data, self.targets = torch.load(os.path.join(self.root, data_file))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            features, target = self.data[index], int(self.targets[index])

            return features, target

        def download(self):
            os.mkdir(self.root)
            url = self.resources[0]
            filename = 'processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z'
            urlretrieve(url, os.path.join(self.root, filename))
            self.prepare_data(filename)
            os.remove(os.path.join(self.root, filename))

        def prepare_data(self, filename):
            h5_file = h5py.File(os.path.join(self.root, filename), 'r')
            tree_array = h5_file['t_allpar_new'][()]

            features_labels = ['j_zlogz', 'j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt', 'j_c2_b1_mmdt', 'j_c2_b2_mmdt', 'j_d2_b1_mmdt', 'j_d2_b2_mmdt', 'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt', 'j_m2_b2_mmdt', 'j_n2_b1_mmdt', 'j_n2_b2_mmdt', 'j_mass_mmdt', 'j_multiplicity']
            target_labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']

            features_target_df = pd.DataFrame(tree_array, columns=list(set(features_labels + target_labels)))
            features_target_df = features_target_df.drop_duplicates()
            features_df = features_target_df[features_labels]
            target_df = features_target_df[target_labels]
            features = features_df.values
            target = target_df.values
            target = np.where(target == 1)[1]
            features = torch.Tensor(features).float()
            target = torch.Tensor(target).long()

            #features = (features - features.mean()) / features.std()
            features = (features - features.mean(dim=0)) / features.std(dim=0)

            indices = np.arange(len(features))
            np.random.shuffle(indices)
            split_size = int(0.8 * len(features))
            train_indices = indices[:split_size]
            test_indices = indices[split_size:]

            data_file = self.training_file
            torch.save((features[train_indices], target[train_indices]), os.path.join(self.root, data_file))
            data_file = self.test_file
            torch.save((features[test_indices], target[test_indices]), os.path.join(self.root, data_file))

    train_dataset = JSC(root=data_dir, train=True, download=True)
    test_dataset = JSC(root=data_dir, train=False)

    return train_dataset, test_dataset



def cifar10contrastive_get_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=TwoCropTransform(train_transform))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def cifar10autoaugment_get_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        #Cutout(n_holes=1, length=16),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset



def food101_get_datasets(data_dir):
    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
#        transforms.RandomRotation(35),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.Food101(root=data_dir, split = 'train',
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = datasets.Food101(root=data_dir, split = 'test',
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def aircraft_get_datasets(data_dir):
    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(35),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.FGVCAircraft(root=data_dir, split = 'trainval',
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.FGVCAircraft(root=data_dir, split = 'test',
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset

def oxfordpets_get_datasets(data_dir):

    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
#        transforms.RandomRotation(35),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.OxfordIIITPet(root=data_dir, split = 'trainval',
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = datasets.OxfordIIITPet(root=data_dir, split = 'test',
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def flowers102_get_datasets(data_dir):

    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.Flowers102(root=data_dir, split = 'train',
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.Flowers102(root=data_dir, split = 'test',
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset



def cifar10_get_datasets(data_dir):
    """Load the CIFAR10 dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        #transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def cifar100_get_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
 #       transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
#        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset



def stanfordcars_get_datasets(data_dir):

    """
    Load the StanfordCars dataset for Transfer Learning

    """

    train_dir = os.path.join(data_dir, 'car_data/car_data/train')
    test_dir = os.path.join(data_dir, 'car_data/car_data/test')

    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(35),
#        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
#        transforms.RandomGrayscale(p=0.5),
#        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
#        transforms.RandomPosterize(bits=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )

    ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def imagenet_get_datasets(data_dir):
    """
    Load the ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def tiny_get_datasets(data_dir):
    """
    Load the Tiny ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[-0.0082, -0.0534, -0.0670],
                                     std=[1.1422, 1.1242, 1.1743])

    train_transform = transforms.Compose([
#        transforms.RandomResizedCrop(64),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        #transforms.Resize(64),
        transforms.Resize((224,224)),
        #transforms.CenterCrop(56),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def __image_size(dataset):
    # un-squeeze is used here to add the batch dimension (value=1), which is missing
    return dataset[0][0].unsqueeze(0).size()


def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def __split_list(l, ratio):
    split_idx = int(np.floor(ratio * len(l)))
    return l[:split_idx], l[split_idx:]


class SwitchingSubsetRandomSampler(Sampler):
    """Samples a random subset of elements from a data source, without replacement.

    The subset of elements is re-chosen randomly each time the sampler is enumerated

    Args:
        data_source (Dataset): dataset to sample from
        subset_size (float): value in (0..1], representing the portion of dataset to sample at each enumeration.
    """
    def __init__(self, data_source, effective_size):
        self.data_source = data_source
        self.subset_length = _get_subset_length(data_source, effective_size)

    def __iter__(self):
        # Randomizing in the same way as in torch.utils.data.sampler.SubsetRandomSampler to maintain
        # reproducibility with the previous data loaders implementation
        indices = torch.randperm(len(self.data_source))
        subset_indices = indices[:self.subset_length]
        return (self.data_source[i] for i in subset_indices)

    def __len__(self):
        return self.subset_length


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """Sequentially samples a subset of the dataset, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def _get_subset_length(data_source, effective_size):
    if effective_size <= 0 or effective_size > 1:
        raise ValueError('effective_size must be in (0..1]')
    return int(np.floor(len(data_source) * effective_size))


def _get_sampler(data_source, effective_size, fixed_subset=False, sequential=False):
    if fixed_subset:
        subset_length = _get_subset_length(data_source, effective_size)
        indices = range(len(data_source))
        subset_indices = indices[:subset_length]
        if sequential:
            return SubsetSequentialSampler(subset_indices)
        else:
            return torch.utils.data.SubsetRandomSampler(subset_indices)
    return SwitchingSubsetRandomSampler(data_source, effective_size)

data_proxy = False
def get_data_loaders(datasets_fn, data_dir, batch_size, num_workers, parallel='DP', validation_split=0.1, deterministic=False,
                     effective_train_size=1., effective_valid_size=1., effective_test_size=1., fixed_subset=False,
                     sequential=False):
    train_dataset, test_dataset = datasets_fn(data_dir)

    worker_init_fn = None
    if deterministic:
        distiller.set_deterministic()
        worker_init_fn = __deterministic_worker_init_fn
    if (data_proxy):
         indices = []
         valid_indices = []
         file1 = open('/data/projects/nullanet/src/seyedarmin_sampling/nulla_distiller/nulla_distiller/examples/classifier_compression/design-42/proxy_train.txt', 'r')
         Lines = file1.readlines()
         for word in Lines:
           index = int(word.split(' ')[0])
           indices.append(index)
         file1 = open('/data/projects/nullanet/src/seyedarmin_sampling/nulla_distiller/nulla_distiller/examples/classifier_compression/design-42/proxy_val.txt', 'r')
         Lines = file1.readlines()
         for word in Lines:
           index = int(word.split(' ')[0])
           indices.append(index)
         #print(indices)
         indices = list(indices)
         #valid_indices = list(valid_indices)
         #train_dataset = train_dataset[indices]
         num_train = len(indices)
    else:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        #print(indices)
    # TODO: Switch to torch.utils.data.datasets.random_split()

    # We shuffle indices here in case the data is arranged by class, in which case we'd would get mutually
    # exclusive datasets if we didn't shuffle
    np.random.shuffle(indices)
   # np.random.shuffle(valid_indices)
    #print(indices)
    #print('parallel is ')
    #print(parallel)
    if parallel != 'DDP':
        valid_indices, train_indices = __split_list(indices, validation_split)
        #train_indices = indices
        #print('Length of training data is: '+str(len(train_indices)))
        #print('Length of valid data is: '+str(len(valid_indices)))
        ########## Mahdi Nazemi <mnazemi@usc.edu> ##########
        train_loader = None
        if train_indices:
            train_sampler = _get_sampler(train_indices, effective_train_size, fixed_subset, sequential)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, sampler=train_sampler,
                                                num_workers=num_workers, pin_memory=True,
                                                worker_init_fn=worker_init_fn)

        valid_loader = None
        if valid_indices:
            valid_sampler = _get_sampler(valid_indices, effective_valid_size, fixed_subset, sequential)
            valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size, sampler=valid_sampler,
                                                    num_workers=num_workers, pin_memory=True,
                                                    worker_init_fn=worker_init_fn)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                num_workers=num_workers, pin_memory=True)
    else:
        val_length = int(np.floor(validation_split * len(indices)))
        train_dataset, valid_dataset = random_split(train_dataset, (len(train_dataset) - val_length, val_length))

        train_loader = None
        if len(train_dataset) - val_length > 0:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size, sampler=train_sampler,
                                                    num_workers=num_workers, pin_memory=True,
                                                    worker_init_fn=worker_init_fn)

        valid_loader = None
        if val_length > 0:
            valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
            valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=batch_size, sampler=valid_sampler,
                                                    num_workers=num_workers, pin_memory=True,
                                                    worker_init_fn=worker_init_fn)

        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size, sampler=test_sampler,
                                                num_workers=num_workers, pin_memory=True,
                                                worker_init_fn=worker_init_fn)

    ########## Mahdi Nazemi <mnazemi@usc.edu> ##########
    # TODO: uncomment the following line
    #input_shape = __image_size(train_dataset)
    input_shape = None

    ########## Mahdi Nazemi <mnazemi@usc.edu> ##########
    # If validation split was 0 we use the test set as the validation set
    # If validation split was 1 we use the validation set as the training set to avoid returning a None value
    return train_loader or valid_loader, valid_loader or test_loader, test_loader, input_shape


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
