import torch
import torchvision
import numpy as np
from collections import Counter

import os.path as pth


class ImageNet(torch.utils.data.Dataset):

    def __init__(self, root, train=True, transform=None):

        folder = 'train' if train else 'val'
        root = pth.join(root, folder)

        self._ds = torchvision.datasets.ImageFolder(root, transform=transform)
        self.targets = [y for x, y in self._ds.samples]
                      
    def __getitem__(self, k):
        return self._ds[k]
                      
    def __len__(self):
        return len(self._ds)


class TinyImageNet(torch.utils.data.Dataset):
    
    def __init__(self, root, train=True, transform=None):
        self._root = root
        ds = None
        if train:
            root = pth.join(self._root, 'train')
            ds = torchvision.datasets.ImageFolder(root, 
                                                  transform=transform)
            Y = [y for _, y in ds.samples]
            
        else:
            
            label_map = torchvision.datasets.ImageFolder(pth.join(self._root, 'train')).class_to_idx
                                                         
            root = pth.join(self._root, 'val')
            ds = torchvision.datasets.ImageFolder(root, transform=transform)
            with open(pth.join(root, 'val_annotations.txt'), 'r') as fid:
                content = fid.readlines()

            img_to_class = {}
            for l in content:
                parts = l.split('\t')
                img_to_class[parts[0]] = label_map[parts[1]]
                
            Y = [img_to_class[pth.basename(img_name)] for img_name, _ in ds.imgs]
       
        assert(ds is not None)
        self._ds = ds
        self.targets = Y
                      
    def __getitem__(self, k):
        return self._ds[k][0], self.targets[k]
                      
    def __len__(self):
        return len(self._ds)


def ds_monkey_patch_targets(dataset):
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    
    if hasattr(dataset, 'targets'):
        return dataset
    
    targets = None
    if isinstance(dataset, (torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100)):
        if dataset.train:
            targets = dataset.train_labels
        else:
            targets = dataset.test_labels
            
    else:
        targets = []
        for i in range(len(dataset)):
            targets.append(dataset[i][1])
            
    dataset.targets = targets
    
    return dataset


def ds_subset(dataset, indices):
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    assert hasattr(dataset, 'targets'), "dataset is expected to have 'targets' attribute"
    assert len(set(indices)) == len(indices), "indices is expected to have unique entries"
    assert all(isinstance(i, int) for i in indices), "indices is expected to have int entries."
    assert min(indices) >= 0 and max(indices) <= len(dataset), "indices is expected to have entries in [0, len(dataset)]."
    
    ret = torch.utils.data.dataset.Subset(dataset, indices)
    
    if hasattr(dataset, 'targets'): 
        ret.targets = [dataset.targets[i] for i in indices]
    
    return ret


def ds_random_subset(dataset, percentage=None, absolute_size=None, replace=False):
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    assert percentage is not None or absolute_size is not None
    assert not (percentage is None and absolute_size is None)
    if percentage is not None: assert 0 < percentage and percentage < 1, "percentage assumed to be > 0 and < 1"
    if absolute_size is not None: assert absolute_size <= len(dataset)
    
    n_samples =  int(percentage*len(dataset)) if percentage is not None else absolute_size
    indices = np.random.choice(list(range(len(dataset))), 
                            n_samples, 
                            replace=replace)
    
    indices = [int(i) for i in indices]
    
    return ds_subset(dataset, indices)


def ds_label_filter(dataset, labels):
    assert isinstance(labels, (tuple, list)), "labels is expected to be list or tuple."
    assert len(set(labels)) == len(labels), "labels is expected to have unique elements."
    assert hasattr(dataset, 'targets'), "dataset is expected to have 'targets' attribute"
    assert set(labels) <= set(dataset.targets), "labels is expected to contain only valid labels of dataset"
    
    indices = [i for i in range(len(dataset)) if dataset.targets[i] in labels]
    
    return ds_subset(dataset, indices)