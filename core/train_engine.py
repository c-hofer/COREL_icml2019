"""train_engine.py

Core routines for training the proposed autoencoder model.

Author(s): chofer, rkwitt (2018)
"""

import os
import sys
import uuid
import glob
import torch
import pickle
import json
import numpy as np
import inspect
import types
import itertools

import config

from collections import defaultdict

import core.autoencoder as autoencoder
from .dataset import ds_random_subset, ds_monkey_patch_targets, TinyImageNet, ImageNet

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

from chofer_torchex.pershom import pershom_backend
vr_l1_persistence = pershom_backend.__C.VRCompCuda__vr_persistence_l1


# region 


def get_keychain_value(d, key_chain=None, allowed_values=(list,)):
    key_chain = [] if key_chain is None else list(key_chain).copy()   
    
    
    if not isinstance(d, dict):
        if allowed_values is not None:
            assert isinstance(d, allowed_values), 'Value needs to be of type {}!'.format(allowed_values)
        yield key_chain, d
    else:
        for k, v in d.items():
            yield from get_keychain_value(v, key_chain + [k], allowed_values=allowed_values)
    

def configs_from_grid(grid):
    tmp = list(get_keychain_value(grid))
    values = [x[1] for x in tmp]
    key_chains = [x[0] for x in tmp]
    
    ret = []
    
    for v in itertools.product(*values):
        
        ret_i = {}
        
        for kc, kc_v in zip(key_chains, v):
            tmp = ret_i
            for k in kc[:-1]:
                if k not in tmp:
                    tmp[k] = {}
                    
                tmp = tmp[k]
                
            tmp[kc[-1]] = kc_v
        
        ret.append(ret_i)        
            
    return ret


# endregion


DS_ROOT = config.paths.dataset_root_generic
DEVICE  = "cuda"


def dataset_factory(*ignore, dataset, subset_ratio, train):

    assert len(ignore) == 0

    # For CIFAR 10/100 we only need to ensure that values are
    # in [0,1]
    cifar_transform = transforms.Compose([
        transforms.ToTensor()])

    # For TinyImageNet-200, we need to make sure that images
    # are of the same size as in CIFAR 10/100 AND values are
    # in the range [0,1]
    tiny_imagenet_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()])
    
    imagenet_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()])

    if dataset == 'cifar10':
        ds = CIFAR10(
            root=os.path.join(DS_ROOT, dataset),
            train=train,
            transform=cifar_transform,
            download=True)
    elif dataset == 'cifar100':
         ds = CIFAR100(
            root=os.path.join(DS_ROOT, dataset),
            train=train,
            transform=cifar_transform,
            download=True)
    elif dataset == 'cifar20':
        ds = CIFAR100(
            root=os.path.join(DS_ROOT, dataset),
            train=train,
            transform=cifar_transform,
            download=True)
        
        info_file = os.path.join(DS_ROOT, dataset,'cifar-100-python')
        if train:
            info_file = os.path.join(info_file, 'train') 
        else: 
            info_file = os.path.join(info_file, 'test')
        
        with open(info_file, 'rb') as fid: 
            tmp = pickle.load(fid, encoding='latin1')
        if train:
            ds.train_labels = tmp['coarse_labels']
        else:
            ds.test_labels = tmp['coarse_labels']
     
    elif dataset == 'tiny-imagenet-200':
        ds = TinyImageNet(
                root=os.path.join(DS_ROOT, dataset),
                transform=tiny_imagenet_transform,
                train=train)
    elif dataset == 'ImageNet':
        # TODO: hardcoded path for NOW
        ds = ImageNet(
                root=os.path.join('/scratch_ssd/chofer/data', dataset),
                transform=imagenet_transform,
                train=train)
    else:
        raise Exception()

    # Monkey patch dataset to have a target member
    # variable that holds one label per image.
    ds_monkey_patch_targets(ds)

    # If the subset_ratio is < 1, create a random
    # subset from the original data that contains
    # only a fraction (subset_ratio) of all data.
    if subset_ratio is not None and subset_ratio < 1:
        ds = ds_random_subset(ds, percentage=subset_ratio)

    return ds


def l1_loss(x_hat, x, reduce=True):
    """
    L1 loss used for reconstruction.
    """
    l = (x - x_hat).abs().view(x.size(0), - 1).sum(dim=1)
    if reduce:
        l = l.mean()
    return l


model_mapping = {
    'DCGEncDec' : autoencoder.DCGEncDec
}


def check_config(config):
    assert 'train_args' in config
    train_args = config['train_args']

    assert 'learning_rate' in train_args
    assert 0 < train_args['learning_rate']

    assert 'batch_size' in train_args
    assert 0 < train_args['batch_size']

    assert 'n_epochs' in train_args
    assert 0 < train_args['n_epochs']

    assert 'rec_loss_w' in train_args
    assert 'top_loss_w' in train_args

    # check model-speficic args
    assert 'model_args' in config
    model_args = config['model_args']
    assert 'class_id' in model_args
    assert model_args['class_id'] in model_mapping
    assert 'kwargs' in model_args
    kwargs = model_args['kwargs']
    s = inspect.getfullargspec(model_mapping[model_args['class_id']].__init__)
    for a in s.kwonlyargs:
        assert a in kwargs

    # check data-specific args
    assert 'data_args' in config
    data_args = config['data_args']
    s = inspect.getfullargspec(dataset_factory)
    for a in s.kwonlyargs:
        assert a in data_args


def train(root_folder, config):
    check_config(config)

    train_args = config['train_args']
    model_args = config['model_args']

    model_class = model_mapping[model_args['class_id']]

    model = model_class(**model_args['kwargs']).to(DEVICE)

    latent_dim = model.n_branches*model.out_features_branch
    branch_siz = model.out_features_branch
    ball_radius = 1.0 # HARD-CODED for now - only affects the scaling

    optim = Adam(
        model.parameters(),
        lr=train_args['learning_rate'])

    ds = dataset_factory(**config['data_args'])
    dl = DataLoader(ds,
                    batch_size=train_args['batch_size'],
                    shuffle=True,
                    drop_last=True)

    log = defaultdict(list)

    model.train()
    for epoch in range(1,train_args['n_epochs']+1):

        for x,_ in dl:

            x = x.to(DEVICE)

            # Get reconstruction x_hat and latent
            # space representation z
            x_hat, z = model(x)

            # Set both losses to 0 in case we ever want to
            # disable one and still use the same logging code.
            top_loss = torch.tensor([0]).type_as(x_hat)
            rec_loss = torch.tensor([0]).type_as(x_hat)

            # Computes l1-reconstruction loss
            rec_loss = l1_loss(x_hat, x, reduce=True)

            # For each branch in the latent space representation,
            # we enforce the topology loss and track the lifetimes
            # for further analysis.
            lifetimes = []
            top_loss_branch = []
            for i in range(0, latent_dim, branch_siz):
                pers = vr_l1_persistence(
                    z[:, i:i+branch_siz].contiguous(), # per-branch z_1,...,z_B
                    0, 0)[0][0] # [0][0] gives non-essential in H_0

                if pers.dim() == 2:
                    pers = pers[:, 1] # all 0-dim. features have birth 0 in VR complex
                    lifetimes.append(pers.tolist())
                    top_loss_branch.append((pers - 2.0*ball_radius).abs().sum())

            # Sum of the topology loss over all branches and divide by the
            # number of branches.
            top_loss += sum(top_loss_branch)/float(len(top_loss_branch))

            # Log lifetimes as well as all losses we compute
            log['lifetimes'].append(lifetimes)
            log['top_loss'].append(top_loss.item())
            log['rec_loss'].append(rec_loss.item())

            loss = train_args['rec_loss_w']*rec_loss + \
                   train_args['top_loss_w']*top_loss

            model.zero_grad()
            loss.backward()
            optim.step()

        print('{}: rec_loss: {:.4f} | top_loss: {:.4f}'.format(
            epoch,
            np.array(log['rec_loss'][-int(len(ds)/train_args['batch_size']):]).mean()*train_args['rec_loss_w'],
            np.array(log['top_loss'][-int(len(ds)/train_args['batch_size']):]).mean()*train_args['top_loss_w']))

    # Create a unique base filename
    the_uuid = str(uuid.uuid4())
    basefile = os.path.join(root_folder, the_uuid)
    config['uuid'] = the_uuid

    # Save model
    torch.save(model, '.'.join([basefile, 'model', 'pht']))

    # Save the config used for training as well as all logging results
    out_data = [config, log]
    file_ext = ['config', 'log']
    for x,y in zip(out_data, file_ext):
        with open('.'.join([basefile, y, 'pickle']), 'wb') as fid:
            pickle.dump(x, fid)


class BackboneTrainResult:

    config_ext = '.config.pickle'
    model_ext  = '.model.pht'
    log_ext    = '.log.pickle'

    def __init__(self, path_base):
        self.path_base = path_base

        self._config_pth = self.path_base + self.config_ext
        self._model_pth  = self.path_base + self.model_ext
        self._log_pth = self.path_base + self.log_ext

    @property
    def config(self):
        with open(self._config_pth, 'rb') as f:
            return pickle.load(f)

    @property
    def model(self):
        return torch.load(self._model_pth)

    @property
    def log(self):
        with open(self._log_pth, 'rb') as f:
            return pickle.load(f)

    def __repr__(self):
        return repr(self.config)


class BackboneTrainResultFolder:
    def __init__(self, root):
        self._res = self._load_train_results_from_folder(root)
        self.root = root

    def _load_train_results_from_folder(self, root):

        f_config = glob.glob(os.path.join(root, '*' + BackboneTrainResult.config_ext))
        f_model  = glob.glob(os.path.join(root, '*' + BackboneTrainResult.model_ext))
        f_log    = glob.glob(os.path.join(root, '*' + BackboneTrainResult.log_ext))

        f_config.sort(), f_model.sort(), f_log.sort()

        ret = []

        for config, model, log in zip(f_config, f_model, f_log):
            assert len(set((config.split('.')[0], model.split('.')[0], log.split('.')[0]))) == 1
            base = config.split('.')[0]
            ret.append(BackboneTrainResult(base))

        return ret

    def __len__(self):
        return len(self._res)

    def __iter__(self):
        return iter(self._res)

    def __getitem__(self, k):
        return self._res[k]

    def where(self, **kwargs):
        res = self._res
        for attribute, values in kwargs.items():
            keys = attribute.split('__')

            def pred(x):
                v = x.config
                for k in keys:
                    v = v[k]

                if isinstance(values, types.LambdaType):
                    return values(v)

                if isinstance(v, (tuple, list)):
                    return v in values

                else:
                    return v == values

            res = [r for r in res if pred(r)]

        return res

    def where_unique(self, **kwargs):
        res = self.where(**kwargs)
        assert len(res) == 1, "Your query was not unique. {} elements were found.".format(len(res))
        return res[0]

    def __repr__(self):
        s = ''
        s += type(self).__name__
        s += ' ({} -> {} element(s)'.format(self.root, len(self))

        return s
