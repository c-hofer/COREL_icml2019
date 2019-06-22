"""plot_util.py

Variety of plotting utilities.

Author(s): chofer, rkwitt

"""

import matplotlib.pyplot as plt 
import torch
import math
import numpy as np


from collections import defaultdict
from itertools import combinations
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader


device = 'cuda'


def plt_img_grid(images, nrow = 8): 
    """
    Plot images on a grid.
    """
    plt.figure(figsize=(16, 16))
    img = make_grid(images, normalize=True, nrow=nrow)
    npimg = img.cpu().detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
    
def plt_reconstructions(dl, m, n_reconstructions=12):
    """
    Plot reconstructions using data from a DataLoader and 
    an autoencoder model.

    Args:
        dl: DataLoader
            Instance of a DataLoader.

        m: nn.Module 
            Instance of an autoencoder model.

        n_reconstructions: int
            n_reconstructions are plotted - This uses
            the first n_reconstructions images supplied
            by dl.
    """
    if isinstance(dl, torch.utils.data.dataset.Dataset):
        dl = DataLoader(dl, batch_size=n_reconstructions)

    m = m if isinstance(m, list) else [m]
    m = [m.to(device) for m in m]
    (m.eval() for m in m)    
    
    x, _ = next(iter(dl))
    x = x[:n_reconstructions]
    x = x.to(device)
    
    for m_i in m:
        x_hat = m_i(x)
        if isinstance(x_hat, (tuple, list)): x_hat = x_hat[0]

        imgs = sum([[org, rec] for org, rec in zip(x, x_hat)], [])
        plt_img_grid(imgs, nrow=6)
    

def apply_model_by_label(dl, m)->[[]]:

    if isinstance(dl, torch.utils.data.dataset.Dataset):
        dl = DataLoader(dl, batch_size=100, num_workers=10)
    
    points_by_label = defaultdict(list)
    m.eval()
    m.to(device)
    for x, y in dl:
        
        y = y.tolist()
        x = x.to(device)
        
        x_hat = m(x)
        if isinstance(x_hat, (tuple, list)): x_hat = x_hat[0]
        
        x_hat = x_hat.detach().tolist()
        for point, label in zip(x_hat, y):  
            points_by_label[label].append(point)
        
    return points_by_label


def activations_as_labeld_points(data_loader, model):
    d = apply_model_by_label(data_loader, model)
    
    X, Y = [], []
    for k, v in d.items():
        X += v
        Y += [k]*len(v)
        
    return X, Y
        
        
def plt_activations(dl, m, dim_0=0, dim_1=1):
    plt.figure(figsize=(5, 5))
    
    activations_by_label = apply_model_by_label(dl,m)
    for label in sorted(activations_by_label):        
        activations = activations_by_label[label]
        activations = np.array(activations)
        plt.scatter(activations[:, dim_0], activations[:, dim_1], alpha=0.75, label=str(label))
        
    plt.legend()
        
        
def plt_activations_border_distributions(dl, m):
       
    
    activations_by_label = apply_model_by_label(dl,m)
    n_dims = len(activations_by_label[next(iter(activations_by_label))][0])
    columns = 5
    rows = math.ceil(n_dims/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 4*rows)) 
    
    for label in sorted(activations_by_label):
        
        activations = activations_by_label[label]
        
        activations = np.array(activations)
        for i in range(n_dims):
            ax = axes[i] if rows == 1 else axes[int(i/columns), i % columns]
            ax.hist(activations[:, i], bins=100, density=True, label=str(label))
        
    for ax in axes.ravel():
        ax.legend()
           
            
def plt_dict(log, ax=None):
    if ax is None:
        plt.figure(figsize=(4,3))
        ax = plt.gca()
        
    for k in sorted(log):
        v = log[k]
        
        ax.plot(v, label=k)
        
    ax.legend()
    
    
def plt_labeled_points(X, Y, ax=None, x_0=0, x_1=1, plot_kwargs=None):
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, list)
    assert isinstance(Y[0], int)
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs
    
    ax = ax if ax is not None else plt.gca()
    
    n_classes = len(set(Y))    
    
    Y_tensor = torch.tensor(Y)
    Y_tensor = Y_tensor.type(X.type())
    for y in range(n_classes):
        points = X.index_select(0, (Y_tensor == y).nonzero().squeeze())
        ax.plot(points[:, x_0].tolist(), points[:, x_1].tolist(), '.', label=str(y), **plot_kwargs)
        
    #ax.legend()
        
        
def plt_labeled_points_to_file(X, Y, file_path):
    fig = plt.figure()
    plt_labeled_points(X, Y)             
    fig.savefig(file_path)    
    plt.close(fig)   


def plt_labeled_dataset(dataset, n_cols=4, size=4, max_rows=10):
    ds_dim    = len(dataset[0][0])
    n_plots   = (ds_dim*(ds_dim-1)/2)
    n_rows    = min(math.ceil( n_plots / n_cols), max_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
    
    X, Y = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X.append(x), Y.append(y)
        
    X = torch.stack(X, dim=0)
    
    for i, (x_0, x_1) in enumerate(combinations(range(ds_dim),2)):
        ax = axes[int(i/n_cols), i % n_cols] if n_rows > 1 else axes[i]
        plt_labeled_points(X, Y, ax=ax, x_0=x_0, x_1=x_1)
        if i == n_rows*n_cols - 1: break
        
    return fig    


def plt_labeled_ds_to_file(X, Y, file_path):
    fig = plt.figure()
    fig = plot_labeled_ds(X, Y, fig)   

            
    fig.savefig(file_path)
        
    
    plt.close(fig)
        