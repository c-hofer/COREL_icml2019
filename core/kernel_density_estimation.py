"""kernel_density_estimation.py

Core routines for computing kernel density estimates.i

Author(s): chofer, rkwitt (2018)
"""

import torch
import math 

from collections import Counter


def pairwise_difference_vector_batch(x, y):
    """
    Computes pairwise differences for two batches of features, i.e.,

        output[i, j] = x[i] - x[j]

    Args:
        x: Tensor, shape=BxD
        y: Tensor, shape=BxD
    """
    assert x.dim() == 2 and y.dim() == 2 
    assert x.size(1) == y.size(1)    
    
    x_ = x.unsqueeze(0).expand(y.size(0), -1, -1)
    y_ = y.unsqueeze(0).expand(x.size(0), -1, -1)
    x_ = x_.transpose(0, 1)
    
    return x_ - y_
        

def distance_matrix_l1(x, y):
    """
    Computes a l1-distance matrix between two batches of features, i.e.,
    each distance matrix entry contains

        output[i, j] = || x[i] - y[j] ||_1

    Args:
        x: Tensor, shape=BxD
        y: Tensor, shape=BxD    
    """
    m = pairwise_difference_vector_batch(x, y)
    
    return  m.abs().sum(2)

def distance_matrix_l2(x, y):
    """
    Computes a l2-distance matrix between two batches of features, i.e.,
    each distance matrix entry contains

        output[i, j] = || x[i] - y[j] ||_2

    Args:
        x: Tensor, shape=BxD
        y: Tensor, shape=BxD    
    """
    m = pairwise_difference_vector_batch(x, y)
    
    return m.pow(2).sum(2).sqrt().squeeze()


class TruncatedExponentialL1KDM:
    
    def __init__(self, dimension, scaling=1, truncation_radius=1, kernel_aggregation=lambda x: x.sum(1)):        
        self._scaling    = scaling
        self._dimension  = dimension    
        self._train_data = None
        self._truncation_radius = truncation_radius
        self._norm_factor = (2*scaling*(1- math.exp(-(truncation_radius/scaling))))**(-dimension)
        self._kernel_aggregation = kernel_aggregation
        self.stat = []
        
    def fit(self, train_data):
        assert torch.is_tensor(train_data)
        assert sum(train_data.size()[1:]) == self._dimension
        self._train_data = train_data
        
    def score_sample(self, X):        
        assert sum(self._train_data.size()[1:]) == self._dimension
        D = distance_matrix_l1(X, self._train_data)

        m = D > self._truncation_radius
        D = D/self._scaling
        x_ = (-D).exp()
        x_[m] = 0
        
        x_ = x_*self._norm_factor
        x_ = self._kernel_aggregation(x_)
        return x_
    
    
class CountScorer:
    
    def __init__(self, dimension, truncation_radius=1):        
        
        self._dimension  = dimension    
        self._train_data = None
        self._truncation_radius = truncation_radius
        
        
    def fit(self, train_data):
        assert torch.is_tensor(train_data)
        assert sum(train_data.size()[1:]) == self._dimension
        self._train_data = train_data
        
    def score_sample(self, X):        
        assert sum(self._train_data.size()[1:]) == self._dimension
        D = distance_matrix_l1(X, self._train_data)

        m = D > self._truncation_radius
        
        x_    = torch.ones_like(D)
        x_[m] = 0
        
        #self.stat = Counter(x_.nonzero()[:, 1].view(-1).tolist())
        
        x_ = x_.sum(1)
        return x_
    
    
class CountScorerProduct:
    
    def __init__(self, dimension, sub_space_dim,  truncation_radius=1, aggregation='sum'):        
        
        assert dimension % sub_space_dim == 0
        self._sub_space_dim = sub_space_dim
        
        self._dimension  = dimension    
        self._train_data = None
        self._truncation_radius = truncation_radius
        self.stat = None
        self.aggregation = aggregation
        
        
    def fit(self, train_data):
        assert torch.is_tensor(train_data)
        assert sum(train_data.size()[1:]) == self._dimension
        self._train_data = train_data
        
    def score_sample(self, X):        
        assert sum(self._train_data.size()[1:]) == self._dimension
        
        res = []

        for i in range(0, self._dimension, self._sub_space_dim):
            
            D = distance_matrix_l1(X[:, i:(i+self._sub_space_dim)] , self._train_data[:, i:(i+self._sub_space_dim)])
            
            m = D > self._truncation_radius

            x_    = torch.ones_like(D)
            x_[m] = 0
            x_ = x_.sum(1)
            
            res.append(x_)
        
        res = torch.stack(res, dim=1)
        
        if self.aggregation == 'sum':
            res = res.sum(1)
            
        elif self.aggregation == 'max':
            res = res.max(1)[0]      
            
        elif self.aggregation == 'mean':
            res = res.mean(1)
        
        return res
