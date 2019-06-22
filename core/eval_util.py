import torch 
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from time import time
from sklearn.metrics import roc_auc_score, roc_curve

from .dataset import ds_label_filter

import warnings
warnings.warn('{} eval_util.py is deprecated'.format(__file__))


def apply_model(source, model, device='cpu')->[[]]:

    dl = None
    if isinstance(source, torch.utils.data.dataset.Dataset):
        dl = DataLoader(source, batch_size=100, num_workers=10)
    elif isinstance(source, DataLoader):
        dl = source 
    else: 
        raise ValueError('Expected DataLoader or DataSet object!')
    
    X, Y = [], []
    model.eval()
    model.to(device)
    for x, y in dl:
        
        y = y.to(device)
        x = x.to(device)
        
        x_hat = model(x)
        
        X.append(x_hat)
        Y.append(y)

    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
        
    return X, Y


def evaluate(model, ds_train, ds_test, kde_factory, device='cpu'):

    collect = defaultdict(list)

    # Get testing data latent space representations with labels
    X_tst, Y_tst = apply_model(ds_test, model, device=device)
    
    n_classes = len(set(ds_test.targets))
    
    # Consider each class as an inlier class once and test against
    # all other classes.
    
    assert set(range(n_classes)) == set(ds_test.targets)
    for inlier_class in range(n_classes):
        t0 = time()
        
        kde = kde_factory()   
        
        # Get a dataset containing ONLY inlier samples from
        # the training portion of the dataset.
        ds_train_inlier = ds_label_filter(ds_train, labels=[inlier_class])

        # Get ONLY inliear latent space representations using the 
        # trained model and fit our counting routine.
        X_inl,_      = apply_model(ds_train_inlier, model, device=device)
        kde.fit(X_inl)
        
        # Create a list of labels which contains 1's for inliers and 0's 
        # for non-inlier's. Then, score each sample using the counting 
        # routine.
        Y_tst_this_class = [(1 if y == inlier_class else 0) for y in Y_tst]
        Y_sco = kde.score_sample(X_tst).cpu()
        
        # Compute AUC score, assuming that counting scores for the inlier
        # class are higher, i.e., for a ball around an inlier sample, we 
        # find more instances of the inlier class we used for training.
        auc_score = 100*roc_auc_score(Y_tst_this_class, Y_sco)
        fpr, tpr, thr = roc_curve(Y_tst_this_class, Y_sco.numpy())
        fpr, tpr = 100.0*fpr, 100.0*tpr

        collect['auc_scores'].append(auc_score.tolist())
        collect['fpr'].append(fpr.tolist())
        collect['tpr'].append(tpr.tolist())
        collect['thr'].append(thr.tolist())        
        
        ela =  time()-t0
        print('Class {:d} {:d} | AUC: {:.2f} [{:.4f}]'.format(
            inlier_class, len(X_inl), auc_score, ela))

        del X_inl
                
    return collect