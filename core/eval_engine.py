import torch 
import uuid
import pickle
import numpy as np
import os
import json
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from time import time
import gc

from sklearn.metrics import roc_auc_score, roc_curve

from .dataset import ds_label_filter,  ds_random_subset
from .kernel_density_estimation import CountScorerProduct
from .train_engine import BackboneTrainResult, BackboneTrainResultFolder, dataset_factory, configs_from_grid
from .train_engine import get_keychain_value


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


def evaluate(model, ds_train, ds_test, kde_factory, 
             train_subset_size=1.0, 
             device='cpu'):

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

        train_subset_size = min(train_subset_size, len(ds_train_inlier))
        ds_train_inlier = ds_random_subset(ds_train_inlier, 
                                           absolute_size=train_subset_size,
                                           replace=False)
        assert len(ds_train_inlier) > 0

        # Get ONLY inliear latent space representations using the 
        # trained model and fit our counting routine.
        X_inl,_      = apply_model(ds_train_inlier, model, device=device)

        kde.fit(X_inl)
        
        # Create a list of labels which contains 1's for inliers and 0's 
        # for non-inlier's. Then, score each sample using the counting 
        # routine.
        Y_tst_this_class = [(1 if y == inlier_class else 0) for y in Y_tst]
        Y_sco = []
        batch_size = 200
        for i in range(0,X_tst.size(0),batch_size):
            batch_score = kde.score_sample(X_tst[i:(i+batch_size), :]).cpu()
            Y_sco += batch_score.tolist()

        # OLD: possibly missed samples
        #for i in range(int(X_tst.size(0)/batch_size)):
        #    batch_score = kde.score_sample(X_tst[i*batch_size:(i+1)*batch_size, :]).cpu()
        #    Y_sco += batch_score.tolist()
        
        # Compute AUC score, assuming that counting scores for the inlier
        # class are higher, i.e., for a ball around an inlier sample, we 
        # find more instances of the inlier class we used for training.
        auc_score = 100*roc_auc_score(Y_tst_this_class, Y_sco)
        fpr, tpr, thr = roc_curve(Y_tst_this_class, Y_sco)
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


def check_eval_config(config):
    key_chains = [kc[:2] for kc, _ in get_keychain_value(config, allowed_values=None)]
   
    assert ['dataset'] in key_chains
    assert ['n_train_samples'] in key_chains 
    assert ['scorer', 'id'] in key_chains
    assert ['scorer', 'kwargs'] in key_chains
    assert ['n_runs'] in key_chains
    

class ScorerFactory:
    def __init__(self, *ignore,      
                 scorer_id, 
                 input_dim, 
                 subspace_dim, 
                 scorer_kwargs = {}):
        assert len(ignore) == 0, ",Kwargs args only!"
        
        assert scorer_id == 'CountScorerProduct'
        self.input_dim = input_dim
        self.subspace_dim = subspace_dim
        self.scorer_kwargs = scorer_kwargs
        
    def __call__(self):
        return CountScorerProduct(self.input_dim, 
                                  self.subspace_dim,
                                  **self.scorer_kwargs)
    

def evaluate_backbone_with_scorer(eval_config, 
                                  backbone_trn_result,
                                  output_folder, 
                                  device=None): 

    assert isinstance(backbone_trn_result, BackboneTrainResult)       
    check_eval_config(eval_config)
    
    # Get whole train dataset
    trn = dataset_factory(dataset=eval_config['dataset'] ,
                          train=True, 
                          subset_ratio=None)

    # Get whole test dataset
    tst = dataset_factory(dataset=eval_config['dataset'] ,
                          train=False,
                          subset_ratio=None)
    
    # The encoder part of the backbone autoencoder is the model 
    model = backbone_trn_result.model
    enc = model.enc

    # This factory produces the final scorer wich will be used in the call to evaluate
    sc_factory = ScorerFactory(scorer_id=eval_config['scorer']['id'], 
                               input_dim=model.latent_dim, 
                               subspace_dim=model.latent_dim//backbone_trn_result.config['model_args']['kwargs']['latent_config']['n_branches'],
                               scorer_kwargs=eval_config['scorer']['kwargs'])
    
    print('eval_config:\n', eval_config, '\n', 'backbone_config:\n', backbone_trn_result.config, '\n')
    
    
    # Accumulate the training results for n runs  
    scores_by_run = []
    for run_i in range(eval_config['n_runs']):
        t_0 = time()
        print(run_i)
        scores = evaluate(enc, 
                        trn, 
                        tst, 
                        sc_factory, 
                        train_subset_size=eval_config['n_train_samples'],
                        device=device) 
        t_1 = time()
        scores['needed_time'] = t_1 - t_0 
        scores_by_run.append(scores)
    
    print('mean: ', np.array(scores['auc_scores']).mean())
    
    
    the_result = {
        'scores_by_run': scores_by_run, 
        'backbone_config': backbone_trn_result.config,
        'eval_config': eval_config
    }
    
    # Dump the result to the filesystem 
    file_path = os.path.join(output_folder, str(uuid.uuid4()) + '.eval.json')
    with open(file_path, 'w') as fid:
        json.dump(the_result, fid)
        
    
import gc
def evaluate_backbones_on_eval_config_grid(backbone_result_folder, 
                                           eval_config_grid, 
                                           output_folder,
                                           device):

    backbone_results = BackboneTrainResultFolder(backbone_result_folder)

    assert len(backbone_result_folder) > 0, "No backbones found!"
    eval_configs = configs_from_grid(eval_config_grid)  

    n = len(backbone_results)*len(eval_configs)
    i = 1
    for bbr in backbone_results:
        for cfg in eval_configs:
            print(i, '/', n)
            evaluate_backbone_with_scorer(cfg, 
                                        bbr, 
                                        output_folder, 
                                        device=device)
            i += 1
            gc.collect()