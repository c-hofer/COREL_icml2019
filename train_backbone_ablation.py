import os
import sys
import json
import datetime
import config
import itertools
from collections import defaultdict

from core.train_engine import check_config, train, configs_from_grid

ROOT_DIR = config.paths.ablation_bkb_dir

if __name__ == "__main__":
    
    grid_many_branches = \
    {
        'train_args': {
            'learning_rate': [0.001], 
            'batch_size'   : [100], 
            'n_epochs'     : [50], 
            'rec_loss_w'   : [1.0],
            'top_loss_w'   : [0.0, 1.0, 10.0, 20.0, 40.0]
        }, 
        'model_args': {
            'class_id'     : ['DCGEncDec'],
            'kwargs'       : {
                'filter_config' : [[3,32,64,128]],
                'input_config'  : [[3,32,32]],
                'latent_config' : {
                    'n_branches'         : [32,16,8], 
                    'out_features_branch': [20,10,5]
                }
            }
        }, 
        'data_args':{
            'dataset'     : ['cifar100'], 
            'subset_ratio': [1.0], 
            'train'       : [True]        
        }    
    }

    grid_one_branch = \
    {
        'train_args': {
            'learning_rate': [0.001], 
            'batch_size'   : [100], 
            'n_epochs'     : [50], 
            'rec_loss_w'   : [1.0],
            'top_loss_w'   : [0.0, 1.0, 10.0, 20.0, 40.0]
        }, 
        'model_args': {
            'class_id'     : ['DCGEncDec'],
            'kwargs'       : {
                'filter_config' : [[3,32,64,128]],
                'input_config'  : [[3,32,32]],
                'latent_config' : {
                    'n_branches'         : [1], 
                    'out_features_branch': [160]
                }
            }
        }, 
        'data_args':{
            'dataset'     : ['cifar100'], 
            'subset_ratio': [1.0], 
            'train'       : [True]        
        }    
    }

    grids = {'grid_one_branch': grid_one_branch, 'grid_many_branches': grid_many_branches}

    now = datetime.datetime.now()
    # path = os.path.join(ROOT_DIR, now.strftime("%Y-%m-%d-%H-%M-%S"))
    path = ROOT_DIR
    os.makedirs(path)

    configs = []
    for k, v in grids.items():
        configs += configs_from_grid(v)

        with open(os.path.join(path, k + '.json'), 'w') as fid:
            json.dump(v, fid)

    for i,c in enumerate(configs):  
        print(c)
        print('Config {}/{}'.format(i+1,len(configs)))
        train(path, c)
