import os
import sys
import json
import datetime
import itertools
import config
from collections import defaultdict

from core.train_engine import check_config, train, configs_from_grid

ROOT_DIR = config.paths.performance_bkb_dir

if __name__ == "__main__":
    
    grid = {
        'train_args': {
            'learning_rate': [0.001], 
            'batch_size'   : [100], 
            'n_epochs'     : [50], 
            'rec_loss_w'   : [1.0],
            'top_loss_w'   : [20.0]
        }, 
        'model_args': {
            'class_id'     : ['DCGEncDec'],
            'kwargs'       : {
                'filter_config' : [[3,32,64,128]],
                'input_config'  : [[3,32,32]],
                'latent_config' : {
                    'n_branches'         : [16], 
                    'out_features_branch': [10]
                }
            }
        }, 
        'data_args':{
            'dataset'     : ['tiny-imagenet-200', 'cifar10', 'cifar100'], 
            'subset_ratio': [1.0], 
            'train'       : [True]        
        }    
    }

    configs = configs_from_grid(grid)
    path = ROOT_DIR

    # now = datetime.datetime.now()
    # path = os.path.join(ROOT_DIR, now.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(path)

    with open(os.path.join(path, 'grid.json'), 'w') as fid:
        json.dump(grid, fid)


    for i,c in enumerate(configs):  
        print(c)
        print('Config {}/{}'.format(i+1,len(configs)))
        train(path, c)
