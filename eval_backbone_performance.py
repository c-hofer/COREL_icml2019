import os
import sys
import json
import config

from core.eval_engine import evaluate_backbones_on_eval_config_grid
device='cuda'

ROOT_FOLDER = config.paths.performance_bkb_dir
OUT_FOLDER  = config.paths.performance_res_dir


def main():
    
    os.makedirs(OUT_FOLDER)

    eval_config_grid = {
        'dataset':  ['cifar10', 'cifar20', 'cifar100', 'tiny-imagenet-200', 'ImageNet'], 
        'n_train_samples': [80, 120, 500],
        'scorer':
        {
            'id': ['CountScorerProduct'],
            'kwargs': {
                'aggregation': ['sum'], 
                'truncation_radius': [2.0]
            }
        }, 
        'n_runs': [5]
    }  

    with open(os.path.join(OUT_FOLDER, 'eval_grid.json'), 'w') as fid:
        json.dump(eval_config_grid, fid)

    evaluate_backbones_on_eval_config_grid(ROOT_FOLDER, 
                                           eval_config_grid, 
                                           OUT_FOLDER,
                                           device=device)

if __name__ == "__main__":
    main()
