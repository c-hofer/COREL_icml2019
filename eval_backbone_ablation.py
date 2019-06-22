import os
import sys
import json
import datetime

import config

from core.eval_engine import evaluate_backbones_on_eval_config_grid
device='cuda'

ABLATION_ROOT_FOLDER = config.paths.ablation_bkb_dir

def main():

    eval_config_grid = {
        'dataset':  ['cifar10'],
        'n_train_samples': [10,50, 5000,80,100,120,250,500],
        'scorer':
        {
            'id': ['CountScorerProduct'],
            'kwargs': {
                'aggregation': ['sum'],
                'truncation_radius': [0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0]
            }
        },
        'n_runs': [5]
    }

    path = config.paths.ablation_res_dir

    # now = datetime.datetime.now()
    # path = os.path.join(path, now.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(path)

    with open(os.path.join(path, 'grid.json'), 'w') as fid:
       json.dump(eval_config_grid, fid)

    evaluate_backbones_on_eval_config_grid(ABLATION_ROOT_FOLDER, 
                                           eval_config_grid, 
                                           path,
                                           device=device)

if __name__ == "__main__":
    main()
