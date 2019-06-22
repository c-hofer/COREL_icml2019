import os
import json
import glob
import pandas as pd
import numpy as np

from collections import defaultdict

key_2_col_name = {
    ('train_args', 'learning_rate'):'learning_rate',
    ('train_args', 'batch_size'):'batch_size',
    ('train_args', 'rec_loss_w'):'rec_loss_w',
    ('train_args', 'top_loss_w'):'top_loss_w',
    ('model_args', 'kwargs', 'filter_config'):'filter_config',
    ('model_args', 'kwargs', 'latent_config', 'n_branches'):'n_branches',
    ('model_args', 'kwargs', 'latent_config', 'out_features_branch'):'out_features_branch',
    ('data_args', 'dataset'):'bb_dataset',
    ('data_args', 'subset_ratio'):'bb_subset_ratio',
    ('dataset',):'ev_dataset',
    ('n_train_samples',):'n_train_samples',
    ('scorer', 'id'):'scorer_id',
    ('scorer', 'kwargs', 'aggregation'):'scorer_aggregation',
    ('scorer', 'kwargs', 'truncation_radius'):'scorer_truncation_radius',
    ('n_runs',):'n_runs',
}


def get_keychain_value(d, key_chain=None):
    key_chain = [] if key_chain is None else list(key_chain).copy()       
    
    if not isinstance(d, dict):
        
        yield tuple(key_chain), d
    else:
        for k, v in d.items():
            yield from get_keychain_value(v, key_chain + [k])


def get_auc_frame(result):
    key_val = list(get_keychain_value(result['backbone_config']))
    key_val += list(get_keychain_value(result['eval_config']))
    
    columns = {key_2_col_name[k]: [v if not isinstance(v, list) else tuple(v)] 
               for k, v in key_val if k in key_2_col_name}
    scores = result['scores_by_run']
    
    auc_scores_by_label = defaultdict(list)
    for run_i, all_scores in enumerate(scores):        
      
        for label, score in enumerate(all_scores['auc_scores']):

            auc_scores_by_label[label].append(score/100.0)

    rows = []
    for label, auc_scores in auc_scores_by_label.items():
        
            
        r = columns.copy()
        r['score'] = ['auc_score']
        r['label'] = label
        assert len(auc_scores) == 5
        r['avg'] = [np.mean(auc_scores)]
        r['std'] = [np.std(auc_scores)]

        rows.append(r)
            
    rows = [pd.DataFrame.from_dict(r) for r in rows]
                
    return pd.concat(rows)


def read_pandas_frame_from_results_folder(path):
    files = glob.glob(os.path.join(path,'*.eval.json'))
    results = []
    for f in files:
        with open(f, 'r') as fid:
            results.append(json.load(fid))
    return pd.concat([get_auc_frame(r) for r in results])

