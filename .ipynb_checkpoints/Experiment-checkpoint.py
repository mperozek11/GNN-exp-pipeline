import os
import numpy as np
from datetime import datetime

from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

DATA_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/data/'
TRANSFORMS_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/transforms/'


class Experiment:
    
    def __init__(self, config, out_path):
        self.device = torch.device(config['device'])
        self.logger = ExperimentLogger(config, out_path)
        self.config = config
        
        torch.manual_seed(11)
        np.random.seed(11)
        
    def run(self):
        
        dataset, kfold = self.prep_data()
        model = 
        optimizer =
        loss_fn = 
        
        
        
        fold_eval_results = np.empty(k_folds)
        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):

            # DataLoader code will work for either a list of torch.geometric Data objects or an arbitrary torch Tensor
            train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=self.config['batch_size'], shuffle=True)
            test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=len(test_idx), shuffle=True) # single batch for test data since wico fits in memory

            self.train_model(model, optimizer, loss_fn, train_loader, epochs)
            fold_acc = self.eval_model(model, test_loader)

            fold_eval_results[fold] = fold_acc
            
        mean_acc = fold_eval_results.mean()
    
    # returned dataset must be of the type list[Data] or torch Tensor with shape (num graphs, num features)
    def prep_data(self):
        
        # load dataset
        dataset = torch.load(DATA_DIR + self.config['dataset'])
        
        # apply transform
        if self.config['transform']:
            # transform dataset
            pass
        
        # kfold
        kfold = KFold(n_splits=self.config['kfolds'], shuffle=True)

        return dataset, kfold
        
        
        
        
def run_model(model, optimizer, loss_fn, batch_size, epochs, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    batch_size = 32

    fold_eval_results = np.empty(k_folds)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(wico)):

        train_loader = DataLoader([wico[i] for i in train_idx], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader([wico[i] for i in test_idx], batch_size=batch_size, shuffle=True)
        
        train_model(model, optimizer, loss_fn, train_loader, epochs)
        fold_acc = eval_model(model, test_loader)
        
        fold_eval_results[fold] = fold_acc
    return fold_eval_results.mean()
        
        
class ExperimentLogger():
    
    def __init__(self, config_file, output_dir):
        self.output_dir = output_dir
        with open(config_file) as file:
            config = yaml.safe_load(file) 
        self.config_name = os.path.basename(config_file)[:-5]
        self.log = { "experiment_run_start": str(datetime.now()) } 
        self.log['config'] = config
    
    def log_model(self, model, optimizer, loss_fn):
        params = {
            'model_class': str(model.__class__.__name__),
            'model_summary': ''.join(s for s in str(model) if ord(s)>31 and ord(s)<126),
            'optimizer_summary': ''.join(s for s in str(optimizer) if ord(s)>31 and ord(s)<126),
            'loss_function': str(loss_fn)
        }
        self.log['experiment_details'] = params
        
    def log_train(self, losses, val_scores, runtime, epochs, final_acc):
        training = {
            'total_train_time': runtime,
            'epochs_trained': epochs,
            'final_test_acc': final_acc,
            'loss_metrics': str(list(losses)),
            'val_metrics': str(list(val_scores))
        }
        self.log['training_metrics'] = training
        
    def dump_log(self):
        with open(self.output_dir + self.config_name + '_result.yaml', 'w') as file:
            yaml.dump(self.log, file)
        print('log saved to: ', file)
    