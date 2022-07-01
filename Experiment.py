import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import yaml
import random

from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from tensorflow import keras

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from transforms.wico_transforms import WICOTransforms
from models.GIN import GIN


DATASETS = {
    'wico': '/Users/maxperozek/GNN-research/GNN-exp-pipeline/data/full_wico.pt'
}
    
class Experiment:
    
    DATA_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/data/'
    TRANSFORMS_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/transforms/'

    def __init__(self, config, out_path):
        self.device = torch.device(config['device'])
        self.logger = ExperimentLogger(config, out_path)
        self.config = config
        
        torch.manual_seed(11)
        np.random.seed(11)
        # need to set random random seed as well
        
    def run(self):
        
        dataset, kfold = self.prep_data()
        in_dim, target_dim = self.get_dimensions(dataset)
        self.model = self.config_model(in_dim, target_dim)
        self.optimizer = self.config_optim()
        self.loss_fn = self.config_loss()
        
        fold_eval_acc = np.empty(kfold.get_n_splits())
        fold_eval_f1 = np.empty(kfold.get_n_splits())

        pbar_kfold = tqdm(total = kfold.get_n_splits(), 
                          desc=f"training model {self.config['model']} on {self.config['dataset']} over {self.config['kfolds']} folds",
                          position=0, leave=True)
        start = datetime.now()
        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):

            # DataLoader code will work for either a list of torch.geometric Data objects or an arbitrary torch Tensor
            train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=self.config['batch_size'], shuffle=True)
            test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=len(test_idx), shuffle=True) # single batch for test data since wico fits in memory

            self.train_model(train_loader)
            fold_acc, fold_f1 = self.eval_model(test_loader)

            fold_eval_acc[fold] = fold_acc
            fold_eval_f1[fold] = fold_f1

            pbar_kfold.update(1)
            
        mean_acc = float(fold_eval_acc.mean())
        runtime = str(datetime.now() - start)
        self.logger.log_train(fold_eval_acc, fold_eval_f1, mean_acc, runtime)
        self.logger.dump_log()
        
    
    # returned dataset must be of the type list[Data] or torch Tensor with shape (num graphs, num features)
    def prep_data(self):
        
        # load dataset
        dataset = torch.load(DATASETS[self.config['dataset']])
        random.shuffle(dataset)
        # apply transform
        if self.config['transform']:
            # transform dataset
            t = getattr(WICOTransforms, self.config['transform'])
            dataset = t(dataset)
        
        
        # calculate class weights NOTE this could also be done on a per batch basis. It is worth trying that as an alternative eventually...
        self.class_weights = None
        if self.config['class_weights'] == True:
            y = np.array([data.y for data in dataset]).astype(int)
            classes = np.unique(np.array(y))
            self.class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)
            
        # kfold
        kfold = KFold(n_splits=self.config['kfolds'], shuffle=True)

        return dataset, kfold
    
    def get_dimensions(self, dataset):
        if type(dataset) == torch.Tensor:
            return 3,2
        elif type(dataset) == list:
            in_dim = dataset[0].x.shape[1]
            out_dim = len(np.unique(np.array([data.y for data in dataset]).astype(int)))
            return in_dim, out_dim
        
    # in_dim is the number of features on input dimension. This should be # features per graph for conventional NN and # node features for GNN
    # target dim is the output dimension of the model (2 for binary classification)
    def config_model(self, in_dim, target_dim):
        if self.config['model'] == 'GIN':
            GIN_config = {
                'hidden_units': self.config['hidden_units'],
                'train_eps': self.config['train_eps'],
                'aggregation': self.config['aggregation'],
                'dropout': self.config['dropout']
            }
            model = GIN(dim_features=in_dim, dim_target=target_dim, config=GIN_config)
        else: return None
        return model
    
    def config_optim(self):
        util = ExperimentUtils(self)
        return util.OPTIMIZERS[self.config['optimizer']]
    def config_loss(self):
        util = ExperimentUtils(self)
        return util.LOSS_FUNCTIONS[self.config['loss_fn']]

    def train_model(self, train_loader):
        epochs = self.config['epochs']
        
        
        for e in tqdm(range(epochs), position=0, leave=True):
            self.model.train()
            for _, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                model_out = self.model(batch.x.float(), batch.edge_index, batch.batch)

                y = keras.utils.to_categorical(batch.y, 2)
                loss = self.loss_fn(model_out, torch.Tensor(y))
                loss.backward()
                self.optimizer.step()
        
    def eval_model(self, test_loader):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                out = self.model(data.x.float(), data.edge_index, data.batch)
                cor_list = (torch.argmax(out, dim=1).numpy() == data.y.numpy())
                correct += cor_list.sum()
                f1 = f1_score(data.y.numpy(), torch.argmax(out, dim=1).numpy())
                
            acc = correct / len(test_loader.dataset)
        # This needs to be fixed to work on large validation datasets which need to be tested in batches. 
        # Code will fail when val set is too large for memory per DataLoader initialization so all reported f1 scores will be correct
        return acc, f1
        
        
class ExperimentUtils:
    
    def __init__(self, experiment):

        self.OPTIMIZERS = {
            'Adam': torch.optim.Adam(experiment.model.parameters(), lr=experiment.config['lr'])
        }

        self.LOSS_FUNCTIONS = {
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss(weight=experiment.class_weights)
        }
        
class ExperimentLogger():
    
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.log = { "experiment_run_start": str(datetime.now()) } 
        self.log['config'] = config
        
    def log_train(self, fold_eval_acc, fold_eval_f1, mean_acc, runtime):
        training = {
            'total_train_time': runtime,
            'mean_eval_accuracy': mean_acc,
            'fold_eval_accs': str(list(fold_eval_acc)),
            'fold_eval_f1': str(list(fold_eval_f1))
        }
        self.log['training_metrics'] = training
        
    def dump_log(self):
        with open(self.output_dir + '_result.yaml', 'w') as file:
            yaml.dump(self.log, file)
        print('log saved to: ', file)
    