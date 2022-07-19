from cProfile import label
import os
import pathlib
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

import sys

root = pathlib.Path().resolve().as_posix()
sys.path.insert(0, f'{root}/GNN-exp-pipeline/transforms')
from wico_transforms import WICOTransforms
sys.path.insert(0, f'{root}/GNN-exp-pipeline/models')
from GIN import GIN
from TorchDummy import TorchDummy


DATASETS = {
    'wico': f'{root}/GNN-exp-pipeline/data/full_wico.pt'
}
    
class Experiment:
    
    DATA_DIR = f'{root}/GNN-exp-pipeline/data/'
    TRANSFORMS_DIR = f'{root}/GNN-exp-pipeline/transforms/'

    def __init__(self, config, out_path):
        self.device = torch.device(config['device'])
        print(f'torch device: {self.device}')
        self.logger = ExperimentLogger(config, out_path)
        self.config = config
        
        torch.manual_seed(11)
        np.random.seed(11)
        # need to set random random seed as well
        
    def run(self):
        
        dataset, kfold = self.prep_data()
        self.in_dim, self.target_dim = self.get_dimensions(dataset)
        for i in range(len(dataset)): # DEBUG
            dataset[i] = dataset[i].cuda() # DEBUG
            
        self.model = self.config_model()
        self.model = self.model.cuda() # test
        
        self.optimizer = self.config_optim()
        
        self.loss_fn = self.config_loss()
        self.loss_fn = self.loss_fn.cuda()
        
        fold_eval_acc = np.empty(kfold.get_n_splits())
        fold_eval_f1 = np.empty(kfold.get_n_splits())

        start = datetime.now()
        state_dicts = []
        optim_dicts = []
        train_message_dict = {}
        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
            
            self.best_model = None # this will be the best model per training fold
            # DataLoader code will work for either a list of torch.geometric Data objects or an arbitrary torch Tensor
            train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=self.config['batch_size'], shuffle=True)
            test_loader = DataLoader([dataset[i] for i in test_idx], batch_size=len(test_idx), shuffle=True) # single batch for test data since wico fits in memory

            train_message, epochs_trained = self.train_model(train_loader, test_loader, patience=int(self.config['patience']))
            fold_acc, fold_f1 = self.eval_model(test_loader)

            train_message_dict[f'fold {fold} training message'] = train_message
            fold_eval_acc[fold] = fold_acc
            fold_eval_f1[fold] = fold_f1
            state_dicts.append(self.best_model['model_state_dict'])
            optim_dicts.append(self.best_model['optim_state_dict'])

        mean_f1 = float(fold_eval_f1.mean())
        mean_acc = float(fold_eval_acc.mean())
        runtime = str(datetime.now() - start)
        self.logger.log_train(fold_eval_acc, fold_eval_f1, mean_acc, runtime, state_dicts, optim_dicts, train_message_dict)
        self.logger.dump_log()
        return mean_f1, mean_acc
        
    
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
            y = ExperimentUtils.get_y(dataset)
            classes = np.unique(np.array(y))

            # below are two different weighting schemes which yeild the same proportions but use different strategies

            # === 1 ===
            # prevs = []
            # for c in classes:
            #     prev = len((y == c).nonzero()[0])
            #     prevs.append(prev)
            # most_ex = sorted(prevs)[-1] # get most prevelant class number of examples
            # self.class_weights = most_ex / torch.Tensor(prevs)

            # === 2 ===
            self.class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float)
            
            
        # kfold
        kfold = KFold(n_splits=self.config['kfolds'], shuffle=True)

        return dataset, kfold
    
    def get_dimensions(self, dataset):
        if type(dataset[0]) == list:
            in_dim = len(dataset[0][0])
            out_dim = len(np.unique(np.array([item[1] for item in dataset]).astype(int)))
            return in_dim, out_dim
        else:
            in_dim = dataset[0].x.shape[1]
            out_dim = len(np.unique(np.array([data.y for data in dataset]).astype(int)))
            return in_dim, out_dim
        
    # in_dim is the number of features on input dimension. This should be # features per graph for conventional NN and # node features for GNN
    # target dim is the output dimension of the model (2 for binary classification)
    def config_model(self):
        if self.config['model'] == 'GIN':
            GIN_config = {
                'hidden_units': self.config['hidden_units'],
                'train_eps': self.config['train_eps'],
                'aggregation': self.config['aggregation'],
                'dropout': self.config['dropout']
            }
            model = GIN(dim_features=self.in_dim, dim_target=self.target_dim, config=GIN_config)
        elif self.config['model'] == 'TorchDummy':
            model = TorchDummy(self.in_dim, self.target_dim)
        else:
            raise Exception(f"model defined in config ({self.config['model']}) has not been implemented")
        
        return model
    
    def config_optim(self):
        util = ExperimentUtils(self)
        return util.OPTIMIZERS[self.config['optimizer']]
    def config_loss(self):
        util = ExperimentUtils(self)
        return util.LOSS_FUNCTIONS[self.config['loss_fn']]

    # trains and evaluates the model each epoch. 
    # The running best model is stored in self.best_model on the criteria of lowest validation loss over training.
    def train_model(self, train_loader, eval_loader, patience=5):
        epochs = self.config['epochs']
        
        val_losses = []
        for e in tqdm(range(epochs), position=0, leave=True):
            self.model.train()
            for _, batch in enumerate(train_loader):
                self.optimizer.zero_grad()

                if type(batch) == list: # non-geometric case 
                    model_out = self.model(batch[0].float())
                    y = torch.Tensor(keras.utils.to_categorical(batch[1], self.target_dim))

                else:
                    model_out = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    y = torch.Tensor(keras.utils.to_categorical(torch.clone(batch.y).cpu(), self.target_dim))
                
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = self.loss_fn(model_out, y)
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            total_loss = 0
            for _, batch in enumerate(eval_loader):
                if type(batch) == list: # non-geometric case 
                    model_out = self.model(batch[0].float())
                    y = torch.Tensor(keras.utils.to_categorical(batch[1], self.target_dim))

                else:
                    model_out = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    y = torch.Tensor(keras.utils.to_categorical(torch.clone(batch.y).cpu(), self.target_dim))
                
                if torch.cuda.is_available():
                    y = y.cuda()
                loss = self.loss_fn(model_out, y)
                total_loss += loss.item()
            
            if self.best_model == None or total_loss < self.best_model['total_eval_loss']:
                self.best_model = {}
                self.best_model['model_state_dict'] = self.model.state_dict()
                self.best_model['optim_state_dict'] = self.optimizer.state_dict()
                self.best_model['total_eval_loss'] = total_loss
            
            val_losses.append(total_loss)

            if len(val_losses) > patience and self.not_improving(val_losses[-patience:], epsilon=self.config['improvement_threshold']):
                return f'stopped training due to stagnating improvement on validation loss after {e+1} epochs', e+1

        return f'completed {e+1} epochs without stopping early', e+1

    def not_improving(self, last_n_losses, epsilon=0.01):
        for i in range(len(last_n_losses[:-1])):
            if last_n_losses[i] + epsilon < last_n_losses[i+1]:
                return False
        return True
        
    # Tests the best model per fold. Best model determined by model with lowest eval loss during training
    def eval_model(self, test_loader):
        
        self.model.load_state_dict(self.best_model['model_state_dict'])
        self.model.eval()
        correct = 0
        total = 0
        average = 'binary' if self.target_dim == 2 else 'micro' # sklearn f1 function's averaging scheme should be understood for multiclass
        with torch.no_grad():
            for data in test_loader:
                if type(data) == list:
                    out = self.model(data[0].float())
                    cor_list = (torch.argmax(out, dim=1).numpy() == data[1].numpy()) # np.array of shape (n_test_examples,) where each index corresponds a binary value for correctness of pred
                    correct += cor_list.sum()
                    f1 = f1_score(data[1].numpy(), torch.argmax(out, dim=1).numpy(), average=average)
                    total += len(data[0])
                else:
                    out = self.model(data.x.float(), data.edge_index, data.batch)
                    cor_list = (torch.clone(torch.argmax(out, dim=1)).cpu().numpy() == torch.clone(data.y).cpu().numpy())
                    correct += cor_list.sum()
                    f1 = f1_score(torch.clone(data.y).cpu().numpy(), torch.clone(torch.argmax(out, dim=1)).cpu().numpy(), average=average)
                    total += len(data.y)

                
            acc = correct / total
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
    def get_y(dataset): # I think there is a pythonic way to do this in one line ~~ y = np.array([item[1] for item in dataset]) if type(dataset[0]) == list else np.array([data.y for data in dataset]).astype(int)

        if type(dataset[0]) == list: # X, y case
            return np.array([item[1] for item in dataset]).astype(int)
        else: # geometric Data object case
            return np.array([data.y for data in dataset]).astype(int)

class ExperimentLogger():
    
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.log = { "experiment_run_start": str(datetime.now()) } 
        self.log['config'] = config
        
    def log_train(self, fold_eval_acc, fold_eval_f1, mean_acc, runtime, state_dicts, optim_dicts, train_message_dict):
        training = {
            'total_train_time': runtime,
            'mean_eval_accuracy': mean_acc,
            'fold_eval_accs': str(list(fold_eval_acc)),
            'fold_eval_f1': str(list(fold_eval_f1))
        }
        for i in range(len(state_dicts)):
            model_fname = f'{self.output_dir}fold_{i}_state_dict.pt'
            optim_fname = f'{self.output_dir}fold_{i}_optim_dict.pt'
            torch.save(state_dicts[i], model_fname)
            torch.save(state_dicts[i], optim_fname)

            self.log[f'fold_{i}_state_dict'] = str(model_fname)
            self.log[f'fold_{i}_optim_dict'] = str(optim_fname)

        self.log['training_metrics'] = training
        self.log['training fold messages'] = train_message_dict

        
    def dump_log(self):
        with open(f'{self.output_dir}result.yaml', 'w') as file:
            yaml.dump(self.log, file)
        print('log saved to: ', file)
    