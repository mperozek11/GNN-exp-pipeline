from cProfile import label
import sys
import pathlib
import numpy as np
from datetime import datetime
from tqdm import tqdm
import yaml

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from tensorflow import keras

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

root = pathlib.Path().resolve().as_posix()
sys.path.insert(0, f'{root}/GNN-exp-pipeline/data')
from DataUtil import target_to_categorical, get_class_weights, to_device
from DatasetBuilder import DatasetBuilder

sys.path.insert(0, f'{root}/GNN-exp-pipeline/transforms')
from wico_transforms import WICOTransforms, MultiTargetData
sys.path.insert(0, f'{root}/GNN-exp-pipeline/models')
from GIN import GIN
from NN import Network
from TorchDummy import TorchDummy

    
class Experiment:
    
    DATA_DIR = f'{root}/GNN-exp-pipeline/data/'
    TRANSFORMS_DIR = f'{root}/GNN-exp-pipeline/transforms/'

    def __init__(self, config, out_path):
        self.device = torch.device(config['device'])
        print(f'torch device: {self.device}')
        print(f'batch size: {config["batch_size"]}')
        self.logger = ExperimentLogger(config, out_path)
        self.config = config
        
        torch.manual_seed(11)
        np.random.seed(11)
        # need to set random random seed as well
        
    def run(self):
        dataset, kfold = self.prep_data()
        self.model = self.config_model()
        
        self.optimizer = self.config_optim()
        self.loss_fn = self.config_loss()
        
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
        
        dataset_builder = DatasetBuilder(self.config, self.DATA_DIR)
        dataset = dataset_builder.get_dataset()
        y = dataset.data.y
        self.class_weights = get_class_weights(self.config['class_weights'], y)
        self.in_dim = dataset.data.x.shape[1]
        self.target_dim = len(np.unique(np.array(dataset.data.y).astype(int)))
        
        # self.in_dim, self.target_dim = self.get_dimensions(dataset) # this call is moving some shit to CPU for some reason

        kfold = KFold(n_splits=self.config['kfolds'], shuffle=True)
        dataset.data = dataset.data.to(self.device)
        
        print(f'dataset: {self.config["dataset"]} {sys.getsizeof(dataset)} bytes in memory')
        return dataset, kfold
    
    def get_dimensions(self, dataset):
#         in_dim = dataset.data.x.shape[1]
#         out_dim = len(np.unique(np.array(dataset.data.y).astype(int)))

#         return in_dim, out_dim
        
        
        if issubclass(type(dataset), torch_geometric.data.Dataset):
            in_dim = dataset.data.x.shape[1]
            out_dim = len(np.unique(np.array(dataset.data.y).astype(int)))
            return in_dim, out_dim
        elif type(dataset[0]) == list:
            in_dim = len(dataset[0][0])
            out_dim = len(np.unique(np.array([item[1] for item in dataset]).astype(int)))
            return in_dim, out_dim
        else: 
            raise RuntimeError('dataset format not recognized')
        
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
        elif self.config['model'] == 'NN':
            model = Network(dim_features=self.in_dim, dim_target=self.target_dim, config=None)
        elif self.config['model'] == 'TorchDummy':
            model = TorchDummy(self.in_dim, self.target_dim)
        else:
            raise Exception(f"model defined in config ({self.config['model']}) has not been implemented")
        
        model = model.to(self.device)
        return model
    
    def config_optim(self):
        util = ExperimentUtils(self)
        optim = util.OPTIMIZERS[self.config['optimizer']]
        return optim
    def config_loss(self):
        util = ExperimentUtils(self)
        loss = util.LOSS_FUNCTIONS[self.config['loss_fn']]
        loss = loss.to(self.device)
        return loss

    # trains and evaluates the model each epoch. 
    # The running best model is stored in self.best_model on the criteria of lowest validation loss over training.
    def train_model(self, train_loader, eval_loader, patience=5):
        epochs = self.config['epochs']
        
        val_losses = []
        for e in tqdm(range(epochs), position=0, leave=True):
            self.model.train()
            for batch in train_loader:
                self.optimizer.zero_grad()

                if type(batch) == list: # non-geometric case 
                    model_out = self.model(batch[0].float())
                    y = batch[1]

                else: # pyg Data
                    # print(f'x: {batch.x.device}, edge_index: {batch.edge_index.device}, batch: {batch.batch.device}')
                    model_out = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    y = batch.y
                
                loss = self.loss_fn(model_out, y)
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            total_loss = 0
            for batch in eval_loader:
                if type(batch) == list: # non-geometric case 
                    model_out = self.model(batch[0].float())
                    y = batch[1]

                else:
                    model_out = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    y = batch.y
                
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
                    correct += float(torch.eq(torch.argmax(out, dim=1), torch.argmax(data[1], dim=1)).sum())
                    f1 = f1_score(torch.clone(torch.argmax(data[1], dim=1)).cpu().numpy(), torch.clone(torch.argmax(out, dim=1)).cpu().numpy(), average=average)
                    total += len(data[0])
                else:
                    out = self.model(data.x.float(), data.edge_index, data.batch)
                    correct += float(torch.eq(torch.argmax(out, dim=1), torch.argmax(data.y, dim=1)).sum())
                    f1 = f1_score(torch.clone(torch.argmax(data.y, dim=1)).cpu().numpy(), torch.clone(torch.argmax(out, dim=1)).cpu().numpy(), average=average)

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
    