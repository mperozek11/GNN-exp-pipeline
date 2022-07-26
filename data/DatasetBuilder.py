from ast import arg
import sys
import pathlib
root = pathlib.Path().resolve().as_posix()
# import all datasets here
sys.path.insert(0, f'{root}/GNN-exp-pipeline/data')
from WICO import WICO, filter_5g_non, wico_data_to_custom


class DatasetBuilder:

    def __init__(self, config, dataset_root):
        self.dataset_root = dataset_root
        self.config = config


    def get_dataset(self):
        dataset_class = globals()[self.config['data']['dataset']]
        arg_dict = self.get_args()
        dataset = dataset_class(**arg_dict)
        return dataset

    # returns a dictionary of args, getting function references for filter/transform from globals
    def get_args(self):
        arg_dict = {}
        for k in self.config['data']:
            if k == 'pre_filter' or k == 'pre_transform':
                arg_dict[k] = globals()[self.config['data'][k]]
            elif k == 'root':
                arg_dict[k] = f'{self.dataset_root}{self.config["data"][k]}'
        
        return arg_dict