import yaml
import os
import sys
import numpy as np

sys.path.insert(0, '/Users/maxperozek/GNN-research/GNN-exp-pipeline/experiments')
from Experiment import Experiment, ExperimentLogger
sys.path.insert(0, '/Users/maxperozek/GNN-research/GNN-exp-pipeline/transforms')
from wico_transforms import WICOTransforms

import torch
from torch_geometric.data import Data
from collections.abc import Iterable

# ========================================================================================
# =================================== GENERAL TESTS ======================================
# ========================================================================================


def test_experiment_class_init():
    RESULT_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/result/'
    config_file = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/config/test_config.yml'
    config_name = os.path.basename(config_file)[:-5]
    with open(config_file) as file:
        config = yaml.safe_load(file)

    exp = Experiment(config, RESULT_DIR + config_name)
    assert exp
    assert exp.device == torch.device('cpu')

def test_experiment_run_e2e():
    RESULT_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/result/'
    config_file = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/config/test_config.yml'
    config_name = os.path.basename(config_file)[:-4]
    with open(config_file) as file:
        config = yaml.safe_load(file)

    exp = Experiment(config, RESULT_DIR + config_name)
    exp.run()


# ========================================================================================
# ===================================== DATA TESTS =======================================
# ========================================================================================


def test_experiment_prep_data():
    RESULT_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/result/'
    config_file = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/config/test_config.yml'
    config_name = os.path.basename(config_file)[:-5]
    with open(config_file) as file:
        config = yaml.safe_load(file)

    exp = Experiment(config, RESULT_DIR + config_name)
    dataset, kfold = exp.prep_data()
    assert len(dataset) == 2914
    assert type(dataset[0]) == Data
    assert isinstance(dataset, Iterable)
    assert kfold.get_n_splits() == 2

def test_wico_5g_vs_non_conspiracy_transform():
    t = getattr(WICOTransforms, 'wico_5g_vs_non_conspiracy')
    DATA_DIR = '/Users/maxperozek/GNN-research/data_pro/data/'
    full_wico_pyg = 'full_wico.pt'
    wico = torch.load(DATA_DIR + full_wico_pyg)
    wico_2_class = t(wico)
    
    assert len(wico) == 3511
    assert len(wico_2_class) == 2914
    assert (np.unique(np.array(wico_2_class, dtype=object)[:,2,1].astype(int)) == np.arange(2)).all() # assert there are exactly 2 classes; 0 and 1

def test_torch_dummy_transform():
    t = getattr(WICOTransforms, 'torch_dummy_transform')
    DATA_DIR = '/Users/maxperozek/GNN-research/data_pro/data/'
    full_wico_pyg = 'full_wico.pt'
    wico = torch.load(DATA_DIR + full_wico_pyg)
    X, y = t(wico)
    
    print(X)
    assert len(wico) == 3511
    assert X.shape[0] == 3511
    assert X.shape[1] == 10

def test_non_geometric_dataset_handling():
    RESULT_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/result/'
    config_file = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/config/non_geo_test.yml'
    config_name = os.path.basename(config_file)[:-5]
    with open(config_file) as file:
        config = yaml.safe_load(file)

    exp = Experiment(config, RESULT_DIR + config_name)
    mean_f1, mean_acc = exp.run()


def main(argv):
    test_non_geometric_dataset_handling()

    return
    # general
    test_experiment_class_init()
    test_experiment_run_e2e()
    
    # data
    test_experiment_prep_data()
    test_wico_5g_vs_non_conspiracy_transform()
    test_torch_dummy_transform()

    # model

    # random


if __name__ == "__main__":
    main(sys.argv[1:])