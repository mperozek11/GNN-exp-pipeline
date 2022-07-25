from cgi import test
import yaml
import os
import glob
import sys
import numpy as np
import pathlib
root = pathlib.Path().resolve().as_posix()

sys.path.insert(0, f'{root}/GNN-exp-pipeline/experiments')
from Experiment import Experiment, ExperimentLogger
from exp_set_runner import run_experiment
sys.path.insert(0, f'{root}/GNN-exp-pipeline/transforms')
from wico_transforms import WICOTransforms

import torch
from torch_geometric.data import Data
from collections.abc import Iterable


# ========================================================================================
# ======================================= PATHS ==========================================
# ========================================================================================

RESULT_DIR = f'{root}/GNN-exp-pipeline/test/result/'
DATA_DIR = f'{root}/GNN-exp-pipeline/data/'

# ========================================================================================
# =================================== GENERAL TESTS ======================================
# ========================================================================================


def test_experiment_class_init():
    config_file = f'{root}/GNN-exp-pipeline/config/test_config.yml'
    config_name = os.path.basename(config_file)[:-5]
    with open(config_file) as file:
        config = yaml.safe_load(file)

    exp = Experiment(config, RESULT_DIR + config_name)
    assert exp
    assert exp.device == torch.device('cpu')

def test_experiment_run_e2e():
    config_file = f'{root}/GNN-exp-pipeline/config/test_config.yml'
    config_name = os.path.basename(config_file)[:-4]
    with open(config_file) as file:
        config = yaml.safe_load(file)

    exp = Experiment(config, RESULT_DIR + config_name)
    exp.run()


# ========================================================================================
# ===================================== DATA TESTS =======================================
# ========================================================================================


def test_experiment_prep_data():
    config_file = f'{root}/GNN-exp-pipeline/config/test_config.yml'
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
    full_wico_pyg = 'full_wico.pt'
    wico = torch.load(DATA_DIR + full_wico_pyg)
    wico_2_class = t(wico)
    
    assert len(wico) == 3511
    assert len(wico_2_class) == 2914
    assert (np.unique(np.array(wico_2_class, dtype=object)[:,2,1].astype(int)) == np.arange(2)).all() # assert there are exactly 2 classes; 0 and 1


def test_wico_5g_non_oversampled_transform():
    t = getattr(WICOTransforms, 'wico_5g_non_oversampled')
    full_wico_pyg = 'full_wico.pt'
    wico = torch.load(DATA_DIR + full_wico_pyg)
    wico_2_class = t(wico)
    
    # print(wico_2_class)
    # print(len(wico_2_class))

    assert len(wico) == 3511
    assert len(wico_2_class) == 5004
    assert (np.unique(np.array(wico_2_class, dtype=object)[:,2,1].astype(int)) == np.arange(2)).all() # assert there are exactly 2 classes; 0 and 1

def test_torch_dummy_transform():
    t = getattr(WICOTransforms, 'torch_dummy_transform')
    full_wico_pyg = 'full_wico.pt'
    wico = torch.load(DATA_DIR + full_wico_pyg)
    dataset = t(wico)
    X = torch.Tensor(np.array([np.array(data[0]) for data in dataset]))
    y = torch.Tensor(np.array([np.array(data[1]) for data in dataset]))

    assert len(wico) == 3511
    assert X.shape[0] == 3511
    assert X.shape[1] == 10
    assert y.shape[0] == 3511

def test_non_geometric_dataset_handling():
    config_file = f'{root}/GNN-exp-pipeline/config/non_geo_test.yml'
    config_name = os.path.basename(config_file)[:-5]
    with open(config_file) as file:
        config = yaml.safe_load(file)

    exp = Experiment(config, RESULT_DIR + config_name)
    mean_f1, mean_acc = exp.run()

# ========================================================================================
# ==================================== TRAINING TESTS ====================================
# ========================================================================================

def test_early_stopping():
    CONFIG_FILE = f'{root}/GNN-exp-pipeline/config/early_stop_test_config.yml'
    RESULT_DIR = f'{root}/GNN-exp-pipeline/result/'

    config_name = os.path.basename(CONFIG_FILE)[:-4]
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    exp = Experiment(config, RESULT_DIR + config_name)
    exp.run()



# ========================================================================================
# ====================================== SET TESTS =======================================
# ========================================================================================


def test_exp_set_runner():
    CONFIG_FILE = f'{root}/GNN-exp-pipeline/config/top_level_small_test.yml'
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    summary = run_experiment(config, RESULT_DIR)
    assert summary['total_runs'] == 2
    assert type(summary['cumulative runtime']) == str
    assert 'mean_f1' in summary['0']
    assert 'mean_acc' in summary['0']
    assert 'mean_f1' in summary['1']
    assert 'mean_acc' in summary['1']

def test_exp_set_runner_more_than_10():
    CONFIG_FILE = f'{root}/GNN-exp-pipeline/config/top_level_exp_test_gr10.yml'
    with open(CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    summary = run_experiment(config, RESULT_DIR)
    
    # assert summary['total_runs'] == 16
    assert type(summary['cumulative runtime']) == str

def delete_unit_test_results(res_dir):
    files = glob.glob(f'{res_dir}/*')
    for f in files:
        os.remove(f)
def main(argv):
    delete_unit_test_results(RESULT_DIR)
    return 0
    test_experiment_run_e2e()
    # general
    test_experiment_class_init()
    test_experiment_run_e2e()
    
    # data
    test_experiment_prep_data()
    test_wico_5g_vs_non_conspiracy_transform()
    test_wico_5g_non_oversampled_transform()
    test_torch_dummy_transform()
    test_non_geometric_dataset_handling()

    # training 
    test_early_stopping()

    # random state

    # top level
    test_exp_set_runner()
    test_exp_set_runner_more_than_10()

    # cleanup 
    delete_unit_test_results(RESULT_DIR)


if __name__ == "__main__":
    main(sys.argv[1:])