

RESULT_DIR = '/Users/maxperozek/GNN-research/GNN-exp-pipeline/result/'

with open(config_file) as file:
    config = yaml.safe_load(file)
    
experiment = Experiment(config, RESULT_DIR)
