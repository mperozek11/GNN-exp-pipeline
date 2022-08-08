import sys, getopt, yaml, os
import numpy as np
from datetime import datetime
from Experiment import Experiment
from ExperimentBuilder import ExperimentBuilder

def main(argv):

    try:
        opts, args = getopt.getopt(argv, 'hc:o:')
    except getopt.GetoptError:
        print('invalid args')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print(f'\nThis tool generates config.yml files for every permutation of a top-level config file')
            print('\nRequired flags: \n -c absolute path of top level config file\n -o absolute path of output directory for config files and filelist\n')
            sys.exit(0)
        if opt == '-c':
            print(f'config file: {arg}')
            config_file = arg
        if opt == '-o':
            print(f'output directory: {arg}')
            output_config_dir = arg
            
    try:
        with open(config_file) as file:
            config = yaml.safe_load(file)
    except Exception:
        print(f'{config_file} not a valid config file')
        sys.exit(2)
        
    names_file, exp_out_dir, n_configs = setup_experiment(config, output_config_dir)
    if not os.path.exists(exp_out_dir):
        os.makedirs(exp_out_dir)
    print(f'\nExperiment setup complete! {n_configs} configs generated.\nTo run experiment using GNU parallel run:\n\n    parallel "python GNN-exp-pipeline/experiments/exp_runner.py -c{{1}} -o {exp_out_dir}" :::: {names_file}\n')

    sys.exit(0) 

def setup_experiment(config, output_config_dir):
    exp_builder = ExperimentBuilder(config)
    configs = exp_builder.get_configs()
    start = datetime.now()
    dt_string = start.strftime('%m.%d.%Y_%H.%M.%S')
    out_root = f'{output_config_dir}/experiment_{dt_string}'

    if not os.path.exists(out_root):
        os.makedirs(out_root)
    

    with open(f'{out_root}/filenames.txt', 'w') as f:
        i = 0
        for c in configs:
            # create new config file and add it to a new line of a text file
            with open(f'{out_root}/config_{i}.yml', 'w') as saved_config:
                yaml.dump(c, saved_config)
            f.write(f'{out_root}/config_{i}.yml\n')
            i += 1

    return f'{out_root}/filenames.txt', f'GNN-exp-pipeline/result/{dt_string}', len(configs)

        
    # save text file to output dir 
    # print text file name and command to run with GNU parallel

if __name__ == "__main__":
    main(sys.argv[1:])

