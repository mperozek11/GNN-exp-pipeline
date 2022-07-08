import sys, getopt, yaml, os
import numpy as np
from datetime import datetime
from Experiment import Experiment
from ExperimentBuilder import ExperimentBuilder

def main(argv):

    try:
        opts, args = getopt.getopt(argv, 'c:o:')
    except getopt.GetoptError:
        print('invalid args')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-c':
            print(f'config file: {arg}')
            config_file = arg
        if opt == '-o':
            print(f'output directory: {arg}')
            output_dir = arg
            
    try:
        with open(config_file) as file:
            config = yaml.safe_load(file)
    except Exception:
        print(f'{config_file} not a valid config file')
        sys.exit(2)
        
    run_experiment(config, output_dir)
    
    sys.exit(0)

def run_experiment(config, output_dir):
    exp_builder = ExperimentBuilder(config)
    configs = exp_builder.get_configs()
    start = datetime.now()
    dt_string = start.strftime('%m.%d.%Y_%H.%M.%S')
    out_root = f'{output_dir}/experiment_{dt_string}'

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    i = 0
    results = np.empty((len(configs), 2))
    for c in configs:
        if not os.path.exists(f'{out_root}/{i}'):
            os.makedirs(f'{out_root}/{i}')

        experiment = Experiment(c, f'{out_root}/{i}/')
        meanf1, mean_acc = experiment.run() 
        results[i,:] = (meanf1, mean_acc) 
        i += 1

    total_runtime = str(datetime.now() - start)

    return summarize(total_runtime, results, len(configs), out_root)


def summarize(total_runtime, results, n_runs, out_root):
    summary = {'cumulative runtime': total_runtime}
    summary['total_runs'] = int(n_runs)
    if n_runs > 10:
        x = np.argsort(results[:,0])[::-1][:10] # get the 10 best models by f1 score
        summary['top_10_models'] = list(x)
        for idx in x:
            summary[str(idx)] = f'mean_f1: {results[idx,0]} mean_acc: {results[idx,1]}'
    else:
        for idx in range(results.shape[0]):
            summary[str(idx)] = f'mean_f1: {results[idx,0]} mean_acc: {results[idx,1]}'

    with open(f'{out_root}/exp_overview.yaml', 'w') as file:
        yaml.dump(summary, file)
    return summary

if __name__ == "__main__":
    main(sys.argv[1:])

