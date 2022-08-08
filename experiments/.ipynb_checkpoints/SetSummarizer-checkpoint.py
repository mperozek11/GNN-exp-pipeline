from operator import indexOf
from pathlib import Path
import yaml
import numpy as np

class SetSummarizer:

    def __init__(self, RESULT_DIR, file_key='result'):
        self.RESULT_DIR = RESULT_DIR
        print(RESULT_DIR)
        self.file_key = file_key
        self.get_results()
        print(self.results)
        
    def get_results(self):
        self.results = {}
        files = Path(self.RESULT_DIR).glob('*')
        for f in files:
            print(f.stem)
            if self.file_key in f.stem:
                with open(f) as config_f:
                    config = yaml.safe_load(config_f)
                self.results[f] = config

    def summarize(self, out_dir=None):
        summary = {}
        n_runs = len(self.results)

        scores, file_keys = self.__get_scores()

        if n_runs > 10:
            x = np.argsort(scores[:,0])[::-1][:10] # get the 10 best models by f1 score
            top_10 = {}
            for idx in x:
                top_10[str(idx)] = f'mean_f1: {scores[idx,0]} mean_acc: {scores[idx,1]}'
            summary['top_10_architectures'] = top_10
        else:
            x = np.argsort(scores[:,0])[::-1]
            all = {}
            for idx in x:
                all[str(idx)] = f'mean_f1: {scores[idx,0]} mean_acc: {scores[idx,1]}'
            summary['sorted_scores'] = all

        if out_dir != None:
            with open(f'{out_dir}/exp_overview.yaml', 'w') as file:
                yaml.dump(summary, file)
        return summary

    def __get_scores(self):
        scores = np.empty((len(self.results), 3))
        i = 0
        keys = [item[0] for item in self.results.items()]
        print(keys)
        for k, v in self.results.items():
            scores[i, 0] = v['training_metrics']['mean_f1_accuracy']
            scores[i, 1] = v['training_metrics']['mean_eval_accuracy']
            scores[i, 2] = keys.index(k)
            i+=1
        return scores, keys

    def visualize(self):
        self.loss_data = {}
        # get all the training and validation losses from the result files
        for k, v in self.results.items():
            if 'loss_records_fold' in k:
                self.loss_data[k] = v
        # build internal datastructure that can be used to generate training and validation loss graphs for each config
        print(self.loss_data) # DEBUG


# summy = SetSummarizer('/Users/maxperozek/GNN-research/GNN-exp-pipeline/result/08.03.2022_12.53.57')
# summary = summy.summarize()
# print(summary)
