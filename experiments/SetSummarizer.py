from cProfile import label
from operator import indexOf
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt

class SetSummarizer:

    def __init__(self, RESULT_DIR, file_key='result'):
        self.RESULT_DIR = RESULT_DIR
        self.file_key = file_key
        self.get_results()
        self.summary = self.summary_full_result = None
        
    def get_results(self):
        self.results = {}
        files = Path(self.RESULT_DIR).glob('*')
        for f in files:
            if self.file_key in f.stem:
                with open(f) as file:
                    config_res = yaml.safe_load(file)
                self.results[f] = config_res

    def summarize(self, out_dir=None):
        if self.summary != None and self.summary_full_result != None:
            return self.summary, self.summary_full_result
        summary = {} # just f1 and acc 
        summary_full_result = {} # full results including loss data
        n_runs = len(self.results)

        scores, file_keys = self.__get_scores()

        if n_runs > 10:
            x = np.argsort(scores[:,0])[::-1][:10] # get the 10 best models by f1 score
            top_10 = {}
            for idx in x:
                top_10[str(idx)] = f'mean_f1: {scores[idx,0]} mean_acc: {scores[idx,1]}'
                summary_full_result[file_keys.index(scores[idx,2])] = self.results[file_keys.index(scores[idx,2])] ## NEW
            summary['top_10_architectures'] = top_10
        else:
            x = np.argsort(scores[:,0])[::-1]
            all = {}
            for idx in x:
                all[str(idx)] = f'mean_f1: {scores[idx,0]} mean_acc: {scores[idx,1]}'
            summary['sorted_scores'] = all
            summary_full_result = self.results

        if out_dir != None:
            with open(f'{out_dir}/exp_overview.yaml', 'w') as file:
                yaml.dump(summary, file)
        self.summary = summary
        self.summary_full_result = summary_full_result
        return summary, summary_full_result

    def __get_scores(self):
        scores = np.empty((len(self.results), 3))
        i = 0
        keys = [item[0] for item in self.results.items()]
        for k, v in self.results.items():
            scores[i, 0] = v['training_metrics']['mean_f1_accuracy']
            scores[i, 1] = v['training_metrics']['mean_eval_accuracy']
            scores[i, 2] = keys.index(k)
            i+=1
        return scores, keys

    def visualize(self):
        self.loss_data = {}
        # get all the training and validation losses from the result files
        # build internal datastructure that can be used to generate training and validation loss graphs for each config
        _, summary_result = self.summarize()
        for k, v in summary_result.items():
            config_losses = {}
            for k_i, v_i in v.items():
                if 'loss_records_fold' in k_i:
                    config_losses[k_i] = v_i
            self.loss_data[k.stem] = config_losses
        
        self.__plot_loss()

    def __plot_loss(self):
        c_keys = list(self.loss_data.keys())
        fig, axis = plt.subplots(len(c_keys), len(self.loss_data[c_keys[0]].keys()))
        rows = []
        for i in range(len(c_keys)): # configs
            rows.append(c_keys[i][:-6])
            f_keys = list(self.loss_data[c_keys[i]])
            cols = []
            for j in range(len(f_keys)): # folds
                cols.append(f_keys[j][-5:])
                train_loss = self.convert_loss_to_stepped(self.loss_data[c_keys[i]][f_keys[j]]['train_losses'])
                val_loss = self.convert_loss_to_stepped(self.loss_data[c_keys[i]][f_keys[j]]['validation_losses'])
                axis[i, j].tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    right=False,
                    left=False,
                    labelleft=False)
                axis[i, j].plot(np.arange(len(train_loss)), train_loss, label='training loss')
                axis[i, j].plot(np.arange(len(val_loss)), val_loss, label='validation loss')
                # axis[i, j].legend()


        for ax, col in zip(axis[0], cols):
            ax.set_title(col)

        for ax, row in zip(axis[:,0], rows):
            ax.set_ylabel(row, rotation=0, size='large')

        # fig.tight_layout()
        plt.legend()
        plt.show()

    # smoothes list of loss values 
    def convert_loss_to_stepped(self, losses):
        min = losses[0]
        converted = [losses[0]]

        for i in range(len(losses) -1):
            min = losses[i+1] if losses[i+1] < min else min
            converted.append(min)

        return converted
