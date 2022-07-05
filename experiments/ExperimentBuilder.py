import itertools

class ExperimentBuilder:


    def __init__(self, top_level):
        self.config_range = []
        for key, value in top_level.items():
            if(type(value) != list):
                self.config_range.append([value])
            else:
                self.config_range.append(value)


    def get_configs(self):
        all_perms = list(itertools.product(*self.config_range))
        all_configs = []
        for perm in all_perms:
            all_configs.append({
                'device': perm[0],
                'dataset': perm[1],
                'transform': perm[2],
                'class_weights': perm[3],
                'model': perm[4],
                'hidden_units': perm[5],
                'train_eps': perm[6],
                'aggregation': perm[7],
                'dropout': perm[8],
                'kfolds': perm[9],
                'optimizer': perm[10],
                'loss_fn': perm[11],
                'batch_size': perm[12],
                'epochs': perm[13],
                'lr': perm[14]
                })
        return all_configs

