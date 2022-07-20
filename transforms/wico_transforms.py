import numpy as np
import random
import torch
from torch_geometric.data import Data


class WICOTransforms:
    
    def wico_5g_vs_non_conspiracy(wico):
        wico = [g for g in wico if g.y != 2]
        for i in range(len(wico)):
            wico[i] = wico_data_to_custom(wico[i], MultiTargetData)
        return wico
    
    def wico_5g_vs_non_conspiracy_downsampled_balanced(wico):
        ones = [data for data in wico if data.y == 1]
        zeros = [data for data in wico if data.y == 0]

        return ones + zeros[:412]
    
    def wico_5g_non_oversampled(wico):
        ones = [data for data in wico if data.y == 1]
        zeros = [data for data in wico if data.y == 0]

        random.shuffle(ones)
        random.shuffle(zeros)

        idxs = np.random.choice(np.arange(len(ones)), len(zeros), replace=True)
        ones = [ones[i] for i in idxs]

        wico = ones + zeros
        random.shuffle(wico)
        
        return wico

    def torch_dummy_transform(wico):
        n_features = 10
        dataset = []

        for data in wico:
            dataset.append([torch.Tensor(np.random.random((n_features))), data.y])

        return dataset

    def wico_data_to_custom(data, custom_class):
        return custom_class(x=data.x, edge_index=data.edge_index, y=data.y)

class MultiTargetData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'y':
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)
