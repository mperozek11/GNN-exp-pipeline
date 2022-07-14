import numpy as np
import random
import torch

class WICOTransforms:
    
    def wico_5g_vs_non_conspiracy(wico):
        return [g for g in wico if g.y != 2]
    
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
