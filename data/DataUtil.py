import torch
import numpy as np
from tensorflow import keras
from sklearn.utils import class_weight


def target_to_categorical(dataset, target_dim):

    if type(dataset[0]) == list: # traditional NN
        for i in range(len(dataset)):
            dataset[i][1] = torch.tensor(keras.utils.to_categorical(dataset[i][1], target_dim))
    else: # pyg Data class
        for i in range(len(dataset)):
            dataset[i].y = torch.tensor(keras.utils.to_categorical(dataset[i].y, target_dim))

    return dataset

def get_class_weights(calculate, y):
    if calculate:
        classes = np.unique(np.array(y))
            # below are two different weighting schemes which yeild the same proportions but use different strategies

            # === 1 ===
            # prevs = []
            # for c in classes:
            #     prev = len((y == c).nonzero()[0])
            #     prevs.append(prev)
            # most_ex = sorted(prevs)[-1] # get most prevelant class number of examples
            # self.class_weights = most_ex / torch.Tensor(prevs)

            # === 2 ===
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y)
        return torch.tensor(class_weights, dtype=torch.float)
    else:
        return None

def to_device(dataset, device):
    if type(dataset[0]) == list:
        for i in range(len(dataset)):
            dataset[i][0] = dataset[i][0].to(device) # graph features
            dataset[i][1] = dataset[i][1].to(device) # graph labels

    else:
        for i in range(len(dataset)):
            dataset[i] = dataset[i].to(device) # pyg Data object

