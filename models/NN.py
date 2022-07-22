import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

import random

import yaml
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_networkx

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.b1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.b2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.b3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, 3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = F.relu(self.fc3(x))
        x = self.b3(x)
        x = F.relu(self.fc4(x))
        return x
    

   