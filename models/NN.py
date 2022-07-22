import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self, dim_features, dim_target, config):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(dim_features, 32)
        self.b1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.b2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.b3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, dim_target)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = F.relu(self.fc3(x))
        x = self.b3(x)
        x = F.relu(self.fc4(x))
        return x
    

   