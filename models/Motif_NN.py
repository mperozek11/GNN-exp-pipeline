import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(48, 24)
        self.b1 = nn.BatchNorm1d(24)
        self.fc2 = nn.Linear(24, 12)
        self.b2 = nn.BatchNorm1d(12)
        self.fc3 = nn.Linear(12, 6)
        self.b3 = nn.BatchNorm1d(6)
        self.fc4 = nn.Linear(6, 3)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = F.relu(self.fc3(x))
        x = self.b3(x)
        x = F.relu(self.fc4(x))
        return x
