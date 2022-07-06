import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU


class TorchDummy(torch.nn.Module):

    def __init__(self, input_dim, target_dim):
        super().__init__()

        self.lin1 = Linear(input_dim, 8)
        self.lin2 = Linear(8, target_dim)
        self.relu = ReLU()


    def forward(self,x):

        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x