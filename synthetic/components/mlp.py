import torch
import torch.nn as nn
from torch.nn import functional as F

class NLayerLeakyMLP(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True, mask=None):
        super().__init__()
        self.mask = mask
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.mask is not None:
            x = x*self.mask
        return self.net(x)