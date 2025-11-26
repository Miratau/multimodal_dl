import torch
from torch import nn


class FusionMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        layers = []
        dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

