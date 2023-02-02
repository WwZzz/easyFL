import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
from utils.fmodule import FModule
from torch.nn import Sequential, Linear, ReLU

class Model(FModule):
    def __init__(self, input_dim=3, hidden_dim=20, output_dim=6):
        super(Model, self).__init__()
        self.num_layers = 3
        self.pre_mp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim))

        self.convs = nn.ModuleList()
        self.nn1 = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(pyg_nn.GINConv(self.nn1))
        for l in range(self.num_layers - 1):
            self.nnk = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(pyg_nn.GINConv(self.nnk))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.pre_mp(x)
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = pyg_nn.global_add_pool(x, batch)
        x = self.post_mp(x)
        x = F.log_softmax(x, dim=1)
        return x
