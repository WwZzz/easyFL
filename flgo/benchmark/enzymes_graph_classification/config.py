import torch_geometric
import os
import random
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch
from torch.nn import Sequential, Linear, ReLU

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAW_DATA', 'ENZYMES')
all_data = torch_geometric.datasets.TUDataset(path, name='ENZYMES', use_node_attr=True)
all_idxs = [i for i in range(len(all_data))]
random.seed(1)
random.shuffle(all_idxs)
num_samples = len(all_data)
train_idxs = all_idxs[:int(0.8*num_samples)]
test_idxs = all_idxs[int(0.8*num_samples):]
train_data = all_data[train_idxs]
test_data = all_data[test_idxs]

class GIN(torch.nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, output_dim=6):
        super(GIN, self).__init__()
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

def get_model():
    return GIN()