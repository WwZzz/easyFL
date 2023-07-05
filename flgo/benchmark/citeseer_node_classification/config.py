import torch.nn
import flgo.benchmark
import os
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3703, 16)
        self.conv2 = GCNConv(16,6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'CITESEER')
dataset = torch_geometric.datasets.Planetoid(path, name='Citeseer')
train_data = dataset[0]

def get_model():
    return GCN()