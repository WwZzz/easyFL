import torch.nn
import flgo.benchmark
import os
import torch_geometric
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3703, 128)
        self.conv2 = GCNConv(128, 64)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index.long())
        x = x.relu()
        return self.conv2(x, edge_index.long())

    def decode(self, z, edge_label_index):
        return (z[edge_label_index.long()[0]] * z[edge_label_index.long()[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'CITESEER')
dataset = torch_geometric.datasets.Planetoid(path, name='Citeseer')
train_data = dataset[0]

def get_model():
    return GCN()