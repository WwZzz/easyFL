import torch.nn
import flgo.benchmark
import os
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
import torch

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(500, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, tdata):
        z = self.encode(tdata.x, tdata.edge_index)
        if self.training:
            neg_edge_index = negative_sampling(
                edge_index=torch.cat([tdata.edge_index, tdata.edge_label_index],dim=-1,), num_nodes=tdata.num_nodes,
                num_neg_samples=tdata.edge_label_index.size(1),
                force_undirected=tdata.is_undirected()
            )
            outputs = self.decode(z, torch.cat([tdata.edge_label_index, neg_edge_index],dim=-1,)).view(-1)
            return outputs, neg_edge_index
        else:
            outputs = self.decode(z, tdata.edge_label_index).view(-1)
            return outputs

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index.long())
        x = x.relu()
        return self.conv2(x, edge_index.long())

    def decode(self, z, edge_label_index):
        return (z[edge_label_index.long()[0]] * z[edge_label_index.long()[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

trans = T.NormalizeFeatures()
path = os.path.join(flgo.benchmark.path,'RAW_DATA', 'PUBMED')
dataset = torch_geometric.datasets.Planetoid(path, name='Pubmed', transform=trans)
train_data = dataset[0]

def get_model():
    return GCN()