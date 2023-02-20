import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fmodule import FModule
from torch_geometric.nn import SAGEConv

class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, neighbor_feture):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feture.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feture.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feture.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}".format(self.aggr_method))
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden


class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu,
                 aggr_neighbor_method="mean", aggr_hidden_method="sum"):
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.aggr_neighbor = aggr_neighbor_method
        self.aggr_hidden = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hidden == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise  ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden


class Model(FModule):
    def __init__(self, hidden_dim=[64, 7], num_neighbors_list=[10, 10]):
        super(Model, self).__init__()
        self.input_dim = 1433
        self.num_neighbors_list = num_neighbors_list
        self.num_layers = len(num_neighbors_list)
        self.gcn = []
        self.gcn.append(SageGCN(1433, hidden_dim[0]))
        self.gcn.append(SageGCN(hidden_dim[0], hidden_dim[1], activation=None))

    def forward(self, node_features_list):
        hidden = node_features_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - 1):
                src_node_features = hidden[hop]
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].\
                    view(src_node_num, self.num_neighbors_list[hop], -1)
                h = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]