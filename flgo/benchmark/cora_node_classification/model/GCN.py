import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNConv(16,7)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)