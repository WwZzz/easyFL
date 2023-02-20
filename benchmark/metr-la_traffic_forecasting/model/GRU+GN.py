from utils.fmodule import FModule
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_add


class Model(FModule):
    def __init__(self, input_size=2, hidden_size=64, output_size=1, dropout=0,cl_decay_steps=1000, gru_num_layers=1):
        super().__init__()
        self.cl_decay_steps = cl_decay_steps
        self.gru_num_layers = gru_num_layers
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(
            input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
        )
        self.decoder = nn.GRU(
            input_size, 2 * hidden_size, num_layers=gru_num_layers, dropout=dropout
        )
        self.out_net = nn.Linear(2 * hidden_size, output_size)

    def _format_input_data(self, data):
        x, x_attr, y, y_attr, graph_encoding = data[0], data[1], data[2], data[3], data[4]
        batch_num, node_num = x.shape[0], x.shape[2]
        return x, x_attr, y, y_attr, graph_encoding, batch_num, node_num

    def _compute_sampling_threshold(self, batches_seen):
        if self.cl_decay_steps == 0:
            return 0
        else:
            return self.cl_decay_steps / (
                    self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, data, batches_seen, return_encoding=False, training=False):
        x, x_attr, y, y_attr, graph_encodings, batch_num, node_num = self._format_input_data(data)
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F
        _, h_encode = self.encoder(x_input)
        encoder_h = h_encode # (B x N) x L x F
        if graph_encodings is None:
            raise ValueError('graph_encoding should not be None')
        h_encode = torch.cat([h_encode, graph_encodings.permute(2, 1, 0, 3).flatten(1, 2)], dim=-1)
        last_input = x_input[-1:]
        last_hidden = h_encode
        step_num = y_attr.shape[1]
        out_steps = []
        y_input = y.permute(1, 0, 2, 3).flatten(1, 2)
        y_attr_input = y_attr.permute(1, 0, 2, 3).flatten(1, 2)
        for t in range(step_num):
            out_hidden, last_hidden = self.decoder(last_input, last_hidden)
            out = self.out_net(out_hidden) # T x (B x N) x F
            out_steps.append(out)
            last_input = torch.cat((out, y_attr_input[t:t + 1]), dim=-1)
            if training and torch.rand(1).item() < self._compute_sampling_threshold(batches_seen):
                last_input = torch.cat((y_input[t:t+1], y_attr_input[t:t+1]), dim=-1)
        out = torch.cat(out_steps, dim=0)
        out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        if return_encoding:
            return out, encoder_h
        else:
            return out

    def forward_encoder(self, data):
        x, x_attr, y, y_attr, graph_encodings, batch_num, node_num = self._format_input_data(data)
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2)  # T x (B x N) x F
        _, h_encode = self.encoder(x_input)
        return h_encode

    def forward_decoder(self, data, h_encode, batches_seen, return_encoding=False, training=False, server_graph_encoding=None):
        x, x_attr, y, y_attr,_, batch_num, node_num = self._format_input_data(data)
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2)
        encoder_h = h_encode
        graph_encoding = server_graph_encoding
        graph_encoding = graph_encoding.permute(2, 1, 0, 3).flatten(1, 2)  # L x (B x N) x F
        h_encode = torch.cat([h_encode, graph_encoding], dim=-1)
        last_input = x_input[-1:]
        last_hidden = h_encode
        step_num = y_attr.shape[1]
        out_steps = []
        y_input = y.permute(1, 0, 2, 3).flatten(1, 2)
        y_attr_input = y_attr.permute(1, 0, 2, 3).flatten(1, 2)
        for t in range(step_num):
            out_hidden, last_hidden = self.decoder(last_input, last_hidden)
            out = self.out_net(out_hidden)  # T x (B x N) x F
            out_steps.append(out)
            last_input = torch.cat((out, y_attr_input[t:t + 1]), dim=-1)
            if training and torch.rand(1).item() < self._compute_sampling_threshold(batches_seen):
                last_input = torch.cat((y_input[t:t + 1], y_attr_input[t:t + 1]), dim=-1)
        out = torch.cat(out_steps, dim=0)
        out = out.view(out.shape[0], batch_num, node_num, out.shape[-1]).permute(1, 0, 2, 3)
        if return_encoding:
            return out, encoder_h
        else:
            return out

class SvrModel(nn.Module):
    def __init__(self, node_input_size=64, edge_input_size=1, global_input_size=64,
                 hidden_size=256,
                 updated_node_size=128, updated_edge_size=128, updated_global_size=128,
                 node_output_size=64,
                 gn_layer_num=2, activation='ReLU', dropout=0, *args, **kwargs):
        super().__init__()

        self.global_input_size = global_input_size

        self.net = []
        last_node_input_size = node_input_size
        last_edge_input_size = edge_input_size
        last_global_input_size = global_input_size
        for _ in range(gn_layer_num):
            edge_model = EdgeModel(last_node_input_size, last_edge_input_size, last_global_input_size, hidden_size,
                                   updated_edge_size,
                                   activation, dropout)
            last_edge_input_size += updated_edge_size
            node_model = NodeModel(last_node_input_size, updated_edge_size, last_global_input_size, hidden_size,
                                   updated_node_size,
                                   activation, dropout)
            last_node_input_size += updated_node_size
            global_model = GlobalModel(updated_node_size, updated_edge_size, last_global_input_size, hidden_size,
                                       updated_global_size,
                                       activation, dropout)
            last_global_input_size += updated_global_size
            self.net.append(MetaLayer(
                edge_model, node_model, global_model
            ))
        self.net = nn.ModuleList(self.net)
        self.node_out_net = nn.Linear(last_node_input_size, node_output_size)

    def forward(self, data):
        if not hasattr(data, 'batch') or data.batch is None:
            data = Batch.from_data_list([data])
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        batch = batch.to(x.device)
        edge_attr = edge_attr.expand(-1, x.shape[1], x.shape[2], -1)
        u = x.new_zeros(*([batch[-1] + 1] + list(x.shape[1:-1]) + [self.global_input_size]))
        for net in self.net:
            updated_x, updated_edge_attr, updated_u = net(x, edge_index, edge_attr, u, batch)
            x = torch.cat([updated_x, x], dim=-1)
            edge_attr = torch.cat([updated_edge_attr, edge_attr], dim=-1)
            u = torch.cat([updated_u, u], dim=-1)
        node_out = self.node_out_net(x)
        return node_out

class MLP_GN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 hidden_layer_num, activation='ReLU', dropout=0.0):
        super().__init__()
        self.net = []
        last_layer_size = input_size
        for _ in range(hidden_layer_num):
            self.net.append(nn.Linear(last_layer_size, hidden_size))
            self.net.append(getattr(nn, activation)())
            self.net.append(nn.Dropout(p=dropout))
            last_layer_size = hidden_size
        self.net.append(nn.Linear(last_layer_size, output_size))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class EdgeModel(nn.Module):
    def __init__(self,
                 node_input_size, edge_input_size, global_input_size,
                 hidden_size, edge_output_size, activation, dropout):
        super(EdgeModel, self).__init__()
        edge_mlp_input_size = 2 * node_input_size + edge_input_size + global_input_size
        self.edge_mlp = MLP_GN(edge_mlp_input_size, hidden_size, edge_output_size, 2, activation, dropout)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], -1)
        if u is not None:
            out = torch.cat([out, u[batch]], -1)
        return self.edge_mlp(out)

class NodeModel(nn.Module):
    def __init__(self,
                 node_input_size, edge_input_size, global_input_size,
                 hidden_size, node_output_size, activation, dropout):
        super(NodeModel, self).__init__()
        node_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.node_mlp = MLP_GN(node_mlp_input_size, hidden_size, node_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        received_msg = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, received_msg], dim=-1)
        if u is not None:
            out = torch.cat([out, u[batch]], dim=-1)
        return self.node_mlp(out)

class GlobalModel(nn.Module):
    def __init__(self,
                 node_input_size, edge_input_size, global_input_size,
                 hidden_size, global_output_size, activation, dropout):
        super(GlobalModel, self).__init__()
        global_mlp_input_size = node_input_size + edge_input_size + global_input_size
        self.global_mlp = MLP_GN(global_mlp_input_size, hidden_size, global_output_size, 2, activation, dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        agg_node = scatter_add(x, batch, dim=0)
        agg_edge = scatter_add(scatter_add(edge_attr, col, dim=0, dim_size=x.size(0)), batch, dim=0)
        out = torch.cat([agg_node, agg_edge, u], dim=-1)
        return self.global_mlp(out)
