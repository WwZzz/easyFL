from utils.fmodule import FModule
import numpy as np
import torch
import torch.nn as nn

class Model(FModule):
    def __init__(self, input_size=2, hidden_size=100, output_size=1, dropout=0, gru_num_layers=1, cl_decay_steps=1000):
        super().__init__()
        self.cl_decay_steps = cl_decay_steps
        self.gru_num_layers = gru_num_layers
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout)
        self.decoder = nn.GRU(input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout)
        self.out_net = nn.Linear(hidden_size, output_size)

    def _compute_sampling_threshold(self, batches_seen):
        if self.cl_decay_steps == 0:
            return 0
        else:
            return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def _format_input_data(self, data):
        x, x_attr, y, y_attr = data[0], data[1], data[2], data[3]
        batch_num, node_num = x.shape[0], x.shape[2]
        return x, x_attr, y, y_attr, batch_num, node_num

    def forward(self, data, batches_seen, training=False):
        # B x T x N x F
        x, x_attr, y, y_attr, batch_num, node_num = self._format_input_data(data)
        x_input = torch.cat((x, x_attr), dim=-1).permute(1, 0, 2, 3).flatten(1, 2) # T x (B x N) x F
        _, h_encode = self.encoder(x_input)
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
        return out