import json
import os
import flgo
import torch
import torch.nn as nn
import torch.nn.functional as F

from flgo.utils.fmodule import FModule


class Model(FModule):

    def __init__(self, hidden_size=50, num_features=8):
        super(Model, self).__init__()
        self.encoder = nn.GRU(input_size=num_features, hidden_size=hidden_size, num_layers=4)
        self.fc = nn.Linear(hidden_size * 4, num_features)

    def forward(self, x):  # [B, W, M]
        x = x.permute([1, 0, 2])  # [W, B, M]
        _, h = self.encoder(x)  # [4, B, h]
        h = h.permute([1, 0, 2]).contiguous()  # [B, 4, h]
        h = h.view(h.shape[0], -1)  # [B, 4*h]
        x = self.fc(h)  # [B, M]
        return F.sigmoid(x)


def init_local_module(object):
    pass


def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)
