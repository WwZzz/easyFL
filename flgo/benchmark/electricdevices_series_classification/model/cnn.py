from torch import nn
import torch.nn.functional as F

import flgo.benchmark
from flgo.utils.fmodule import FModule

import os
from flgo.benchmark.toolkits.series.time_series_classification.datasets import UCRArchiveDataset

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = x.view(x.shape[0], 1, 96)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = F.relu(self.fc1(x))
        return x

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)

if __name__ == '__main__':
    model = Model()
    train_data = UCRArchiveDataset(root=os.path.join(flgo.benchmark.path, 'RAW_DATA', 'UCRArchive'), dataset_name='ElectricDevices')
    x, y = train_data[0]
    output = model(x)