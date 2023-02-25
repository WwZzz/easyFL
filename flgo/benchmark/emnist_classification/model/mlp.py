from torch import nn
import torch.nn.functional as F
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 62)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.fc1(x)
        x = F.relu(x)
        return x

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)