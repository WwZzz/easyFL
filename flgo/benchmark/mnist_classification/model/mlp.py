
from torch import nn
import torch.nn.functional as F
from flgo.utils.fmodule import FModule


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU()
        )
        self.head = nn.Linear(200, 10)


    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x



def init_local_module(object):
    pass

def init_global_module(object):
    object.model = Model().to(object.device) if 'Server' in object.__class__.__name__ else None