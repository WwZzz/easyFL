from torch import nn
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self, dim_in=3*32*32, dim_hidden=200, dim_out=10):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)