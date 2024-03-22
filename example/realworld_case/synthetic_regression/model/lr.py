from torch import nn
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self, dim_in = 60, dim_out = 10):
        super(Model, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.layer.bias.data.zero_()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer(x)
        return x

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)