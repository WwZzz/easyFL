from torch import nn
from flgo.utils.fmodule import FModule


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(51, 2, bias=True)

    def forward(self, x):
        return self.fc(x)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)