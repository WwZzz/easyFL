from torch import nn
from flgo.utils.fmodule import FModule

class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 576)

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.num_inp = 784
        self.net = nn.Sequential(*[nn.Conv2d(1, 32, 5), nn.ReLU(), nn.Conv2d(32, 32, 5), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(32, 64, 5),
                                 nn.MaxPool2d(2), nn.ReLU(), Reshape(), nn.Linear(576, 256), nn.ReLU(), nn.Linear(256, 62)])
        self.softmax = nn.Softmax(-1)

    def forward(self, data):
        data = data.reshape(-1, 1, 28, 28)
        out = self.net(data)
        return out

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)