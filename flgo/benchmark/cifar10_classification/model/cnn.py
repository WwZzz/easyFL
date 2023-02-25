from torch import nn
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.embedder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(1600, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
        )
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = self.get_embedding(x)
        return self.fc(x)

    def get_embedding(self, x):
        return self.embedder(x)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)