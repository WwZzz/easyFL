
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(99, 2)

    def forward(self, x):
        return self.fc(x)

