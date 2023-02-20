from torch import nn
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self, dim_in = 784, dim_out = 3):
        super(Model, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.layer.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer(x)
        return x
