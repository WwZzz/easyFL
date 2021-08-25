from torch import nn
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self, dim_in = 60, dim_out = 10):
        super(Model, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.layer.bias.data.zero_()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer(x)
        return x

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)