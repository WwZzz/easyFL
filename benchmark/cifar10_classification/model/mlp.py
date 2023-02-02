from paddle import nn
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self, dim_in=3*32*32, dim_hidden=200, dim_out=10):
        super(Model, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(dim_hidden, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = x.flatten(1, 3)
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        return x