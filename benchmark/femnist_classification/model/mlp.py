
from paddle import nn
import paddle.nn.functional as F
from utils.fmodule import FModule


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc3(x)
        return x

    def get_embedding(self, x):
        x = x.flatten(1, 3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x
