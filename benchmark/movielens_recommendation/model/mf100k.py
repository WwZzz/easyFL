from torch import nn
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self, n_users=943, n_items=1682, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)

    def forward(self, user, item):
        user = user - 1
        item = item - 1
        u, it = self.user_factors(user), self.item_factors(item)
        x = (u * it).sum(1)
        return x * 5