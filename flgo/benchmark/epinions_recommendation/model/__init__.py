from torch import nn
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self, n_users=5441, n_items=101998, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch_data):
        user_id = batch_data['user_id'].long()
        item_id = batch_data['item_id'].long()
        u, it = self.user_factors(user_id), self.item_factors(item_id)
        x = self.sigmoid((u * it).sum(1))
        return x * 5

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model(object.test_data.num_users, object.test_data.num_items, 20).to(object.device)