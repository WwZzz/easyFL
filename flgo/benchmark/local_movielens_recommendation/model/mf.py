import torch.nn as nn
from flgo.utils.fmodule import FModule
import torch
import numpy as np

class RecEmbedding(FModule):
    def __init__(self, dim=1682, embedding_size=20):
        super().__init__()
        self.embeddings = nn.Embedding(dim, embedding_size)

    def get_embedding(self, item_index=None):
        if item_index is None: return self.embeddings.weight
        return self.embeddings.weight[item_index]

    def set_embedding(self, array):
        assert tuple(self.embeddings.weight.shape)==array.shape
        with torch.no_grad():
            self.embeddings.weight.data = torch.tensor(array, dtype=torch.float32)

# globally shared item vectors
class ItemEmbedding(RecEmbedding):
    def __init__(self, n_items=1682, embedding_size=20):
        super().__init__(dim=n_items, embedding_size=embedding_size)

    def forward(self, batch_data, local_model):
        user_id = batch_data['user_id'].long()
        item_id = batch_data['item_id'].long()
        user_id_map = (user_id.unique().sort()).indices
        user_id = (user_id.view(-1,1)==user_id_map).int().argmax(dim=1)
        user_embedding = local_model.get_embedding(user_id)
        item_embedding = self.get_embedding(item_id)
        rating = (user_embedding*item_embedding).sum(1)
        return rating

class UserEmbedding(RecEmbedding):
    def __init__(self, embedding_size=20):
        super().__init__(dim=1, embedding_size=embedding_size)

def init_local_module(object):
    if 'Client' in object.__class__.__name__:
        object.set_model(UserEmbedding(20))

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.set_model(ItemEmbedding(object.test_data.num_items, 20))