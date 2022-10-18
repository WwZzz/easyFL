"""
This is a non-official implementation of the work 'Secure Federated Matrix Factorization' (https://arxiv.org/abs/1906.05108).
"""

import torch
from .fedbase import BasicServer
from .fedbase import BasicClient
import torch.nn as nn
from phe import paillier
from utils.fmodule import FModule
import numpy as np
import utils.fflow as flw

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024, )
num_items = 0

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        # the number of items
        global num_items
        num_items = test_data._NUM_ITEMS
        # the number of total ratings for training (i.e. M in the equation (2) of the original paper)
        self.num_train_samples = sum([len(c.train_data) for c in self.clients])
        for c in self.clients:c.num_train_samples = self.num_train_samples
        # hyper-parameters
        self.init_algo_para({'embedding_size': 100, 'lambda':1e-4})
        # initialize item vectors as [0.01 ... 0.01] as the official code did
        self.item_vectors = ItemVectors(num_items, self.embedding_size)
        self.item_vectors.set_embedding(np.zeros([num_items, self.embedding_size]) + 0.01)
        self.decrypted_items  = np.array(self.item_vectors.get_embedding().cpu().detach(), dtype=np.float64)
        # encrypt the item vectors
        flw.logger.time_start('Encrypt Item Vectors')
        self.encrypted_item_vectors = [[public_key.encrypt(e, precision=1e-5) for e in vector] for vector in self.decrypted_items]
        flw.logger.time_end('Encrypt Item Vectors')
        # initialize the user vectors
        for c in self.clients: c.user_embedding = CltModel(self.embedding_size)

    def pack(self, client_id):
        return {
            'encrypted_item_vectors': self.encrypted_item_vectors
        }

    def aggregate(self, encrypted_grads):
        for eng in encrypted_grads:
            for i in range(len(self.encrypted_item_vectors)):
                for j in range(len(self.encrypted_item_vectors[i])):
                    self.encrypted_item_vectors[i][j] = self.encrypted_item_vectors[i][j] - eng[i][j]

    def iterate(self, t):
        self.selected_clients = list(range(self.num_clients))
        en_grads = self.communicate(self.selected_clients)['encrypted_gradient']
        self.aggregate(en_grads)
        # update decrypted item embeddings
        self.decrypted_items = np.array([[private_key.decrypt(e) for e in vector] for vector in self.encrypted_item_vectors],dtype=np.float32)

    def test(self):
        user_embeddings = self.get_user_embeddings()
        item_embeddings = self.decrypted_items
        labels = torch.tensor([d[2] for d in self.test_data])
        predict = []
        for data in self.test_data:
            vec_u = user_embeddings[int(data[0])-1]
            vec_i = item_embeddings[data[1]-1]
            predict.append(vec_u.detach().cpu().numpy().dot(vec_i))
        predict = torch.tensor(np.array(predict))
        # mse
        loss_rmse = torch.sqrt(((labels-predict)**2).sum()/len(labels))
        loss_mae = torch.abs((labels-predict)).sum()/len(labels)
        return {'rmse': loss_rmse, 'mae': loss_mae}

    def get_user_embeddings(self):
        user_embeddings = []
        for c in self.clients:
            user_embeddings.append(c.user_embedding.get_embedding())
        return user_embeddings

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def reply(self, svr_pkg):
        item_vectors = self.unpack(svr_pkg)
        gradient = self.train(item_vectors)
        clt_pkg = self.pack(gradient)
        return clt_pkg

    def unpack(self, received_pkg):
        global num_items
        encrypted_item_vectors = received_pkg['encrypted_item_vectors']
        item_vector_np = np.array([[private_key.decrypt(e) for e in vector] for vector in encrypted_item_vectors], dtype=np.float32)
        item_vectors = ItemVectors(num_items, self.embedding_size)
        item_vectors.set_embedding(item_vector_np)
        return item_vectors

    def train(self, item_vectors):
        # zero grad
        item_vectors.embeddings.zero_grad()
        self.user_embedding.embeddings.zero_grad()
        # get a batch of data
        batch_data = self.get_batch_data()
        vec_items = item_vectors.get_embedding(batch_data[1])
        vec_user = self.user_embedding.to(vec_items.device).get_embedding()
        predict = (vec_items.matmul(vec_user.T)).squeeze()
        loss = ((batch_data[2] -predict)**2).sum()/self.num_train_samples + 1e-4*((vec_user**2).sum() + (vec_items**2).sum())
        loss.backward()
        gradient = np.array(item_vectors.embeddings.weight.grad.detach().cpu(), dtype=np.float64)
        with torch.no_grad():
            self.user_embedding.embeddings.weight -= self.learning_rate*self.user_embedding.embeddings.weight.grad
        return gradient

    def pack(self, gradient):
        encrypted_gradient = [[public_key.encrypt(e, precision=1e-5) for e in vector] for vector in gradient]
        return {'encrypted_gradient': encrypted_gradient}

    def test(self, model, dataflag='valid'):
        data = self.train_data if dataflag=='train' else self.valid_data
        self.data_loader = iter(self.calculator.get_data_loader(data, batch_size=len(data),num_workers=self.loader_num_workers))
        batch_data = next(self.data_loader)
        item_vectors = ItemVectors(num_items, self.embedding_size)
        item_vectors.set_embedding(self.server.decrypted_items)
        vec_items = item_vectors.get_embedding(batch_data[1])
        vec_user = self.user_embedding.to(vec_items.device).get_embedding()
        predict = vec_items.matmul(vec_user.T)
        len_pred = len(predict)
        predict =predict.squeeze()
        loss_rmse = torch.sqrt(((batch_data[2] -predict)**2).sum()/len_pred)
        loss_mae = (torch.abs(batch_data[2] -predict).sum())/len_pred
        return {'rmse': loss_rmse.data, 'mae': loss_mae.data}

class MyEmbedding(FModule):
    def __init__(self, dim=1682, embedding_size=100):
        super().__init__()
        self.embeddings = nn.Embedding(dim, embedding_size)

    def get_embedding(self, item_index=None):
        if item_index is None: return self.embeddings.weight
        return self.embeddings.weight[item_index-1]

    def set_embedding(self, array):
        assert tuple(self.embeddings.weight.shape)==array.shape
        with torch.no_grad():
            self.embeddings.weight.data = torch.tensor(array, dtype=torch.float32)

# globally shared item vectors
class ItemVectors(MyEmbedding):
    def __init__(self, n_items=1682, embedding_size=100):
        super().__init__(dim=n_items, embedding_size=embedding_size)

class CltModel(MyEmbedding):
    def __init__(self, embedding_size=100):
        super().__init__(dim=1, embedding_size=embedding_size)
