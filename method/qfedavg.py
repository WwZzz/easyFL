from task import modelfuncs
from .fedbase import BaseServer, BaseClient
import numpy as np
import torch

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)
        self.q = option['q']
        self.learning_rate = option['learning_rate']
        self.L = 1.0/option['learning_rate']
        self.paras_name = ['q']

    def communicate(self, cid):
        # setting client(cid)'s model with latest parameters
        self.trans_model.load_state_dict(self.model.state_dict())
        self.clients[cid].setModel(self.trans_model)
        # wait for replying of the update and loss
        w = self.clients[cid].reply()[0]
        # calculate the loss  Fk(w_(t-1)) of the global model on client[k]'s training dataset
        loss = self.clients[cid].test('train')[1]
        return w, loss

    def iterate(self, t):
        ws, losses, Deltas, hs = [], [], [], []
        # sample clients
        selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss = self.communicate(cid)
            ws.append(w)
            losses.append(loss)
            # plug in the weight updates into the gradient
            grad = modelfuncs.modeldict_scale(modelfuncs.modeldict_sub(self.model.state_dict(), w), 1.0 / self.learning_rate)
            delta = modelfuncs.modeldict_scale(grad, np.float_power(loss + 1e-10, self.q))
            Deltas.append(delta)
            # estimation of the local Lipchitz constant
            hs.append(self.q * np.float_power(loss + 1e-10, (self.q - 1)) * (
                        modelfuncs.modeldict_norm(grad) ** 2) + self.L * np.float_power(loss + 1e-10, self.q))
        # aggregate
        w_new = self.aggregate(Deltas, hs)
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        return loss_avg

    def sample(self):
        cids = [i for i in range(self.num_clients)]
        return list(np.random.choice(cids, self.clients_per_round, replace=False, p=[nk/self.data_vol for nk in self.client_vols]))

    def aggregate(self, Deltas, hs):
        demominator = np.sum(np.asarray(hs))
        scaled_deltas = []
        for delta in Deltas:
            scaled_deltas.append(modelfuncs.modeldict_scale(delta, 1.0 / demominator))
        updates = {}
        for layer in scaled_deltas[0].keys():
            updates[layer] = torch.zeros_like(scaled_deltas[0][layer])
            for sdelta in scaled_deltas:
                updates[layer] += sdelta[layer]
        w_new = modelfuncs.modeldict_sub(self.model.state_dict(), updates)
        return w_new

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option, name, data_train_dict, data_test_dict, partition)