from utils import fmodule
from .fedbase import BaseServer, BaseClient
import numpy as np
import torch

class Server(BaseServer):
    def __init__(self, option, model, clients, dtest = None):
        super(Server, self).__init__(option, model, clients, dtest)
        self.q = option['q']
        self.learning_rate = option['learning_rate']
        self.L = 1.0/option['learning_rate']
        self.paras_name = ['q']

    def iterate(self, t):
        ws, losses, Deltas, hs = [], [], [], []
        # sample clients
        selected_clients = self.sample()
        # training
        ws, losses = self.communicate(selected_clients)
        # plug in the weight updates into the gradient
        grads = [fmodule.modeldict_scale(fmodule.modeldict_sub(self.model.state_dict(), w), 1.0 / self.learning_rate) for w in ws]
        Deltas = [fmodule.modeldict_scale(gi, np.float_power(li + 1e-10, self.q)) for gi,li in zip(grads,losses)]
        # estimation of the local Lipchitz constant
        hs = [self.q * np.float_power(li + 1e-10, (self.q - 1)) * (fmodule.modeldict_norm(gi) ** 2) + self.L * np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,losses)]
        # aggregate
        w_new = self.aggregate(Deltas, hs)
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        return loss_avg

    def aggregate(self, Deltas, hs):
        demominator = np.sum(np.asarray(hs))
        scaled_deltas = []
        for delta in Deltas:
            scaled_deltas.append(fmodule.modeldict_scale(delta, 1.0 / demominator))
        updates = {}
        for layer in scaled_deltas[0].keys():
            updates[layer] = torch.zeros_like(scaled_deltas[0][layer])
            for sdelta in scaled_deltas:
                updates[layer] += sdelta[layer]
        w_new = fmodule.modeldict_sub(self.model.state_dict(), updates)
        return w_new

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, partition = 0.8, drop_rate = 0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, partition, drop_rate)