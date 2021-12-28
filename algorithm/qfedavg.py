from .fedbase import BasicServer, BasicClient
import numpy as np
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.q = option['q']
        self.paras_name = ['q']

    def iterate(self, t):
        # sample clients
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients)
        if self.selected_clients == []: return
        # plug in the weight updates into the gradient
        grads = [(self.model- model) / self.lr for model in models]
        Deltas = [gi*np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,train_losses)]
        # estimation of the local Lipchitz constant
        hs = [self.q * np.float_power(li + 1e-10, (self.q - 1)) * (gi.norm() ** 2) + 1.0 / self.lr * np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,train_losses)]
        # aggregate
        self.model = self.aggregate(Deltas, hs)
        return

    def aggregate(self, Deltas, hs):
        demominator = np.sum(np.asarray(hs))
        scaled_deltas = [delta/demominator for delta in Deltas]
        updates = fmodule._model_sum(scaled_deltas)
        new_model = self.model - updates
        return new_model

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)