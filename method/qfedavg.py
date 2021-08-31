from .fedbase import BaseServer, BaseClient
import numpy as np


class Server(BaseServer):
    def __init__(self, option, model, clients, dtest = None):
        super(Server, self).__init__(option, model, clients, dtest)
        self.q = option['q']
        self.learning_rate = option['learning_rate']
        self.L = 1.0/option['learning_rate']
        self.paras_name = ['q']

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # training
        ws, losses = self.communicate(selected_clients)
        # plug in the weight updates into the gradient
        grads = [(self.model- w)/self.learning_rate for w in ws]
        Deltas = [gi*np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,losses)]
        # estimation of the local Lipchitz constant
        hs = [self.q * np.float_power(li + 1e-10, (self.q - 1)) * (gi.norm() ** 2) + self.L * np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,losses)]
        # aggregate
        self.model = self.aggregate(Deltas, hs)
        return selected_clients

    def aggregate(self, Deltas, hs):
        demominator = np.sum(np.asarray(hs))
        scaled_deltas = [delta/demominator for delta in Deltas]
        updates = sum(scaled_deltas)
        w_new = self.model - updates
        return w_new

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, train_rate = 0.8, drop_rate = 0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, train_rate, drop_rate)