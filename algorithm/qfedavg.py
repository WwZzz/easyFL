from .fedbase import BasicServer, BasicClient
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.q = option['q']
        self.learning_rate = option['learning_rate']
        self.L = 1.0/option['learning_rate']
        self.paras_name = ['q']

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(selected_clients)
        # plug in the weight updates into the gradient
        grads = [(self.model- model)/self.learning_rate for model in models]
        Deltas = [gi*np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,train_losses)]
        # estimation of the local Lipchitz constant
        hs = [self.q * np.float_power(li + 1e-10, (self.q - 1)) * (gi.norm() ** 2) + self.L * np.float_power(li + 1e-10, self.q) for gi,li in zip(grads,train_losses)]
        # aggregate
        self.model = self.aggregate(Deltas, hs)
        return selected_clients

    def aggregate(self, Deltas, hs):
        demominator = np.sum(np.asarray(hs))
        scaled_deltas = [delta/demominator for delta in Deltas]
        updates = sum(scaled_deltas)
        w_new = self.model - updates
        return w_new

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None, drop_rate=-1):
        super(Client, self).__init__(option, name, train_data, valid_data, drop_rate)