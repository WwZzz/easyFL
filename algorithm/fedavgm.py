"""
This is a non-official implementation of 'Measuring the Effects
of Non-Identical Data Distribution for Federated Visual Classification'
(http://arxiv.org/abs/1909.06335).
"""

from .fedavg import Client
from .fedbase import BasicServer

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.init_algo_para({'beta': 0.01})
        self.v = model.zeros_like()

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        new_model = self.aggregate(models)
        self.v = self.beta*self.v + (self.model - new_model)
        self.model = self.model - self.v
        return
