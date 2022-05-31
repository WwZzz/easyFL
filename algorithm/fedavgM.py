from .fedavg import Client
from .fedbase import BasicServer

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.beta = option['beta']
        self.paras_name=['beta']
        self.v = model.zeros_like()

    def iterate(self, t):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        new_model = self.aggregate(models,p=[1.0 * self.client_vols[cid] / self.data_vol for cid in self.selected_clients])
        self.v = self.beta*self.v + (self.model - new_model)
        self.model = self.model - self.v
        return
