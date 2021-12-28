from .fedbase import BasicServer, BasicClient
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.update_table = [None for _ in range(self.num_clients)]
        self.initflag = False
        self.sample_option = 'active'
        self.waiting = True

    def check_if_init(self):
        """Check whether the update_table is initialized"""
        for i in range(self.num_clients):
            if self.update_table[i]==None:
                return False
        print("G_i Initialized For All The Clients.")
        return True

    def iterate(self, t):
        # sample all the active clients
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients)
        # update G
        for client_id in range(len(self.selected_clients)):
            self.update_table[self.selected_clients[client_id]] = 1.0 / self.lr * (self.model - models[client_id])
        # wait for initialization of update_table before aggregation
        if not self.initflag:
            self.initflag = self.check_if_init()
            return
        # aggregate: w = w - eta_t * 1/N * sum(G_i)
        self.model = self.aggregate()
        return

    def aggregate(self):
        return self.model-self.lr * fmodule._model_average(self.update_table)


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)





