"""
This is a non-official implementation of 'Fast Federated Learning in the
Presence of Arbitrary Device Unavailability' (http://arxiv.org/abs/2106.04159)
"""
from flgo.algorithm.fedbase import BasicServer, BasicClient
from flgo.algorithm.fedavg import Client
from flgo.utils import fmodule

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'c':1.0})
        self.update_table = [None for _ in range(self.num_clients)]
        self.initflag = False
        # choose all the clients that are active
        self.clients_per_round = self.num_clients

    def check_if_init(self):
        """Check whether the update_table is initialized"""
        s = len([u for u in self.update_table if u])
        # c==0 infers that updating starts immediately
        if s < self.c*self.num_clients: return False
        self.gv.logger.info("G_i Initialized For {}/{} The Clients.".format(s,self.num_clients))
        self.initflag = True
        return True

    def iterate(self):
        # sample all the active clients
        self.selected_clients = self.sample()
        # training
        models = self.communicate(self.selected_clients)['model']
        # update G
        for k in range(len(self.received_clients)):
            self.update_table[self.received_clients[k]] = 1.0 / self.lr * (self.model - models[k])
        # check if the update_table being initialized
        if not self.initflag:
            if not self.check_if_init():
                return
        # aggregate: w = w - eta_t * 1/N * sum(G_i)
        self.model = self.aggregate()
        return

    def aggregate(self):
        return self.model - self.lr * fmodule._model_average([update_i for update_i in self.update_table if update_i])