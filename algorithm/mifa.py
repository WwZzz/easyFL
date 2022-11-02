from .fedbase import BasicServer, BasicClient
from .fedavg import Client
from utils import fmodule
import utils.fflow as flw

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
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
        flw.logger.info("G_i Initialized For {}/{} The Clients.".format(s,self.num_clients))
        self.initflag = True
        return True

    def iterate(self):
        # sample all the active clients
        self.selected_clients = self.sample()
        # training
        models = self.communicate(self.selected_clients)['model']
        # update G
        for k in range(len(self.selected_clients)):
            self.update_table[self.selected_clients[k]] = 1.0 / self.lr * (self.model - models[k])
        # check if the update_table being initialized
        if not self.initflag:
            if not self.check_if_init():
                return
        # aggregate: w = w - eta_t * 1/N * sum(G_i)
        self.model = self.aggregate()
        return

    def aggregate(self):
        return self.model - self.lr * fmodule._model_average([update_i for update_i in self.update_table if update_i])


