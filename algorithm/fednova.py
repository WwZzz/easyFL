from .fedbase import BasicServer, BasicClient
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def iterate(self, t):
        self.selected_clients = self.sample()
        # training
        res = self.communicate(self.selected_clients)
        models, taus = res['model'], res['tau']
        ds = [(model-self.model)/tauk for model, tauk in zip(models, taus)]
        self.model = self.aggregate(ds, taus, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return

    def aggregate(self, ds, taus, p=[]):
        if not ds: return self.model
        if self.agg_option == 'weighted_scale':
            K = len(ds)
            N = self.num_clients
            tau_eff = sum([tauk*pk for tauk,pk in zip(taus, p)])
            delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)]) * N / K
        elif self.agg_option == 'uniform':
            tau_eff = 1.0*sum(taus)/len(ds)
            delta = fmodule._model_average(ds)

        elif self.agg_option == 'weighted_com':
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)])
        else:
            sump = sum(p)
            p = [pk/sump for pk in p]
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)])
        return self.model + tau_eff * delta

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def pack(self, model):
        return {
            "model" : model,
            "tau": self.num_steps,
        }
