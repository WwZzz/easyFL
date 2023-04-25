"""
This is a non-official implementation of 'Tackling the Objective Inconsistency Problem
in Heterogeneous Federated Optimization' (http://arxiv.org/abs/2007.07481)
"""
from flgo.algorithm.fedbase import BasicServer, BasicClient
from flgo.utils import fmodule

class Server(BasicServer):
    def iterate(self):
        self.selected_clients = self.sample()
        # training
        res = self.communicate(self.selected_clients)
        models, taus = res['model'], res['tau']
        ds = [(model-self.model)/tauk for model, tauk in zip(models, taus)]
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        self.model = self.aggregate(ds, taus, p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients])
        return

    def aggregate(self, ds, taus, p=[]):
        if not ds: return self.model
        if self.aggregation_option == 'weighted_scale':
            K = len(ds)
            N = self.num_clients
            tau_eff = sum([tauk*pk for tauk,pk in zip(taus, p)])
            delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)]) * N / K
        elif self.aggregation_option == 'uniform':
            tau_eff = 1.0*sum(taus)/len(ds)
            delta = fmodule._model_average(ds)

        elif self.aggregation_option == 'weighted_com':
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)])
        else:
            sump = sum(p)
            p = [pk/sump for pk in p]
            tau_eff = sum([tauk * pk for tauk, pk in zip(taus, p)])
            delta = fmodule._model_sum([dk * pk for dk, pk in zip(ds, p)])
        return self.model + tau_eff * delta

class Client(BasicClient):
    def pack(self, model):
        tau = self._working_amount if hasattr(self, '_working_amount') else self.num_steps
        return {
            "model" : model,
            "tau": tau,
        }