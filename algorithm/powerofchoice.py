"""
This is a non-official implementation of 'Client Selection in Federated Learning:
Convergence Analysis and Power-of-Choice Selection Strategies' (https://arxiv.org/abs/2010.01243).
"""
import numpy as np
from .fedavg import Client
from .fedbase import BasicServer
import utils.systemic_simulator as ss

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.init_algo_para({'d': self.num_clients})

    @ss.with_inactivity
    def sample(self):
        # create candidate set A
        num_candidate = min(self.d, len(self.active_clients))
        p_candidate = np.array([self.local_data_vols[cid] for cid in self.active_clients])
        candidate_set = np.random.choice(self.active_clients, num_candidate, p=p_candidate/p_candidate.sum(), replace=False)
        candidate_set = sorted(candidate_set)
        # communicate with the candidates for their local loss
        losses = []
        for cid in candidate_set:
            losses.append(self.clients[cid].test(self.model)['loss'])
        # sort candidate set according to their local loss value, and choose the top-M highest ones
        sort_id = np.array(losses).argsort().tolist()
        sort_id.reverse()
        num_selected = min(self.clients_per_round, len(self.active_clients))
        selected_clients = np.array(self.active_clients)[sort_id][:num_selected]
        return selected_clients.tolist()