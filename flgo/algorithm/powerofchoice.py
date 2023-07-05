"""
This is a non-official implementation of 'Client Selection in Federated Learning:
Convergence Analysis and Power-of-Choice Selection Strategies' (https://arxiv.org/abs/2010.01243).
"""
import numpy as np
from flgo.algorithm.fedavg import Client
from flgo.algorithm.fedbase import BasicServer
import flgo.simulator.base as ss

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'d': self.num_clients})

    def sample(self):
        # create candidate set A
        num_candidate = min(self.d, len(self.available_clients))
        p_candidate = np.array([len(self.clients[cid].train_data) for cid in self.available_clients])
        candidate_set = np.random.choice(self.available_clients, num_candidate, p=p_candidate / p_candidate.sum(), replace=False)
        candidate_set = sorted(candidate_set)
        # communicate with the candidates for their local_movielens_recommendation loss
        losses = []
        for cid in candidate_set:
            losses.append(self.clients[cid].test(self.model)['loss'])
        # sort candidate set according to their local_movielens_recommendation loss value, and choose the top-M highest ones
        sort_id = np.array(losses).argsort().tolist()
        sort_id.reverse()
        num_selected = min(self.clients_per_round, len(self.available_clients))
        selected_clients = np.array(self.available_clients)[sort_id][:num_selected]
        return selected_clients.tolist()