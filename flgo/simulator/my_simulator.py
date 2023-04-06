from flgo.simulator.base import BasicSimulator
import numpy as np
class Simulator(BasicSimulator):
    def update_client_connectivity(self, client_ids):
        probs = [0.1 for _ in client_ids]
        self.set_variable(client_ids, 'prob_drop', probs)

    def update_client_availability(self):
        self.roundwise_fixed_availability = True
        pa = [0.9 for _ in self.clients]
        pua = [0.1 for _ in self.clients]
        self.set_variable(self.all_clients, 'prob_available', pa)
        self.set_variable(self.all_clients, 'prob_unavailable', pua)

    def update_client_responsiveness(self, client_ids, *args, **kwargs):
        latency = [np.random.randint(5,100) for _ in client_ids]
        self.set_variable(client_ids, 'latency', latency)

    def update_client_completeness(self, client_ids, *args, **kwargs):
        working_amount = [max(int(self.clients[cid].num_steps*np.random.rand()), 1) for cid in client_ids]
        self.set_variable(client_ids, 'working_amount', working_amount)