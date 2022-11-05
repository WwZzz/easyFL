from .fedbase import BasicServer
from .fedbase import BasicClient
import utils.system_simulator as ss
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.init_algo_para({'T':5, 'I':10})
        self.tolerance_for_latency = 0
        self.tiers = self.profiling_and_tiering()

    def profiling_and_tiering(self):
        # collecting the latency for each client
        client_latencies = [c.response_latency for c in self.clients]
        ordered_cids = np.argsort(client_latencies)
        groups = np.split(ordered_cids, self.T)
        group_latency = [(np.array(client_latencies)[group]).mean() for group in groups]
        tiers = {tid: {
            'clients': groups,
            'ave_latency': group_latency[tid],
            'accuracy':0.0,
            'prob':1.0/self.T,
            'credit': self.num_rounds,
        } for tid in range(len(self.T))}
        return tiers

    @ss.time_step
    @ss.update_systemic_state
    def iterate(self):
        if (self.current_round-1)%self.I==0 and self.current_round-1>=self.I:
            self.change_probs()
        self.selected_clients = self.sample()
        models = self.communicate()['model']
        self.model = self.aggregate(models)
        # update accuracy of each tier
        valid_accs = np.array(self.test_on_clients('valid')['accuracy'])
        for t in self.tiers:
            self.tiers[t]['accuracy'] = (valid_accs[self.tiers[t]['clients']]).mean()
        return

    def sample(self):
        probs = [self.tiers[t]['prob'] for t in self.tiers.keys()]
        while True:
            self.current_tier = np.random.choice(list(self.tiers.keys()), 1, p=probs)
            if self.tiers[self.current_tier]['credit'] > 0:
                self.tiers[self.current_tier]['credit'] -= 1
                break
        selected_clients = np.random.choice(self.tiers[self.current_tier]['clients'], min(self.clients_per_round, len(self.tiers[self.current_tier]['clients'])), replacement=False)
        return selected_clients

    def change_probs(self):
        n = len([t for t in self.tiers if self.tiers[t]['credit']>0])
        D = n * (n + 1) / 2
        tier_accs = [self.tiers[t]['accuracy'] for t in self.tiers]
        sorted_tid = np.argsort(tier_accs)
        tiers = np.array([t for t in self.tiers])[sorted_tid]
        idx_decr = 0
        for idx, t in enumerate(tiers):
            if self.tiers[t]['credit'] > 0:
                self.tiers[t]['prob'] = (n - (idx - idx_decr)) / D
            else:
                self.tiers[t]['prob'] = 0
                idx_decr += 1
        return