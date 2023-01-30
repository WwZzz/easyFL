"""
This is a non-official implementation of the work 'TiFL: A Tier-based Federated Learning System'
(http://arxiv.org/abs/2001.09249). This implementation refers to the github repository https://github.com/bacox/fltk.
Thanks for their great work.

T: the number of tiers
I: the interval of updating testing accuracy for each tier
C: the degree of the diverse in the `Credits` for tiers. The smaller C is, the uniform the Credits distributes.
"""
from .fedbase import BasicServer
from .fedbase import BasicClient as Client
import utils.system_simulator as ss
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.init_algo_para({'T':5, 'I':10,'C':0})
        self.tolerance_for_latency = 0
        self.tiers = None

    def profiling_and_tiering(self):
        # collecting the latency for each client
        client_latencies = [c._latency for c in self.clients]
        ordered_cids = np.argsort(client_latencies)
        groups = np.split(ordered_cids, self.T)
        group_latency = [(np.array(client_latencies)[group]).mean() for group in groups]
        tiers = {tid: {
            'clients': groups[tid],
            'ave_latency': group_latency[tid],
            'accuracy':0.0,
            'prob':1.0/self.T,
            'credit': self.num_rounds,
        } for tid in range(self.T)}
        p = (-4 * np.log(self.C + 10e-8)) ** 4
        credits = np.random.dirichlet([p for _ in tiers])
        credits = np.around(credits, 3)
        credits = credits/credits.sum()
        credits.sort()
        rounds = np.ones(self.num_rounds)
        credits = np.split(rounds, (np.cumsum(credits)*self.num_rounds).astype(int)[:-1])
        credits = [ci.sum() for ci in credits]
        tids = [tids for tids in tiers]
        tids = sorted(tids, key=lambda x:  tiers[x]['ave_latency'], reverse=True)
        for tid, c in zip(tids, credits): tiers[tid]['credit'] = int(c)
        return tiers

    def iterate(self):
        if self.tiers is None: self.tiers =  self.profiling_and_tiering()
        if (self.current_round-1)%self.I==0 and self.current_round-1>=self.I:
            self.change_probs()
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        self.model = self.aggregate(models)
        # update accuracy of each tier
        valid_accs = np.array(self.test_on_clients('valid')['accuracy'])
        for t in self.tiers:
            self.tiers[t]['accuracy'] = (valid_accs[self.tiers[t]['clients']]).mean()
        return

    def sample(self):
        probs = [self.tiers[t]['prob'] for t in self.tiers.keys()]
        while True:
            self.current_tier = int(np.random.choice(list(self.tiers.keys()), 1, p=probs))
            if self.tiers[self.current_tier]['credit'] > 0:
                self.tiers[self.current_tier]['credit'] = self.tiers[self.current_tier]['credit']-1
                break
        selected_clients = np.random.choice(self.tiers[self.current_tier]['clients'], min(self.clients_per_round, len(self.tiers[self.current_tier]['clients'])))
        return selected_clients.tolist()

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