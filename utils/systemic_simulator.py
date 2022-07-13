"""This module is designed for simulating the network environment of the real world.
Decorators are written here to decorate the originnal methods owned by the central server.
Here we implement three different network heterogeneity for a FL system:
    1. with_accessibility: some clients are not available during the stage of sampling
    2. with_latency: accumulating latencies of clients and dropout the overdue clients
"""
import numpy as np
TIME_UNIT = 1

def update_activity(server):
    return

def update_local_computing_resource(server, clients = []):
    return

def init_network_mode(server, mode='ideal'):
    global update_activity
    if mode=='ideal':
        for c in server.clients:
            c.network_active_rate = 1
            c.network_drop_rate = 0
            c.network_latency_amount = 0

    elif mode.startswith('MIFA'):
        """
        This setting follows the activity mode in 'Fast Federated Learning in the 
        Presence of Arbitrary Device Unavailability' , where each client ci will be ready
        for join in a communcation round with the probability:
            pi = pmin * min({label kept by ci}) / max({all labels}) + ( 1 - pmin )
        and the participation of client is independent for different rounds.
        """
        import collections
        pmin = float(mode[mode.find('p')+1:]) if mode.find('p')!=-1 else 0.1
        def label_counter(dataset):
            return collections.Counter([int(dataset[di][-1]) for di in range(len(dataset))])
        label_num = len(label_counter(server.test_data))
        for c in server.clients:
            c.network_drop_rate = 0
            c.network_latency_amount = 0
            c_counter = label_counter(c.train_data+c.valid_data)
            c_label = [lb for lb in c_counter.keys()]
            c.network_active_rate = (pmin * min(c_label) / max(1, label_num-1)) + (1 - pmin)

    elif mode=='F3AST-Scarce':
        """The following four settings are from 'Federated Learning Under Intermittent 
        Client Availability and Time-Varying Communication Constraints' (http://arxiv.org/abs/2205.06730), 
        which proposes F3AST algorithm to handle the intermittent client availability. More details about 
        these availability models can be found  in their paper."""
        for c in server.clients:
            c.network_active_rate = 0.8
            c.network_drop_rate = 0
            c.network_latency_amount = 0

    elif mode=='F3AST-Home':
        Tks = [np.random.lognormal(0,0.5) for _ in server.clients]
        max_Tk = max(Tks)
        for c,Tk in zip(server.clients, Tks):
            c.network_active_rate = 1.0 * Tk / max_Tk
            c.network_drop_rate = 0
            c.network_latency_amount = 0

    elif mode=='F3AST-Smartphones':
        def f(server):
            times = np.linspace(start=0, stop=2*np.pi, num=24)
            fts = 0.4 * np.sin(times) + 0.5
            t = server.current_round % 24
            for c in server.clients:
                c.network_active_rate = fts[t] * c.qk
        update_activity = f
        Tks = [np.random.lognormal(0,0.25) for _ in server.clients]
        max_Tk = max(Tks)
        for c,Tk in zip(server.clients, Tks):
            c.qk = 1.0 * Tk / max_Tk
            c.network_active_rate = 1
            c.network_drop_rate = 0
            c.network_latency_amount = 0

    elif mode=='F3AST-Uneven':
        max_data_vol = max(server.local_data_vols)
        for c, cvol in zip(server.clients, server.local_data_vols):
            c.network_active_rate = 1.0 * cvol / max_data_vol
            c.network_drop_rate = 0
            c.network_latency_amount = 0
    else:
        for c in server.clients:
            c.network_active_rate = 1
            c.network_drop_rate = 0
            c.network_latency_amount = 0
        return

def init_computing_mode(server, mode='ideal'):
    global update_local_computing_resource
    if mode == 'ideal':
        return

    elif mode.startswith('FEDPROX'):
        """
        This setting follows the setting in the paper 'Federated Optimization in Heterogeneous Networks' 
        (http://arxiv.org/abs/1812.06127). The string `mode` should be like `FEDPROX-pk` where `k` should be 
        replaced by a float number. The `k` specifies the number of selected clients who perform incomplete updates
        in each communication round. THe default value of `k` is 0.5 as the middle case of heterogeneity in the original
        paper.
        """
        p = float(mode[mode.find('p')+1:]) if mode.find('p')!=-1 else 0.5
        def f(server, clients=[]):
            incomplete_clients = np.random.choice(clients, round(len(clients)*p), replace=False)
            for cid in incomplete_clients:
                server.clients[cid].num_steps = np.random.randint(low=1, high=server.clients[cid].num_steps)
            return
        update_local_computing_resource = f
        return

    elif mode.startswith('FEDNOVA-Uniform'):
        """
        This setting follows the setting in the paper 'Tackling the Objective Inconsistency Problem in 
        Heterogeneous Federated Optimization' (http://arxiv.org/abs/2007.07481). The string `mode` should be like 
        'FEDNOVA-Uniform(a,b)' where `a` is the minimal value of the number of local epochs and `b` is the maximal 
        value. If this mode is active, the `num_epochs` and `num_steps` of clients will be disable.
        """
        a = int(mode[mode.find('(')+1: mode.find(',')])
        b = int(mode[mode.find(',')+1: mode.find(')')])
        def f(server, clients=[]):
            for cid in server.clients:
                server.clients[cid].set_local_epochs(np.random.randint(low=a, high=b))
            return
        update_local_computing_resource = f
        return

    elif mode.startswith('FEDNOVA-Fixed-Uniform'):
        """Under this setting, the heterogeneity of computing resource is static across the whole training process."""
        a = int(mode[mode.find('(')+1: mode.find(',')])
        b = int(mode[mode.find(',')+1: mode.find(')')])
        for cid in server.clients:
            server.clients[cid].set_local_epochs(np.random.randint(low=a, high=b))
        return

    elif mode=='Fixed-Uniform':
        for c in server.clients:
            c.num_steps = max(1, int(c.num_steps*np.random.rand()))
        return

def init_systemic_config(server, option):
    # init network config
    init_network_mode(server, option['network_config'])
    # init computing power distribution
    init_computing_mode(server, option['computing_config'])

# sampling phase
def with_inactivity(sample):
    def sample_with_active(self, all_clients=None):
        update_activity(self)
        active_clients = []
        while len(active_clients)==0:
            active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active()]
        selected_clients = sample(self, all_clients=active_clients)
        return selected_clients
    return sample_with_active

# communication phase
def with_dropout(communicate):
    def communicate_with_dropout(self, selected_clients):
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if not self.clients[selected_clients[i]].is_drop()]
        return communicate(self, self.selected_clients)
    return communicate_with_dropout

def with_latency(communicate):
    def communicate_with_latency(self, selected_clients):
        client_latencies = [self.clients[cid].get_network_latency() for cid in selected_clients]
        time_sync = min(max(client_latencies), self.TIME_LATENCY_BOUND)
        self.virtual_clock['time_sync'].append(time_sync)
        return communicate(self, selected_clients)
    return communicate_with_latency

def with_incomplete_update(communicate):
    def communicate_with_incomplete_update(self, selected_clients):
        original_local_num_steps = [self.clients[cid].num_steps for cid in selected_clients]
        update_local_computing_resource(self, selected_clients)
        res = communicate(self, selected_clients)
        for cid,onum_steps in zip(selected_clients, original_local_num_steps):
            self.clients[cid].num_steps = onum_steps
        return res
    return communicate_with_incomplete_update