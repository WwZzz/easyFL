import collections
import numpy as np

class VirtualClock:
    _TIME_UNIT = 1
    def __init__(self):
        self.time_cost = []

    def add_time_cost(self,t=0):
        self.time_cost.append(t*self._TIME_UNIT)
        return

    @property
    def time(self):
        return np.sum(self.time_cost)*self._TIME_UNIT

random_seed_gen = None
clock = VirtualClock()

def seed_generator(seed=0):
    while True:
        yield seed+1
        seed+=1

def update_activity(server, random_module=np.random):
    return

def update_dropout(server, clients=[], random_module=np.random):
    return

def update_local_computing_resource(server, clients=[], random_module=np.random):
    return

def update_response_time(server, clients=[], random_module=np.random):
    return

def init_network_mode(server, mode='ideal'):
    global update_activity
    if mode=='ideal':
        for c in server.clients:
            c.network_active_rate = 1
            c.network_drop_rate = 0
            c.time_response = 0

    elif mode.startswith('YMinFirst'):
        """
        This setting follows the activity mode in 'Fast Federated Learning in the 
        Presence of Arbitrary Device Unavailability' , where each client ci will be ready
        for join in a communcation round with the probability:
            pi = alpha * min({label kept by ci}) / max({all labels}) + ( 1 - alpha )
        and the participation of client is independent for different rounds. The string mode
        should be like 'YMinFirst-x' where x should be replaced by a float number.
        """
        alpha = float(mode[mode.find('-')+1:]) if mode.find('-')!=-1 else 0.1
        def label_counter(dataset):
            return collections.Counter([int(dataset[di][-1]) for di in range(len(dataset))])
        label_num = len(label_counter(server.test_data))
        for c in server.clients:
            c.network_drop_rate = 0
            c.time_response = 0
            c_counter = label_counter(c.train_data+c.valid_data)
            c_label = [lb for lb in c_counter.keys()]
            c.network_active_rate = (alpha * min(c_label) / max(1, label_num-1)) + (1 - alpha)

    elif mode.startswith('MoreDataFirst'):
        """
        Clients with more data will have a larger active rate at each round.
        e.g. ci=tanh(-|Di| ln(alpha+epsilon)), pi=ci/cmax, alpha ∈ [0,1)
        """
        alpha = float(mode[mode.find('-')+1:]) if mode.find('-')!=-1 else 0.00001
        p = np.array(server.local_data_vols)
        p = p**alpha
        maxp = np.max(p)
        for c, pc in zip(server.clients, p):
            c.network_active_rate = pc / maxp
            c.network_drop_rate = 0
            c.time_response = 0

    elif mode.startswith('LessDataFirst'):
        """
        Clients with less data will have a larger active rate at each round.
                ci=(1-alpha)^(-|Di|), pi=ci/cmax, alpha ∈ [0,1)
        """
        alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
        prop = np.array(server.local_data_vols)
        prop = prop ** (-alpha)
        maxp = np.max(prop)
        for c, pc in zip(server.clients, prop):
            c.network_active_rate = pc/maxp
            c.network_drop_rate = 0
            c.time_response = 0

    elif mode.startswith('FewerYFirst'):
        """
        Clients with fewer kinds of labels will owe a larger active rate.
            ci = |set(Yi)|/|set(Y)|, pi = alpha*ci + (1-alpha)
        """
        alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
        label_num = len(set([int(server.test_data[di][-1]) for di in range(len(server.test_data))]))
        for c in server.clients:
            c.network_drop_rate = 0
            c.time_response = 0
            train_set = set([int(c.train_data[di][-1]) for di in range(len(c.train_data))])
            valid_set = set([int(c.valid_data[di][-1]) for di in range(len(c.valid_data))])
            label_set = train_set.union(valid_set)
            c.network_active_rate = alpha * len(label_set) / label_num + (1 - alpha)

    elif mode.startswith('Homogeneous'):
        """
        All the clients share a homogeneous active rate `1-alpha` where alpha ∈ [0,1)
        """
        alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.8
        for c in server.clients:
            c.network_active_rate = 1 - alpha
            c.network_drop_rate = 0
            c.time_response = 0

    elif mode.startswith('LogNormal'):
        """The following two settings are from 'Federated Learning Under Intermittent 
        Client Availability and Time-Varying Communication Constraints' (http://arxiv.org/abs/2205.06730).
            ci ~ logmal(0, lognormal(0, -ln(1-alpha)), pi=ci/cmax
        """
        alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
        epsilon=0.000001
        Tks = [np.random.lognormal(0,-np.log(1-alpha-epsilon)) for _ in server.clients]
        max_Tk = max(Tks)
        for c,Tk in zip(server.clients, Tks):
            c.network_active_rate = 1.0 * Tk / max_Tk
            c.network_drop_rate = 0
            c.time_response = 0

    elif mode.startswith('SinLogNormal'):
        """This setting shares the same active rate distribution with LogNormal, however, the active rates are 
        also influenced by the time (i.e. communication round). The active rates obey a sin wave according to the 
        time with period T.
            ci ~ logmal(0, lognormal(0, -ln(1-alpha)), pi=ci/cmax, p(i,t)=(0.4sin((1+R%T)/T*2pi)+0.5) * pi
        """
        alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
        epsilon=0.000001
        Tks = [np.random.lognormal(0,-np.log(1-alpha-epsilon)) for _ in server.clients]
        max_Tk = max(Tks)
        for c,Tk in zip(server.clients, Tks):
            c._qk = 1.0 * Tk / max_Tk
            c.network_active_rate = 1
            c.network_drop_rate = 0
            c.time_response = 0
        def f(server, random_module=np.random):
            T = 24
            times = np.linspace(start=0, stop=2*np.pi, num=T)
            fts = 0.4 * np.sin(times) + 0.5
            t = server.current_round % T
            for c in server.clients:
                c.network_active_rate = fts[t] * c._qk
        update_activity = f

    elif mode.startswith('YCycle'):
        alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.5
        max_label = max(set([int(server.test_data[di][-1]) for di in range(len(server.test_data))]))
        for c in server.clients:
            c.network_drop_rate = 0
            c.time_response = 0
            train_set = set([int(c.train_data[di][-1]) for di in range(len(c.train_data))])
            valid_set = set([int(c.valid_data[di][-1]) for di in range(len(c.valid_data))])
            label_set = train_set.union(valid_set)
            c._min_label = min(label_set)
            c._max_label = max(label_set)
            c.network_active_rate = 1
        def f(server, random_module=np.random):
            T = 24
            r = 1.0*(1+server.current_round%T)/T
            for c in server.clients:
                ic = int(r>=(1.0*c._min_label/max_label) and r<=(1.0*c._max_label/max_label))
                c.network_active_rate = alpha*ic + (1-alpha)
        update_activity = f

    else:
        for c in server.clients:
            c.network_active_rate = 1
            c.network_drop_rate = 0
            c.time_response = 0
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
        def f(server, clients=[], random_module=np.random):
            incomplete_clients = random_module.choice(clients, round(len(clients)*p), replace=False)
            for cid in incomplete_clients:
                server.clients[cid].num_steps = random_module.randint(low=1, high=server.clients[cid].num_steps)
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
        def f(server, clients=[], random_module=np.random):
            for cid in server.clients:
                server.clients[cid].set_local_epochs(random_module.randint(low=a, high=b))
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
    global random_seed_gen
    random_seed_gen = seed_generator(option['seed'])

# sampling phase
def with_inactivity(sample):
    def sample_with_active(self):
        global random_seed_gen
        random_module = np.random.RandomState(next(random_seed_gen))
        # refresh the client's active rate per round
        update_activity(self, random_module)
        active_clients = []
        # ensure there are at least one client being available
        while len(active_clients)==0:
            active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active(random_module)]
        # call the original sampling function
        selected_clients = sample(self)
        # filter the selected but unavailable clients
        effective_clients = set(selected_clients).intersection(set(active_clients))
        time_cost = 1
        # wait for the activity of those unavailable if the due of waiting for the client activity is larger than 1 time unit (i.e. )
        if self.due_active>1:
            active_stamp = [list(effective_clients)]
            # wait until the time expired or all the clients are active
            for t in range(1, self.due_active):
                active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active(random_module)]
                client_to_be_active = set(selected_clients).difference(effective_clients)
                if len(client_to_be_active)==0:
                    break
                new_effective_clients = list(client_to_be_active.intersection(active_clients))
                effective_clients = effective_clients.union(new_effective_clients)
            time_cost = t
        # add time cost to clock
        clock.add_time_cost(time_cost)
        # return the selected and available clients (e.g. sampling with replacement should be considered here)
        selected_clients = [cid for cid in selected_clients if cid in effective_clients]
        return selected_clients
    return sample_with_active

# communication phase
def with_dropout(communicate):
    def communicate_with_dropout(self, selected_clients):
        global random_seed_gen
        random_module = np.random.RandomState(next(random_seed_gen))
        # refresh the connectivity of clients
        for c in self.clients: c.dropped = False
        update_dropout(self, selected_clients, random_module)
        # ensure there is at least one client won't dropout
        while True:
            selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if not self.clients[selected_clients[i]].is_drop(random_module)]
            if len(selected_clients)>0: break
        # simulating the client drop out by not communicating with them
        return communicate(self, selected_clients)
    return communicate_with_dropout

def with_due(communicate):
    def communicate_with_due(self, selected_clients):
        global random_seed_gen
        # update the response time per round
        random_module = np.random.RandomState(next(random_seed_gen))
        update_response_time(self, selected_clients, random_module)
        # calculate the time of response of each selected client (the dropped )
        client_time_response = [self.clients[cid].get_time_response() for cid in self.selected_clients]
        # take the maximum time cost of response
        if np.any([t>self.due_response for t in client_time_response]):
            time_cost = self.due_response
        else:
            time_cost = np.max(client_time_response)
        # add time cost to virtual clock
        clock.add_time_cost(time_cost)
        # filter the clients will be over due or dropped out \ change server.selected_clients to ensure that the received packages can match the selected clients
        self.selected_clients = [cid for cid,time in zip(self.selected_clients, client_time_response) if time<=self.due_response]
        return communicate(self, self.selected_clients)
    return communicate_with_due

def with_incomplete_update(communicate):
    def communicate_with_incomplete_update(self, selected_clients):
        # refresh clients' computing resource per round
        global random_seed_gen
        random_module = np.random.RandomState(next(random_seed_gen))
        original_local_num_steps = [self.clients[cid].num_steps for cid in selected_clients]
        update_local_computing_resource(self, selected_clients, random_module=random_module)
        res = communicate(self, selected_clients)
        # reset clients' computing resource to the initial state
        for cid,onum_steps in zip(selected_clients, original_local_num_steps):
            self.clients[cid]._taken_computing_resource = 1.0 * self.clients[cid].num_steps
            self.clients[cid].num_steps = onum_steps
        return res
    return communicate_with_incomplete_update