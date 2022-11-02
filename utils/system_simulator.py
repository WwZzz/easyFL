import collections
import numpy as np
import queue

class ElemClock:
    class Elem:
        def __init__(self, x, time):
            self.x = x
            self.time = time

        def __str__(self):
            return '{} at Time {}'.format(self.x, self.time)

        def __lt__(self, other):
            return self.time < other.time

    def __init__(self):
        self.q = queue.PriorityQueue()
        self.time = 0

    def step(self, delta_t=1):
        if delta_t < 0: raise RuntimeError("Cannot inverse time of systemic_simulator.clock.")
        self.time += delta_t

    def set_time(self, t):
        if t < self.time: raise RuntimeError("Cannot inverse time of systemic_simulator.clock.")
        self.time = t

    def put(self, x, time):
        self.q.put_nowait(self.Elem(x, time))

    def get(self):
        if self.q.empty(): return None
        return self.q.get().x

    def get_until(self, t):
        res = []
        while not self.empty():
            elem = self.q.get()
            if elem.time > t:
                self.put(elem.x, elem.time)
                break
            pkg = elem.x
            res.append(pkg)
        return res

    def get_sofar(self):
        return self.get_until(self.current_time)

    def gets(self):
        if self.empty(): return []
        res = []
        while not self.empty(): res.append(self.q.get())
        res = [rx.x for rx in res]
        return res

    def clear(self):
        while not self.empty():
            self.get()

    def empty(self):
        return self.q.empty()

    @ property
    def current_time(self):
        return self.time

def seed_generator(seed=0):
    while True:
        yield seed+1
        seed+=1

clock = None
random_seed_gen = None
random_module = None

def update_systemic_state(iterate):
    def iterate_with_changable_state(self):
        global random_module
        random_module = np.random.RandomState(next(random_seed_gen))
        # update the generator of clients' availability, connectivity and completeness
        # +++++++++++++++++++ availability +++++++++++++++++++++
        # ensure there is at least one client to be available during sampling phase
        update_client_availability(self, random_module=random_module)
        available_clients = []
        while len(available_clients) == 0:
            available_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active(random_module)]
        # +++++++++++++++++++ connectivity +++++++++++++++++++++
        # decide whether the clients will drop out once being selected
        update_client_connectivity(self, random_module=random_module)
        # +++++++++++++++++++ completeness +++++++++++++++++++++
        update_client_completeness(self, random_module=random_module)
        # iterate
        return iterate(self)
    return iterate_with_changable_state

def update_client_availability(server, random_module=np.random):
    return

def update_client_connectivity(server, random_module=np.random):
    return

def update_client_completeness(server, random_module=np.random):
    return

#================================================Decorators==========================================
# Time Counter for any function which forces the `clock` to
# step one unit of time once the decorated function is called
def time_step(f):
    def f_timestep(*args, **kwargs):
        global clock
        clock.step()
        return f(*args, **kwargs)
    return f_timestep

# Decorators for three types of system heterogeneity respectively
# in sampling, communicating and local training phase.

# sampling phase
def with_availability(sample):
    def sample_with_availability(self):
        available_clients = [cid for cid in range(self.num_clients) if self.clients[cid].active]
        # call the original sampling function
        selected_clients = sample(self)
        # filter the selected but unavailable clients
        effective_clients = set(selected_clients).intersection(set(available_clients))
        # return the selected and available clients (e.g. sampling with replacement should be considered here)
        selected_clients = [cid for cid in selected_clients if cid in effective_clients]
        return selected_clients
    return sample_with_availability

# communicating phase: broadcast
def with_dropout(communicate):
    def communicate_with_dropout(self, selected_clients):
        if len(selected_clients)==0: return communicate(self, self.selected_clients)
        # ensure that there is at least one non-dropping client in the selected_clients
        global random_module
        random_module = np.random.RandomState(next(random_seed_gen))
        effective_clients = []
        while(len(effective_clients)==0):
            effective_clients = [cid for cid in set(selected_clients) if not self.clients[cid].is_drop(random_module)]
        # simulating the client drop out by not communicating with them
        self.selected_clients = [cid for cid in selected_clients if cid in effective_clients]
        self.dropped_clients = [cid for cid in selected_clients if cid not in effective_clients]
        return communicate(self, self.selected_clients)
    return communicate_with_dropout

# communicating phase: broadcast
def with_clock(communicate):
    def communicate_with_clock(self, selected_clients):
        if len(selected_clients)==0: return communicate(self, selected_clients)
        global clock
        res = communicate(self, selected_clients)
        if '__t' not in res.keys(): res['__t'] = [self.clients[cid].response_latency for cid in self.selected_clients]
        if '__cid' not in res.keys(): res['__cid'] = self.selected_clients
        pkgs = [{key:vi[id] for key,vi in res.items()} for id in range(len(list(res.values())[0]))]
        for pi in pkgs: clock.put(pi, pi['__t'])
        eff_pkgs = clock.get_until(clock.current_time + self.tolerance_for_latency)
        eff_cids = [pkg_i['__cid'] for pkg_i in eff_pkgs]
        # compute delta of time
        delta_t = self.tolerance_for_latency if (self.tolerance_for_latency==0 or len(eff_pkgs)<len(pkgs) or len(self.dropped_clients)>0) else (eff_pkgs[-1]['__t']-clock.current_time)
        clock.step(delta_t)
        # update selected clients if with synchronized communicating
        if self.tolerance_for_latency > 0:
            clock.clear()
            overdue_clients = list(set([cid for cid in selected_clients if cid not in eff_cids]))
            self.dropped_clients.extend(overdue_clients)
            self.selected_clients = [cid for cid in selected_clients if cid in eff_cids]
            # resort effective packages
            pkg_map = {pkg_i['__cid']: pkg_i for pkg_i in eff_pkgs}
            eff_pkgs = [pkg_map[cid] for cid in self.selected_clients]
        return self.unpack(eff_pkgs)
    return communicate_with_clock

# communicating phase: single point communication
def with_latency(communicate_with):
    def delayed_communicate_with(self, client_id):
        global clock
        res = communicate_with(self, client_id)
        res['__cid'] = client_id
        res['__t'] = clock.current_time + self.clients[client_id].response_latency
        return res
    return delayed_communicate_with

# local training phase
def with_completeness(train):
    def train_with_incomplete_update(self, model, *args, **kwargs):
        old_num_steps = self.num_steps
        self.num_steps = self.effective_num_steps
        res = train(self, model, *args, **kwargs)
        self.num_steps = old_num_steps
        return res
    return train_with_incomplete_update

################################### Availability Mode ##########################################
def ideal_client_availability(server, *args, **kwargs):
    for c in server.clients: c.network_active_rate = 1

def y_max_first_client_availability(server, beta=0.1):
    """
    This setting follows the activity mode in 'Fast Federated Learning in the
    Presence of Arbitrary Device Unavailability' , where each client ci will be ready
    for joining in a round with a static probability:
        pi = alpha * min({label kept by ci}) / max({all labels}) + ( 1 - alpha )
    and the participation of client is independent across rounds. The string mode
    should be like 'YMaxFirst-x' where x should be replaced by a float number.
    """
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
    def label_counter(dataset):
        return collections.Counter([int(dataset[di][-1]) for di in range(len(dataset))])
    label_num = len(label_counter(server.test_data))
    for c in server.clients:
        c.network_drop_rate = 0
        c_counter = label_counter(c.train_data + c.valid_data)
        c_label = [lb for lb in c_counter.keys()]
        c.network_active_rate = (beta * min(c_label) / max(1, label_num - 1)) + (1 - beta)

def more_data_first_client_availability(server, beta=0.0001):
    """
    Clients with more data will have a larger active rate at each round.
    e.g. ci=tanh(-|Di| ln(beta+epsilon)), pi=ci/cmax, beta ∈ [0,1)
    """
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.00001
    p = np.array(server.local_data_vols)
    p = p ** beta
    maxp = np.max(p)
    for c, pc in zip(server.clients, p):
        c.network_active_rate = pc / maxp

def less_data_first_client_availability(server, beta=0.5):
    """
    Clients with less data will have a larger active rate at each round.
            ci=(1-beta)^(-|Di|), pi=ci/cmax, beta ∈ [0,1)
    """
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
    prop = np.array(server.local_data_vols)
    prop = prop ** (-beta)
    maxp = np.max(prop)
    for c, pc in zip(server.clients, prop):
        c.network_active_rate = pc / maxp

def y_fewer_first_client_availability(server, beta=0.2):
    """
    Clients with fewer kinds of labels will owe a larger active rate.
        ci = |set(Yi)|/|set(Y)|, pi = beta*ci + (1-beta)
    """
    label_num = len(set([int(server.test_data[di][-1]) for di in range(len(server.test_data))]))
    for c in server.clients:
        train_set = set([int(c.train_data[di][-1]) for di in range(len(c.train_data))])
        valid_set = set([int(c.valid_data[di][-1]) for di in range(len(c.valid_data))])
        label_set = train_set.union(valid_set)
        c.network_active_rate = beta * len(label_set) / label_num + (1 - beta)

def homogeneous_client_availability(server, beta=0.2):
    """
    All the clients share a homogeneous active rate `1-beta` where beta ∈ [0,1)
    """
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.8
    for c in server.clients:
        c.network_active_rate = 1 - beta

def lognormal_client_availability(server, beta=0.1):
    """The following two settings are from 'Federated Learning Under Intermittent
    Client Availability and Time-Varying Communication Constraints' (http://arxiv.org/abs/2205.06730).
        ci ~ logmal(0, lognormal(0, -ln(1-beta)), pi=ci/cmax
    """
    epsilon = 0.000001
    Tks = [np.random.lognormal(0, -np.log(1 - beta - epsilon)) for _ in server.clients]
    max_Tk = max(Tks)
    for c, Tk in zip(server.clients, Tks):
        c.network_active_rate = 1.0 * Tk / max_Tk

def sin_lognormal_client_availability(server, beta=0.1):
    """This setting shares the same active rate distribution with LogNormal, however, the active rates are
    also influenced by the time (i.e. communication round). The active rates obey a sin wave according to the
    time with period T.
        ci ~ logmal(0, lognormal(0, -ln(1-beta)), pi=ci/cmax, p(i,t)=(0.4sin((1+R%T)/T*2pi)+0.5) * pi
    """
    # beta = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
    epsilon = 0.000001
    Tks = [np.random.lognormal(0, -np.log(1 - beta - epsilon)) for _ in server.clients]
    max_Tk = max(Tks)
    for c, Tk in zip(server.clients, Tks):
        c._qk = 1.0 * Tk / max_Tk
        c.network_active_rate = 1
        c.network_drop_rate = 0
        c.time_response = 0
    global update_client_availability
    def f(server, random_module=np.random):
        T = 24
        times = np.linspace(start=0, stop=2 * np.pi, num=T)
        fts = 0.4 * np.sin(times) + 0.5
        t = server.current_round % T
        for c in server.clients:
            c.network_active_rate = fts[t] * c._qk
    update_client_availability = f

def y_cycle_client_availability(server, beta=0.5):
    # beta = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.5
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
    global update_client_availability
    def f(server, random_module=np.random):
        T = 24
        r = 1.0 * (1 + server.current_round % T) / T
        for c in server.clients:
            ic = int(r >= (1.0 * c._min_label / max_label) and r <= (1.0 * c._max_label / max_label))
            c.network_active_rate = beta * ic + (1 - beta)
    update_client_availability = f

################################### Connectivity Mode ##########################################
def ideal_client_connectivity(server, *args, **kwargs):
    for c in server.clients: c.network_drop_rate = 0

################################### Completeness Mode ##########################################
def ideal_client_completeness(server, *args, **kwargs):
    return

def part_dynamic_uniform_client_completeness(server, p=0.5):
    """
    This setting follows the setting in the paper 'Federated Optimization in Heterogeneous Networks'
    (http://arxiv.org/abs/1812.06127). The `p` specifies the number of selected clients with
    incomplete updates.
    """
    def f(server, random_module=np.random):
        incomplete_clients = random_module.choice(server.clients, round(len(server.clients) * p), replace=False)
        for cid in incomplete_clients:
            server.clients[cid].effective_num_steps = random_module.randint(low=1, high=server.clients[cid].num_steps)
        return
    global update_client_completeness
    update_client_completeness = f
    return

def full_static_unifrom_client_completeness(server):
    for c in server.clients:
        c.num_steps = max(1, int(c.num_steps * np.random.rand()))
    return

def arbitrary_dynamic_unifrom_client_completeness(server, a=1, b=1):
    """
    This setting follows the setting in the paper 'Tackling the Objective Inconsistency Problem in
    Heterogeneous Federated Optimization' (http://arxiv.org/abs/2007.07481). The string `mode` should be like
    'FEDNOVA-Uniform(a,b)' where `a` is the minimal value of the number of local epochs and `b` is the maximal
    value. If this mode is active, the `num_epochs` and `num_steps` of clients will be disable.
    """
    a = min(a, 1)
    b = max(b, a)
    def f(server, random_module=np.random):
        for cid in server.clients:
            server.clients[cid].set_local_epochs(random_module.randint(low=a, high=b))
            server.clients[cid].effective_num_steps = server.clients[cid].num_steps
        return
    global update_client_completeness
    update_client_completeness = f
    return

def arbitrary_static_unifrom_client_completeness(server, a=1, b=1):
    """
    This setting follows the setting in the paper 'Tackling the Objective Inconsistency Problem in
    Heterogeneous Federated Optimization' (http://arxiv.org/abs/2007.07481). The string `mode` should be like
    'FEDNOVA-Uniform(a,b)' where `a` is the minimal value of the number of local epochs and `b` is the maximal
    value. If this mode is active, the `num_epochs` and `num_steps` of clients will be disable.
    """
    a = min(a, 1)
    b = max(b, a)
    for cid in server.clients:
        server.clients[cid].set_local_epochs(np.random.randint(low=a, high=b))
    return

################################### Timeliness Mode ############################################
def ideal_client_timeliness(server, *args, **kwargs):
    for c in server.clients:
        c.response_latency = 0
    if server.tolerance_for_latency != 0:
        server.tolerance_for_latency = max([c.response_latency for c in server.clients])

def lognormal_client_timeliness(server, mean_latency=100, var_latency=10):
    mu = np.log(mean_latency) - 0.5 * np.log(1 + var_latency / mean_latency / mean_latency)
    sigma = np.sqrt(np.log(1 + var_latency / mean_latency / mean_latency))
    client_latency = np.random.lognormal(mu, sigma, len(server.clients))
    client_latency = [int(ct) for ct in client_latency]
    for c, ltc in zip(server.clients, client_latency):
        c.response_latency = ltc
    if server.tolerance_for_latency != 0:
        server.tolerance_for_latency = max([c.response_latency for c in server.clients])

def uniform_client_timeliness(server, min_latency=0, max_latency=1):
    for c in server.clients:
        c.response_latency = np.random.randint(low=min_latency, high=max_latency)
    if server.tolerance_for_latency != 0:
        server.tolerance_for_latency = max([c.response_latency for c in server.clients])

#************************************************************************************************
availability_modes = {
    'IDL': ideal_client_availability,
    'YMF': y_max_first_client_availability,
    'MDF': more_data_first_client_availability,
    'LDF': less_data_first_client_availability,
    'YFF': y_fewer_first_client_availability,
    'HOMO': homogeneous_client_availability,
    'LN': lognormal_client_availability,
    'SLN': sin_lognormal_client_availability,
    'YC': y_cycle_client_availability,
}

connectivity_modes = {
    'IDL': ideal_client_connectivity,
}

completeness_modes = {
    'IDL': ideal_client_completeness,
    'PDU': part_dynamic_uniform_client_completeness,
    'FSU': full_static_unifrom_client_completeness,
    'ADU': arbitrary_dynamic_unifrom_client_completeness,
    'ASU': arbitrary_static_unifrom_client_completeness,
}

timeliness_modes = {
    'IDL': ideal_client_timeliness,
    'LN': lognormal_client_timeliness,
    'UNI': uniform_client_timeliness,
}

def get_mode(mode_string):
    mode = mode_string.split('-')
    mode, para = mode[0], mode[1:]
    if len(para) > 0: para = [float(pi) for pi in para]
    return mode, tuple(para)

def init_system_environment(server, option):
    # the random module of systemic simulator
    global random_seed_gen
    global random_module
    global clock
    random_seed_gen = seed_generator(option['seed'])
    random_module = np.random.RandomState(next(random_seed_gen))
    clock = ElemClock()

    # +++++++++++++++++++++ availability +++++++++++++++++++++
    avl_mode, avl_para  = get_mode(option['availability'])
    if avl_mode not in availability_modes: avl_mode, avl_para = 'IDL', ()
    availability_modes[avl_mode.upper()](server, *avl_para)

    # +++++++++++++++++++++ connectivity +++++++++++++++++++++
    con_mode, con_para = get_mode(option['connectivity'])
    if con_mode not in connectivity_modes: con_mode, con_para = 'IDL', ()
    connectivity_modes[con_mode.upper()](server, *con_para)

    # +++++++++++++++++++++ completeness +++++++++++++++++++++
    cmp_mode, cmp_para = get_mode(option['completeness'])
    if cmp_mode not in completeness_modes: cmp_mode, cmp_para = 'IDL', ()
    completeness_modes[cmp_mode.upper()](server, *cmp_para)

    # +++++++++++++++++++++ timeliness ++++++++++++++++++++++++
    ltc_mode, ltc_para = get_mode(option['timeliness'])
    if ltc_mode not in timeliness_modes: ltc_mode, ltc_para = 'IDL', ()
    timeliness_modes[ltc_mode.upper()](server, *ltc_para)
    return
