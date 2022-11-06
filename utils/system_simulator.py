import collections
import numpy as np
import queue
import math

clock = None
random_seed_gen = None
random_module = None
state_updater = None

def update_client_availability(state_updater, *args, **kwargs):
    return

def update_client_connectivity(state_updater, *args, **kwargs):
    return

def update_client_completeness(state_updater, *args, **kwargs):
    return

def update_client_timeliness(state_updater, *args, **kwargs):
    return

def seed_generator(seed=0):
    while True:
        yield seed+1
        seed+=1

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
        self.state_updater = None

    def step(self, delta_t=1):
        if delta_t < 0: raise RuntimeError("Cannot inverse time of system_simulator.clock.")
        if self.state_updater is not None:
            for t in range(delta_t):
                self.state_updater.flush()
        self.time += delta_t

    def set_time(self, t):
        if t < self.time: raise RuntimeError("Cannot inverse time of system_simulator.clock.")
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

    def register_state_updater(self, state_updater):
        self.state_updater = state_updater

class BasicStateUpdater:

    _STATE = ['offline', 'idle', 'selected', 'working', 'dropped']
    def __init__(self, server, clients):
        self.server = server
        self.clients = clients
        self.all_clients = list(range(len(self.clients)))
        self.random_module = np.random.RandomState(next(random_seed_gen))
        # client states and the variables
        self.client_states = ['idle' for _ in self.clients]
        self.roundly_fixed_availability = False
        self.availability_latest_round = -1
        self.variables = [{
            'prob_available': 1.,
            'prob_unavailable': 0.,
            'prob_drop': 0.,
            'working_amount': c.num_steps,
            'latency': 0,
        } for c in self.clients]
        self.state_counter = [{'dropped_counter': 0, 'latency_counter': 0, } for _ in self.clients]

    def get_client_with_state(self, state='idle'):
        return [cid for cid, cstate in enumerate(self.client_states) if cstate == state]

    def set_client_state(self, client_ids, state):
        if state not in self._STATE: raise RuntimeError('{} not in the default state'.format(state))
        if type(client_ids) is not list: client_ids = [client_ids]
        for cid in client_ids: self.client_states[cid] = state
        if state == 'dropped':
            self.set_client_dropped_counter(client_ids)
        if state == 'working':
            self.set_client_latency_counter(client_ids)

    def set_client_latency_counter(self, client_ids = []):
        if type(client_ids) is not list: client_ids = [client_ids]
        global clock
        for cid in client_ids:
            self.state_counter[cid]['dropped_counter'] = 0
            self.state_counter[cid]['latency_counter'] = self.variables[cid]['latency']

    def set_client_dropped_counter(self, client_ids = []):
        if type(client_ids) is not list: client_ids = [client_ids]
        global clock
        for cid in client_ids:
            self.state_counter[cid]['latency_counter'] = 0
            self.state_counter[cid]['dropped_counter'] = self.server.get_tolerance_for_latency()

    @property
    def idle_clients(self):
        return self.get_client_with_state('idle')

    @property
    def working_clients(self):
        return self.get_client_with_state('working')

    @property
    def offline_clients(self):
        return self.get_client_with_state('offline')

    @property
    def selected_clients(self):
        return self.get_client_with_state('selected')

    @property
    def dropped_clients(self):
        return self.get_client_with_state('dropped')

    def get_variable(self, client_ids, varname):
        if len(self.variables) ==0 or varname not in self.variables[0].keys(): return None
        if type(client_ids) is not list: client_ids = [client_ids]
        return [self.variables[cid][varname] for cid in client_ids]

    def set_variable(self, client_ids, varname, values):
        if type(client_ids) is not list: client_ids = [client_ids]
        assert len(client_ids) == len(values)
        for cid, v in zip(client_ids, values):
            self.variables[cid][varname] = v

    def flush(self):
        # +++++++++++++++++++ availability +++++++++++++++++++++
        # change self.variables[cid]['prob_available'] and self.variables[cid]['prob_unavailable'] for each client `cid`
        update_client_availability(self)
        # +++++++++++++++++++ connectivity +++++++++++++++++++++
        # change self.variables[cid]['prob_drop'] for each client `cid`
        update_client_connectivity(self)
        # +++++++++++++++++++ completeness +++++++++++++++++++++
        # change self.variables[cid]['working_amount'] for each client `cid`
        update_client_completeness(self)
        # +++++++++++++++++++ timeliness +++++++++++++++++++++++
        # change self.variables[cid]['latency'] for each client `cid`
        update_client_timeliness(self)
        # update states for offline & idle clients
        if len(self.idle_clients)==0 or not self.roundly_fixed_availability or self.server.current_round > self.availability_latest_round:
            self.availability_latest_round = self.server.current_round
            for cid in self.offline_clients:
                self.clients[cid].available = (self.random_module.rand() <= self.variables[cid]['prob_available'])
                if self.clients[cid].is_available(): self.client_states[cid] = 'idle'
            for cid in self.idle_clients:
                self.clients[cid].available = (self.random_module.rand() > self.variables[cid]['prob_unavailable'])
                if not self.clients[cid].is_available(): self.client_states[cid] = 'offline'
        # update states for dropped clients
        for cid in self.dropped_clients:
            self.state_counter[cid]['dropped_counter'] -= 1
            if self.state_counter[cid]['dropped_counter'] <= 0:
                self.state_counter[cid]['dropped_counter'] = 0
                self.client_states[cid] = 'offline'
        # update states for working clients
        for cid in self.working_clients:
            self.state_counter[cid]['latency_counter'] -= 1
            if self.state_counter[cid]['latency_counter'] <=0:
                self.state_counter[cid]['latency_counter'] = 0
                self.client_states[cid] = 'idle'

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
        global clock
        global state_updater
        available_clients = state_updater.idle_clients
        # ensure that there is at least one client to be available at the current moment
        while len(available_clients) == 0:
            clock.step()
            available_clients = state_updater.idle_clients
        # call the original sampling function
        selected_clients = sample(self)
        # filter the selected but unavailable clients
        effective_clients = set(selected_clients).intersection(set(available_clients))
        # return the selected and available clients (e.g. sampling with replacement should be considered here)
        self._unavailable_selected_clients = [cid for cid in selected_clients if cid not in effective_clients]
        selected_clients = [cid for cid in selected_clients if cid in effective_clients]
        state_updater.set_client_state(selected_clients, 'selected')
        return selected_clients
    return sample_with_availability

# communicating phase: broadcast
def with_dropout(communicate):
    def communicate_with_dropout(self, selected_clients):
        if len(selected_clients) > 0:
            global state_updater
            probs_drop = state_updater.get_variable(selected_clients, 'prob_drop')
            self._dropped_selected_clients = [cid for cid,prob in zip(selected_clients, probs_drop) if state_updater.random_module.rand()<=prob]
            state_updater.set_client_state(self._dropped_selected_clients, 'dropped')
            self.selected_clients = [cid for cid in selected_clients if cid not in self._dropped_selected_clients]
        return communicate(self, self.selected_clients)
    return communicate_with_dropout

# communicating phase: broadcast
def with_clock(communicate):
    def communicate_with_clock(self, selected_clients):
        global clock
        global state_updater
        res = communicate(self, selected_clients)
        if len(selected_clients)==0:
            if hasattr(self, '_dropped_selected_clients') and len(self._dropped_selected_clients)>0:
                clock.step(self.get_tolerance_for_latency())
            return res
        state_updater.set_client_state(selected_clients, 'working')
        latencies = state_updater.get_variable(selected_clients, 'latency')
        for cid, lt in zip(selected_clients, latencies):
            self.clients[cid]._latency = lt
        # Check if the returned package has the attribute `__t` and `__cid`. If not, assign the attributes to the packages.
        if '__t' not in res.keys(): res['__t'] = [clock.current_time + self.clients[cid]._latency for cid in self.selected_clients]
        if '__cid' not in res.keys(): res['__cid'] = self.selected_clients
        # Convert the unpacked packages to a list of packages of each client.
        pkgs = [{key:vi[id] for key,vi in res.items()} for id in range(len(list(res.values())[0]))]
        # Put the packages into a queue according to their arrival time `__t`
        for pi in pkgs: clock.put(pi, pi['__t'])
        tolerance_for_latency = self.get_tolerance_for_latency()
        # Wait for client packages. If communicating in asynchronous way, the waiting time is 0.
        if self.asynchronous:
            eff_pkgs = clock.get_until(clock.current_time)
        else:
            eff_pkgs = clock.get_until(clock.current_time + tolerance_for_latency)
            # Compute delta of time for the communication.
            delta_t = tolerance_for_latency if len(eff_pkgs)<len(pkgs) or len(self._dropped_selected_clients)>0 else (eff_pkgs[-1]['__t']-clock.current_time)
            clock.step(delta_t)
            eff_cids = [pkg_i['__cid'] for pkg_i in eff_pkgs]
            clock.clear()
            self._overdue_clients = list(set([cid for cid in selected_clients if cid not in eff_cids]))
            state_updater.set_client_state(self._overdue_clients, 'idle')
            for cid in self._overdue_clients:
                state_updater.state_counter[cid]['latency_counter'] = 0
            self.selected_clients = [cid for cid in selected_clients if cid in eff_cids]
            # Resort effective packages
            pkg_map = {pkg_i['__cid']: pkg_i for pkg_i in eff_pkgs}
            eff_pkgs = [pkg_map[cid] for cid in self.selected_clients]
        return self.unpack(eff_pkgs)
    return communicate_with_clock

# communicating phase: single point communication
def with_latency(communicate_with):
    def delayed_communicate_with(self, client_id):
        global clock
        global state_updater
        res = communicate_with(self, client_id)
        res['__cid'] = client_id
        res['__t'] = clock.current_time + state_updater.get_variable(client_id, 'latency')[0]
        return res
    return delayed_communicate_with

# local training phase
def with_completeness(train):
    def train_with_incomplete_update(self, model, *args, **kwargs):
        global state_updater
        old_num_steps = self.num_steps
        self.num_steps = state_updater.get_variable(self.id, 'working_amount')[0]
        self._effective_num_steps = self.num_steps
        res = train(self, model, *args, **kwargs)
        self.num_steps = old_num_steps
        return res
    return train_with_incomplete_update

################################### Initial Availability Mode ##########################################
def ideal_client_availability(server, *args, **kwargs):
    global state_updater
    probs1 = [1. for _ in server.clients]
    probs2 = [0. for _ in server.clients]
    state_updater.set_variable(state_updater.all_clients, 'prob_available', probs1)
    state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', probs2)

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
    global state_updater
    def label_counter(dataset):
        return collections.Counter([int(dataset[di][-1]) for di in range(len(dataset))])
    label_num = len(label_counter(server.test_data))
    probs = []
    for c in server.clients:
        c_counter = label_counter(c.train_data + c.valid_data)
        c_label = [lb for lb in c_counter.keys()]
        probs.append((beta * min(c_label) / max(1, label_num - 1)) + (1 - beta))
    state_updater.set_variable(state_updater.all_clients, 'prob_available', probs)
    state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', [1 - p for p in probs])
    state_updater.roundly_fixed_availability = True

def more_data_first_client_availability(server, beta=0.0001):
    """
    Clients with more data will have a larger active rate at each round.
    e.g. ci=tanh(-|Di| ln(beta+epsilon)), pi=ci/cmax, beta ∈ [0,1)
    """
    global state_updater
    p = np.array(server.local_data_vols)
    p = p ** beta
    maxp = np.max(p)
    probs = p/maxp
    state_updater.set_variable(state_updater.all_clients, 'prob_available', probs)
    state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', [1 - p for p in probs])
    state_updater.roundly_fixed_availability = True

def less_data_first_client_availability(server, beta=0.5):
    """
    Clients with less data will have a larger active rate at each round.
            ci=(1-beta)^(-|Di|), pi=ci/cmax, beta ∈ [0,1)
    """
    global state_updater
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
    prop = np.array(server.local_data_vols)
    prop = prop ** (-beta)
    maxp = np.max(prop)
    probs = prop/maxp
    state_updater.set_variable(state_updater.all_clients, 'prob_available', probs)
    state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', [1 - p for p in probs])
    state_updater.roundly_fixed_availability = True

def y_fewer_first_client_availability(server, beta=0.2):
    """
    Clients with fewer kinds of labels will owe a larger active rate.
        ci = |set(Yi)|/|set(Y)|, pi = beta*ci + (1-beta)
    """
    global state_updater
    label_num = len(set([int(server.test_data[di][-1]) for di in range(len(server.test_data))]))
    probs = []
    for c in server.clients:
        train_set = set([int(c.train_data[di][-1]) for di in range(len(c.train_data))])
        valid_set = set([int(c.valid_data[di][-1]) for di in range(len(c.valid_data))])
        label_set = train_set.union(valid_set)
        probs.append(beta * len(label_set) / label_num + (1 - beta))
    state_updater.set_variable(state_updater.all_clients, 'prob_available', probs)
    state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', [1 - p for p in probs])
    state_updater.roundly_fixed_availability = True

def homogeneous_client_availability(server, beta=0.2):
    """
    All the clients share a homogeneous active rate `1-beta` where beta ∈ [0,1)
    """
    global state_updater
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.8
    probs = [1.-beta for _ in server.clients]
    state_updater.set_variable(state_updater.all_clients, 'prob_available', probs)
    state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', [1 - p for p in probs])
    state_updater.roundly_fixed_availability = True

def lognormal_client_availability(server, beta=0.1):
    """The following two settings are from 'Federated Learning Under Intermittent
    Client Availability and Time-Varying Communication Constraints' (http://arxiv.org/abs/2205.06730).
        ci ~ logmal(0, lognormal(0, -ln(1-beta)), pi=ci/cmax
    """
    global state_updater
    epsilon = 0.000001
    Tks = [np.random.lognormal(0, -np.log(1 - beta - epsilon)) for _ in server.clients]
    max_Tk = max(Tks)
    probs = np.array(Tks)/max_Tk
    state_updater.set_variable(state_updater.all_clients, 'prob_available', probs)
    state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', [1 - p for p in probs])
    state_updater.roundly_fixed_availability = True

def sin_lognormal_client_availability(server, beta=0.1):
    """This setting shares the same active rate distribution with LogNormal, however, the active rates are
    also influenced by the time (i.e. communication round). The active rates obey a sin wave according to the
    time with period T.
        ci ~ logmal(0, lognormal(0, -ln(1-beta)), pi=ci/cmax, p(i,t)=(0.4sin((1+R%T)/T*2pi)+0.5) * pi
    """
    # beta = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
    global state_updater
    epsilon = 0.000001
    Tks = [np.random.lognormal(0, -np.log(1 - beta - epsilon)) for _ in server.clients]
    max_Tk = max(Tks)
    q = np.array(Tks)/max_Tk
    state_updater.set_variable(state_updater.all_clients, 'q', q)
    state_updater.set_variable(state_updater.all_clients, 'prob_available', q)
    def f(state_updater):
        T = 24
        times = np.linspace(start=0, stop=2 * np.pi, num=T)
        fts = 0.4 * np.sin(times) + 0.5
        t = state_updater.server.current_round % T
        q = state_updater.get_variable(state_updater.all_clients, 'q')
        probs = [fts[t]*qi for qi in q]
        state_updater.set_variable(state_updater.all_clients, 'prob_available', probs)
        state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', [1 - p for p in probs])
    global update_client_availability
    update_client_availability = f
    state_updater.roundly_fixed_availability = True

def y_cycle_client_availability(server, beta=0.5):
    # beta = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.5
    max_label = max(set([int(server.test_data[di][-1]) for di in range(len(server.test_data))]))
    for c in server.clients:
        train_set = set([int(c.train_data[di][-1]) for di in range(len(c.train_data))])
        valid_set = set([int(c.valid_data[di][-1]) for di in range(len(c.valid_data))])
        label_set = train_set.union(valid_set)
        c._min_label = min(label_set)
        c._max_label = max(label_set)
    global state_updater
    def f(state_updater):
        T = 24
        r = 1.0 * (1 + state_updater.server.current_round % T) / T
        probs = []
        for c in state_updater.clients:
            ic = int(r >= (1.0 * c._min_label / max_label) and r <= (1.0 * c._max_label / max_label))
            probs.append(beta * ic + (1 - beta))
        state_updater.set_variable(state_updater.all_clients, 'prob_available', probs)
        state_updater.set_variable(state_updater.all_clients, 'prob_unavailable', [1 - p for p in probs])
    global update_client_availability
    update_client_availability = f
    state_updater.roundly_fixed_availability = True

################################### Initial Connectivity Mode ##########################################
def ideal_client_connectivity(server, *args, **kwargs):
    global state_updater
    probs = [0. for _ in server.clients]
    state_updater.set_variable(state_updater.all_clients, 'prob_drop', probs)

def homogeneous_client_connectivity(server, gamma=0.05):
    global state_updater
    probs = [gamma for _ in server.clients]
    state_updater.set_variable(state_updater.all_clients, 'prob_drop', probs)

################################### Initial Completeness Mode ##########################################
def ideal_client_completeness(server, *args, **kwargs):
    return

def part_dynamic_uniform_client_completeness(server, p=0.5):
    """
    This setting follows the setting in the paper 'Federated Optimization in Heterogeneous Networks'
    (http://arxiv.org/abs/1812.06127). The `p` specifies the number of selected clients with
    incomplete updates.
    """
    def f(state_updater):
        incomplete_clients = state_updater.random_module.choice(state_updater.all_clients, round(len(state_updater.clients) * p), replace=False)
        working_amounts = [state_updater.random_module.randint(low=1, high=state_updater.clients[cid].num_steps) for cid in incomplete_clients]
        state_updater.set_variable(incomplete_clients, 'working_amount', working_amounts)
        return
    global state_updater
    global update_client_completeness
    update_client_completeness = f
    return

def full_static_unifrom_client_completeness(server):
    global state_updater
    working_amounts = [max(1, int(c.num_steps * np.random.rand())) for c in server.clients]
    state_updater.set_variable(state_updater.all_clients, 'working_amount', working_amounts)
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
    def f(state_updater):
        for cid in state_updater.clients:
            state_updater.clients[cid].set_local_epochs(state_updater.random_module.randint(low=a, high=b))
        working_amounts = [state_updater.clients[cid].num_steps for cid in state_updater.all_clients]
        state_updater.set_variable(state_updater.all_clients, 'working_amount', working_amounts)
        return
    global state_updater
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
    global state_updater
    a = min(a, 1)
    b = max(b, a)
    for cid in server.clients:
        server.clients[cid].set_local_epochs(np.random.randint(low=a, high=b))
    working_amounts = [server.clients[cid].num_steps for cid in state_updater.all_clients]
    state_updater.set_variable(state_updater.all_clients, 'working_amount', working_amounts)
    return

################################### Initial Timeliness Mode ############################################
def ideal_client_timeliness(server, *args, **kwargs):
    global state_updater
    latency = [0 for _ in server.clients]
    for c, lt in zip(server.clients, latency): c._latency = lt
    state_updater.set_variable(state_updater.all_clients, 'latency', latency)

def lognormal_client_timeliness(server, mean_latency=100, var_latency=10):
    global state_updater
    mu = np.log(mean_latency) - 0.5 * np.log(1 + var_latency / mean_latency / mean_latency)
    sigma = np.sqrt(np.log(1 + var_latency / mean_latency / mean_latency))
    client_latency = np.random.lognormal(mu, sigma, len(server.clients))
    latency = [int(ct) for ct in client_latency]
    for c, lt in zip(server.clients, latency): c._latency = lt
    state_updater.set_variable(state_updater.all_clients, 'latency', latency)

def uniform_client_timeliness(server, min_latency=0, max_latency=1):
    global state_updater
    latency = [np.random.randint(low=min_latency, high=max_latency) for _ in server.clients]
    for c,lt in zip(server.clients, latency): c._latency = lt
    state_updater.set_variable(state_updater.all_clients, 'latency', latency)

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
    'HOMO': homogeneous_client_connectivity,
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
    global state_updater
    random_seed_gen = seed_generator(option['seed'])
    random_module = np.random.RandomState(next(random_seed_gen))
    clock = ElemClock()
    state_updater = BasicStateUpdater(server, server.clients)
    # +++++++++++++++++++++ availability +++++++++++++++++++++
    avl_mode, avl_para  = get_mode(option['availability'])
    if avl_mode not in availability_modes: avl_mode, avl_para = 'IDL', ()
    availability_modes[avl_mode](server, *avl_para)

    # +++++++++++++++++++++ connectivity +++++++++++++++++++++
    con_mode, con_para = get_mode(option['connectivity'])
    if con_mode not in connectivity_modes: con_mode, con_para = 'IDL', ()
    connectivity_modes[con_mode](server, *con_para)

    # +++++++++++++++++++++ completeness +++++++++++++++++++++
    cmp_mode, cmp_para = get_mode(option['completeness'])
    if cmp_mode not in completeness_modes: cmp_mode, cmp_para = 'IDL', ()
    completeness_modes[cmp_mode](server, *cmp_para)

    # +++++++++++++++++++++ timeliness ++++++++++++++++++++++++
    ltc_mode, ltc_para = get_mode(option['timeliness'])
    if ltc_mode not in timeliness_modes: ltc_mode, ltc_para = 'IDL', ()
    timeliness_modes[ltc_mode](server, *ltc_para)

    clock.register_state_updater(state_updater)
    return
