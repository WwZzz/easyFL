r"""
This simulator supports for the following system heterogeneity:

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

responsiveness_modes = {
    'IDL': ideal_client_responsiveness,
    'LN': lognormal_client_responsiveness,
    'UNI': uniform_client_responsiveness,
}

"""

from flgo.simulator.base import BasicSimulator
import random
import numpy as np
import collections

################################### Initial Availability Mode ##########################################
def ideal_client_availability(simulator, *args, **kwargs):
    probs1 = [1. for _ in simulator.clients]
    probs2 = [0. for _ in simulator.clients]
    simulator.set_variable(simulator.all_clients, 'prob_available', probs1)
    simulator.set_variable(simulator.all_clients, 'prob_unavailable', probs2)
    return

def y_max_first_client_availability(simulator, beta=0.1):
    """
    This setting follows the activity mode in 'Fast Federated Learning in the
    Presence of Arbitrary Device Unavailability' , where each client ci will be ready
    for joining in a round with a static probability:
        pi = beta * min({label kept by ci}) / max({all labels}) + ( 1 - beta )
    and the participation of client is independent across rounds. The string mode
    should be like 'YMaxFirst-x' where x should be replaced by a float number.
    """
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
    def label_counter(dataset):
        return collections.Counter([int(dataset[di][-1]) for di in range(len(dataset))])
    label_num = len(label_counter(simulator.server.test_data))
    probs = []
    for c in simulator.get_clients():
        c_counter = label_counter((c.train_data + c.val_data) if c.val_data is not None else c.train_data)
        c_label = [lb for lb in c_counter.keys()]
        probs.append((beta * min(c_label) / max(1, label_num - 1)) + (1 - beta))
    simulator.set_variable(simulator.all_clients, 'prob_available', probs)
    simulator.set_variable(simulator.all_clients, 'prob_unavailable', [1 - p for p in probs])
    simulator.roundwise_fixed_availability = True
    return

def more_data_first_client_availability(simulator, beta=0.0001):
    """
    Clients with more data will have a larger active rate at each round.
    e.g. ci=tanh(-|Di| ln(beta+epsilon)), pi=ci/cmax, beta ∈ [0,1)
    """
    p = np.array([len(c.train_data) for c in simulator.server.clients])
    p = p ** beta
    maxp = np.max(p)
    probs = p/maxp
    simulator.set_variable(simulator.all_clients, 'prob_available', probs)
    simulator.set_variable(simulator.all_clients, 'prob_unavailable', [1 - p for p in probs])
    simulator.roundwise_fixed_availability = True

def less_data_first_client_availability(simulator, beta=0.5):
    """
    Clients with less data will have a larger active rate at each round.
            ci=(1-beta)^(-|Di|), pi=ci/cmax, beta ∈ [0,1)
    """
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
    prop = np.array([len(c.train_data) for c in simulator.server.clients])
    prop = prop ** (-beta)
    maxp = np.max(prop)
    probs = prop/maxp
    simulator.set_variable(simulator.all_clients, 'prob_available', probs)
    simulator.set_variable(simulator.all_clients, 'prob_unavailable', [1 - p for p in probs])
    simulator.roundwise_fixed_availability = True

def y_fewer_first_client_availability(simulator, beta=0.2):
    """
    Clients with fewer kinds of labels will owe a larger active rate.
        ci = |set(Yi)|/|set(Y)|, pi = beta*ci + (1-beta)
    """
    label_num = len(set([int(simulator.server.test_data[di][-1]) for di in range(len(simulator.server.test_data))]))
    probs = []
    for c in simulator.server.clients:
        train_set = set([int(c.train_data[di][-1]) for di in range(len(c.train_data))])
        val_set = set([int(c.val_data[di][-1]) for di in range(len(c.val_data))])
        label_set = train_set.union(val_set)
        probs.append(beta * len(label_set) / label_num + (1 - beta))
    simulator.set_variable(simulator.all_clients, 'prob_available', probs)
    simulator.set_variable(simulator.all_clients, 'prob_unavailable', [1 - p for p in probs])
    simulator.roundwise_fixed_availability = True

def homogeneous_client_availability(simulator, beta=0.2):
    """
    All the clients share a homogeneous active rate `1-beta` where beta ∈ [0,1)
    """
    # alpha = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.8
    probs = [1.-beta for _ in simulator.clients]
    simulator.set_variable(simulator.all_clients, 'prob_available', probs)
    simulator.set_variable(simulator.all_clients, 'prob_unavailable', [1 - p for p in probs])
    simulator.roundwise_fixed_availability = True

def lognormal_client_availability(simulator, beta=0.1):
    """The following two settings are from 'Federated Learning Under Intermittent
    Client Availability and Time-Varying Communication Constraints' (http://arxiv.org/abs/2205.06730).
        ci ~ logmal(0, lognormal(0, -ln(1-beta)), pi=ci/cmax
    """
    epsilon = 0.000001
    Tks = [np.random.lognormal(0, -np.log(1 - beta - epsilon)) for _ in simulator.clients]
    max_Tk = max(Tks)
    probs = np.array(Tks)/max_Tk
    simulator.set_variable(simulator.all_clients, 'prob_available', probs)
    simulator.set_variable(simulator.all_clients, 'prob_unavailable', [1 - p for p in probs])
    simulator.roundwise_fixed_availability = True

def sin_lognormal_client_availability(simulator, beta=0.1):
    """This setting shares the same active rate distribution with LogNormal, however, the active rates are
    also influenced by the time (i.e. communication round). The active rates obey a sin wave according to the
    time with period T.
        ci ~ logmal(0, lognormal(0, -ln(1-beta)), pi=ci/cmax, p(i,t)=(0.4sin((1+R%T)/T*2pi)+0.5) * pi
    """
    # beta = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.1
    epsilon = 0.000001
    Tks = [np.random.lognormal(0, -np.log(1 - beta - epsilon)) for _ in simulator.clients]
    max_Tk = max(Tks)
    q = np.array(Tks)/max_Tk
    simulator.set_variable(simulator.all_clients, 'q', q)
    simulator.set_variable(simulator.all_clients, 'prob_available', q)
    def f(self):
        T = 24
        times = np.linspace(start=0, stop=2 * np.pi, num=T)
        fts = 0.4 * np.sin(times) + 0.5
        t = self.server.current_round % T
        q = self.get_variable(self.all_clients, 'q')
        probs = [fts[t]*qi for qi in q]
        self.set_variable(self.all_clients, 'prob_available', probs)
        self.set_variable(self.all_clients, 'prob_unavailable', [1 - p for p in probs])
    simulator.roundwise_fixed_availability = True
    return f

def y_cycle_client_availability(simulator, beta=0.5):
    # beta = float(mode[mode.find('-') + 1:]) if mode.find('-') != -1 else 0.5
    max_label = max(set([int(simulator.server.test_data[di][-1]) for di in range(len(simulator.server.test_data))]))
    for c in simulator.get_clients():
        train_set = set([int(c.train_data[di][-1]) for di in range(len(c.train_data))])
        val_set = set([int(c.val_data[di][-1]) for di in range(len(c.val_data))])
        label_set = train_set.union(val_set)
        c._min_label = min(label_set)
        c._max_label = max(label_set)
    def f(self):
        T = 24
        r = 1.0 * (1 + self.server.current_round % T) / T
        probs = []
        for c in self.clients:
            ic = int(r >= (1.0 * c._min_label / max_label) and r <= (1.0 * c._max_label / max_label))
            probs.append(beta * ic + (1 - beta))
        self.set_variable(self.all_clients, 'prob_available', probs)
        self.set_variable(self.all_clients, 'prob_unavailable', [1 - p for p in probs])
    simulator.roundwise_fixed_availability = True
    return f

################################### Initial Connectivity Mode ##########################################
def ideal_client_connectivity(simulator, *args, **kwargs):
    probs = [0. for _ in simulator.clients]
    simulator.set_variable(simulator.all_clients, 'prob_drop', probs)

def homogeneous_client_connectivity(simulator, gamma=0.05):
    probs = [gamma for _ in simulator.clients]
    simulator.set_variable(simulator.all_clients, 'prob_drop', probs)

def uniform_client_connectivity(simulator, gamma=0.5):
    probs = [np.random.uniform(0,1) if np.random.rand()<gamma else 0.0 for _ in simulator.all_clients]
    simulator.set_variable(simulator.all_clients, 'prob_drop', probs)

################################### Initial Completeness Mode ##########################################
def ideal_client_completeness(simulator, *args, **kwargs):
    simulator.set_variable(simulator.all_clients, 'working_amount', [c.num_steps for c in simulator.get_clients()])
    return

def part_dynamic_uniform_client_completeness(simulator, p=0.5):
    """
    This setting follows the setting in the paper 'Federated Optimization in Heterogeneous Networks'
    (http://arxiv.org/abs/1812.06127). The `p` specifies the number of selected clients with
    incomplete updates.
    """
    simulator.prob_incomplete = p
    def f(self, client_ids = []):
        was = []
        for cid in client_ids:
            wa = self.random_module.randint(low=0, high=self.clients[cid].num_steps) if self.random_module.rand() < self.prob_incomplete else self.clients[cid].num_steps
            wa = max(1, wa)
            was.append(wa)
            self.clients[cid].num_steps = wa
        self.set_variable(client_ids, 'working_amount', was)
        return
    return f

def full_static_unifrom_client_completeness(simulator):
    working_amounts = [max(1, int(c.num_steps * np.random.rand())) for c in simulator.get_clients()]
    simulator.set_variable(simulator.all_clients, 'working_amount', working_amounts)
    return

def arbitrary_dynamic_unifrom_client_completeness(simulator, a=1, b=1):
    """
    This setting follows the setting in the paper 'Tackling the Objective Inconsistency Problem in
    Heterogeneous Federated Optimization' (http://arxiv.org/abs/2007.07481). The string `mode` should be like
    'FEDNOVA-Uniform(a,b)' where `a` is the minimal value of the number of local_movielens_recommendation epochs and `b` is the maximal
    value. If this mode is active, the `num_epochs` and `num_steps` of clients will be disable.
    """
    simulator._incomplete_a = min(a, 1)
    simulator._incomplete_b = max(b, simulator._incomplete_a)
    def f(self, client_ids = []):
        for cid in client_ids:
            self.clients[cid].set_local_epochs(self.random_module.randint(low=self._incomplete_a, high=self._incomplete_b))
        working_amounts = [self.clients[cid].num_steps for cid in self.all_clients]
        self.set_variable(self.all_clients, 'working_amount', working_amounts)
        return
    return f

def arbitrary_static_unifrom_client_completeness(simulator, a=1, b=1):
    """
    This setting follows the setting in the paper 'Tackling the Objective Inconsistency Problem in
    Heterogeneous Federated Optimization' (http://arxiv.org/abs/2007.07481). The string `mode` should be like
    'FEDNOVA-Uniform(a,b)' where `a` is the minimal value of the number of local_movielens_recommendation epochs and `b` is the maximal
    value. If this mode is active, the `num_epochs` and `num_steps` of clients will be disable.
    """
    a = min(a, 1)
    b = max(b, a)
    for cid in simulator.clients:
        simulator.clients[cid].set_local_epochs(np.random.randint(low=a, high=b))
    working_amounts = [simulator.clients[cid].num_steps for cid in simulator.all_clients]
    simulator.set_variable(simulator.all_clients, 'working_amount', working_amounts)
    return

################################### Initial Timeliness Mode ############################################
def ideal_client_responsiveness(simulator, *args, **kwargs):
    latency = [0 for _ in simulator.clients]
    for c, lt in zip(simulator.clients, latency): simulator.clients[c]._latency = lt
    simulator.set_variable(simulator.all_clients, 'latency', latency)

def lognormal_client_responsiveness(simulator, mean_latency=100, var_latency=10):
    mu = np.log(mean_latency) - 0.5 * np.log(1 + var_latency / mean_latency / mean_latency)
    sigma = np.sqrt(np.log(1 + var_latency / mean_latency / mean_latency))
    client_latency = np.random.lognormal(mu, sigma, len(simulator.clients))
    latency = [int(ct) for ct in client_latency]
    for c, lt in zip(simulator.clients, latency): simulator.clients[c]._latency = lt
    simulator.set_variable(simulator.all_clients, 'latency', latency)

def uniform_client_responsiveness(simulator, min_latency=0, max_latency=1):
    latency = [np.random.randint(low=min_latency, high=max_latency) for _ in simulator.clients]
    for c,lt in zip(simulator.clients, latency): simulator.clients[c]._latency = lt
    simulator.set_variable(simulator.all_clients, 'latency', latency)

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
    'UNI': uniform_client_connectivity,
}

completeness_modes = {
    'IDL': ideal_client_completeness,
    'PDU': part_dynamic_uniform_client_completeness,
    'FSU': full_static_unifrom_client_completeness,
    'ADU': arbitrary_dynamic_unifrom_client_completeness,
    'ASU': arbitrary_static_unifrom_client_completeness,
}

responsiveness_modes = {
    'IDL': ideal_client_responsiveness,
    'LN': lognormal_client_responsiveness,
    'UNI': uniform_client_responsiveness,
}

class Simulator(BasicSimulator):
    def __init__(self, objects, option = None):
        super().__init__(objects)
        self.option = option
        # +++++++++++++++++++++ availability +++++++++++++++++++++
        avl_mode, avl_para = self.get_mode(option['availability'])
        if avl_mode not in availability_modes: avl_mode, avl_para = 'IDL', ()
        f_avl = availability_modes[avl_mode](self, *avl_para)
        if f_avl is not None: self.__class__.update_client_availability = f_avl
        # +++++++++++++++++++++ connectivity +++++++++++++++++++++
        con_mode, con_para = self.get_mode(option['connectivity'])
        if con_mode not in connectivity_modes: con_mode, con_para = 'IDL', ()
        f_con = connectivity_modes[con_mode](self, *con_para)
        if f_con is not None: self.__class__.update_client_connectivity = f_con
        # +++++++++++++++++++++ completeness +++++++++++++++++++++
        cmp_mode, cmp_para = self.get_mode(option['completeness'])
        if cmp_mode not in completeness_modes: cmp_mode, cmp_para = 'IDL', ()
        f_cmp = completeness_modes[cmp_mode](self, *cmp_para)
        if f_cmp is not None: self.__class__.update_client_completeness = f_cmp
        # +++++++++++++++++++++ responsiveness ++++++++++++++++++++++++
        rsp_mode, rsp_para = self.get_mode(option['responsiveness'])
        if rsp_mode not in responsiveness_modes: rsp_mode, rsp_para = 'IDL', ()
        f_rsp = responsiveness_modes[rsp_mode](self, *rsp_para)
        if f_rsp is not None: self.__class__.update_client_responsiveness = f_rsp
        if self.server.tolerance_for_latency == 0:
            self.server.tolerance_for_latency = max([c._latency for c in self.clients])
        return

    def get_mode(self, mode_string):
        mode = mode_string.split('-')
        mode, para = mode[0], mode[1:]
        if len(para) > 0: para = [float(pi) for pi in para]
        return mode, tuple(para)