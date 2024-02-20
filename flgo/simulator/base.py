import sys
import numpy as np
from collections.abc import Iterable
from abc import ABCMeta, abstractmethod
import functools
import torch
import heapq

import flgo.utils.fmodule


class PriorityQueue:
    r"""Priority Queue"""
    def __init__(self):
        self.queue = []

    def size(self):
        r"""The size of the queue"""
        return len(self.queue)

    def empty(self):
        r"""Return whether the queue is empty"""
        return len(self.queue)==0

    def put(self, item):
        r"""Put item into the queue"""
        heapq.heappush(self.queue, item)

    def get(self):
        r"""Get item from the queue"""
        return heapq.heappop(self.queue)

class AbstractSimulator(metaclass=ABCMeta):
    @abstractmethod
    def flush(self):
        # flush the states for all the things in the system as time steps
        pass

random_seed_gen = None
random_module = None

def seed_generator(seed=0):
    """Return an integer as the seed"""
    while True:
        yield seed+1
        seed+=1

def size_of_package(package):
    r"""
    Compute the size of the package

    Args:
        package (dict): the pacakge

    Returns:
        size (int): the size of the package
    """
    size = 0
    if not isinstance(package, dict): return 0
    for v in package.values():
        size += v.__sizeof__()
    return size

class ElemClock:
    r"""Simulate the clock by the timestamp of each Element"""
    class Elem:
        r"""
        Element with a timestamp

        Args:
            x: element
            time (int): the timestamp
        """
        def __init__(self, x, time):
            self.x = x
            self.time = time

        def __str__(self):
            return '{} at Time {}'.format(self.x, self.time)

        def __lt__(self, other):
            return self.time < other.time

    def __init__(self):
        self.q = PriorityQueue()
        self.time = 0
        self.simulator = None

    def step(self, delta_t=1):
        r"""
        Step delta_t units of the virtual time

        Args:
            delta_t (int): the delta of time
        """
        if delta_t < 0: raise RuntimeError("Cannot inverse time of simulator.base.clock.")
        if self.simulator is not None:
            for t in range(delta_t):
                self.simulator.flush()
        self.time += delta_t

    def set_time(self, t):
        r"""
        Set time

        Args:
            t (int): time
        """
        if t < self.time: raise RuntimeError("Cannot inverse time of simulator.base.clock.")
        self.time = t

    def put(self, x, time):
        r"""
        Put an element into the time queue with timestamp

        Args:
            x: element
            time (int): the timestamp
        """
        self.q.put(self.Elem(x, time))

    def get(self):
        r"""
        Get an element from the queue

        Returns:
            the element in the nearest coming time
        """
        if self.q.empty(): return None
        return self.q.get().x

    def get_until(self, t):
        r"""
        Get elements from the queue until time t

        Args:
            t (int): time

        Returns:
            a list of elements whose timestamps is no larger than t
        """
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
        r"""
        Get elements from the queue until now

        Returns:
            a list of elements whose timestamps is no larger than the current time
        """
        return self.get_until(self.current_time)

    def gets(self):
        r"""
        Get all the elements in the queue

        Returns:
            a list of elements in the queue
        """
        if self.empty(): return []
        res = []
        while not self.empty(): res.append(self.q.get())
        res = [rx.x for rx in res]
        return res

    def clear(self):
        r"""
        Clear the queue
        """
        while not self.empty():
            self.get()

    def conditionally_clear(self, f):
        r"""
        Clear elements if f(element) is False

        Args:
            f (function): a function that receives element and returns bool variable
        """
        buf = []
        while not self.empty(): buf.append(self.q.get())
        for elem in buf:
            if not f(elem.x): self.q.put(elem)
        return

    def empty(self):
        r"""Return whether the queue is empty"""
        return self.q.empty()

    @ property
    def current_time(self):
        r"""Return the current time"""
        return self.time

    def register_simulator(self, simulator):
        r"""Set self.simulator=simulator"""
        self.simulator = simulator

class BasicSimulator(AbstractSimulator):
    r"""
    Simulate the system heterogeneity with the client state machine.

    Args:
        object (list): a list of objects in the federated scenario
    """
    _STATE = ['offline', 'idle', 'selected', 'working', 'dropped']
    _VAR_NAMES = ['prob_available', 'prob_unavailable', 'prob_drop', 'working_amount', 'latency', 'capacity']
    def __init__(self, objects, *args, **kwargs):
        if len(objects)>0:
            self.server = objects[0]
            self.clients = {c.id:c for c in objects[1:]}
        else:
            self.server = None
            self.clients = {}
        self.all_clients = list(self.clients.keys())
        self.random_module = np.random.RandomState(0)
        # client states and the variables
        self.client_states = {cid:'idle' for cid in self.clients}
        self.roundwise_fixed_availability = False
        self.availability_latest_round = -1
        self.variables = {c.id:{
            'prob_available': 1.,
            'prob_unavailable': 0.,
            'prob_drop': 0.,
            'working_amount': c.num_steps,
            'latency': 0,
            'capacity':1.0,
        } for c in self.clients.values()}
        for var in self._VAR_NAMES:
            self.set_variable(self.all_clients, var, [self.variables[cid][var] for cid in self.all_clients])
        self.state_counter = {c:{'dropped_counter': 0, 'latency_counter': 0, } for c in self.clients}

    def initialize(self, *args, **kwargs):
        r"""
        API for initialize some client-specific characteristics
        """
        return

    def get_client_with_state(self, state='idle'):
        r"""
        Get clients according to their states.

        Args:
            state (str): the state in ['offline', 'idle', 'selected', 'working', 'dropped']

        Returns:
            a list of clients whose states are state
        """
        return [cid for cid, cstate in self.client_states.items() if cstate == state]

    def set_client_state(self, client_ids, state):
        r"""
        Set the states of clients in client_ids to the state

        Args:
            client_ids (list): a list of clients' ids
            state (str): the state in ['offline', 'idle', 'selected', 'working', 'dropped']

        Returns:
            a list of clients whose states are state
        """
        if state not in self._STATE: raise RuntimeError('{} not in the default state'.format(state))
        if type(client_ids) is not list: client_ids = [client_ids]
        for cid in client_ids: self.client_states[cid] = state
        if state == 'dropped':
            self.set_client_dropped_counter(client_ids)
        if state == 'working':
            self.set_client_latency_counter(client_ids)
        if state == 'idle':
            self.reset_client_counter(client_ids)

    def set_client_latency_counter(self, client_ids = []):
        r"""Set the latency_counter"""
        if type(client_ids) is not list: client_ids = [client_ids]
        for cid in client_ids:
            self.state_counter[cid]['dropped_counter'] = 0
            self.state_counter[cid]['latency_counter'] = self.variables[cid]['latency']

    def set_client_dropped_counter(self, client_ids = []):
        r"""Set the dropped_counter"""
        if type(client_ids) is not list: client_ids = [client_ids]
        for cid in client_ids:
            self.state_counter[cid]['latency_counter'] = 0
            self.state_counter[cid]['dropped_counter'] = self.server.get_tolerance_for_latency()

    def reset_client_counter(self, client_ids = []):
        r"""Reset the clients' counter"""
        if type(client_ids) is not list: client_ids = [client_ids]
        for cid in client_ids:
            self.state_counter[cid]['dropped_counter'] = self.state_counter[cid]['latency_counter'] = 0
        return

    def get_clients(self, client_ids:list=None):
        """
        Args:
            client_ids (list): a list of client ids
        Returns:
            res (list): a list of client object
        """
        if client_ids is None: return [self.clients[cid] for cid in self.all_clients]
        return [self.clients[cid] for cid in client_ids]

    @property
    def idle_clients(self):
        """Return ideal clients"""
        return self.get_client_with_state('idle')

    @property
    def working_clients(self):
        """Return working clients"""
        return self.get_client_with_state('working')

    @property
    def offline_clients(self):
        """Return offline clients"""
        return self.get_client_with_state('offline')

    @property
    def selected_clients(self):
        """Return the selected clients"""
        return self.get_client_with_state('selected')

    @property
    def dropped_clients(self):
        """Return the dropped clients"""
        return self.get_client_with_state('dropped')

    def get_variable(self, client_ids, varname):
        r"""
        Get the simulator-private variables of the clients in client_ids according to varname

        Args:
            client_ids (list): a list of clients' ids
            varname (str): the name of the simulator-private variable

        Returns:
            the simulator-private variables of the clients in client_ids
        """
        if len(self.variables) ==0: return None
        if type(client_ids) is not list: client_ids = [client_ids]
        return [self.variables[cid][varname] if varname in self.variables[cid].keys() else None for cid in client_ids]

    def set_variable(self, client_ids, varname, values):
        r"""
        Set the simulator-private variables of the clients in client_ids to values

        Args:
            client_ids (list): a list of clients' ids
            varname (str): the name of the simulator-private variable
            values (list): a list of things
        """
        if type(client_ids) is not list: client_ids = [client_ids]
        if not isinstance(values, Iterable): values = [values]
        assert len(client_ids) == len(values)
        for cid, v in zip(client_ids, values):
            self.variables[cid][varname] = v
            setattr(self.clients[cid], '_'+varname, v)

    def update_client_availability(self, *args, **kwargs):
        """API to update client availability every time unit"""
        return

    def update_client_connectivity(self, client_ids, *args, **kwargs):
        """API to update client connectivity every time unit"""
        return

    def update_client_completeness(self, client_ids, *args, **kwargs):
        """API to update client completeness every time unit"""
        return

    def update_client_responsiveness(self, client_ids, *args, **kwargs):
        """API to update client responsiveness every time unit"""
        return

    def update_client_capacity(self, *args, **kwargs):
        """API to update client capacity every time unit"""
        return

    def flush(self):
        """Flush the client state machine as time goes by"""
        # +++++++++++++++++++ availability +++++++++++++++++++++
        # change self.variables[cid]['prob_available'] and self.variables[cid]['prob_unavailable'] for each client `cid`
        self.update_client_availability()
        # change self.variables[cid]['capacity'] for each client
        self.update_client_capacity()
        # update states for offline & idle clients
        if len(self.idle_clients)==0 or not self.roundwise_fixed_availability or self.server.current_round > self.availability_latest_round:
            self.availability_latest_round = self.server.current_round
            offline_clients = {cid: 'offline' for cid in self.offline_clients}
            idle_clients = {cid:'idle' for cid in self.idle_clients}
            for cid in offline_clients:
                if (self.random_module.rand() <= self.variables[cid]['prob_available']): offline_clients[cid] = 'idle'
            for cid in self.idle_clients:
                if  (self.random_module.rand() <= self.variables[cid]['prob_unavailable']): idle_clients[cid] = 'offline'
            new_idle_clients = [cid for cid in offline_clients if offline_clients[cid] == 'idle']
            new_offline_clients = [cid for cid in idle_clients if idle_clients[cid] == 'offline']
            self.set_client_state(new_idle_clients, 'idle')
            self.set_client_state(new_offline_clients, 'offline')
        # update states for dropped clients
        for cid in self.dropped_clients:
            self.state_counter[cid]['dropped_counter'] -= 1
            if self.state_counter[cid]['dropped_counter'] < 0:
                self.state_counter[cid]['dropped_counter'] = 0
                self.client_states[cid] = 'offline'
                if (self.random_module.rand() < self.variables[cid]['prob_unavailable']):
                    self.set_client_state([cid], 'offline')
                else:
                    self.set_client_state([cid], 'idle')
        # Remark: the state transfer fo working clients is instead made once the server received from clients
        # # update states for working clients
        # for cid in self.working_clients:
        #     self.state_counter[cid]['latency_counter'] -= 1
        #     if self.state_counter[cid]['latency_counter'] < 0:
        #         self.state_counter[cid]['latency_counter'] = 0
        #         self.set_client_state([cid], 'offline')

#================================================Decorators==========================================
# Time Counter for any function which forces the `cfg.clock` to
# step one unit of time once the decorated function is called
# def time_step(f):
#     def f_timestep(*args, **kwargs):
#         cfg.clock.step()
#         return f(*args, **kwargs)
#     return f_timestep

# sampling phase
def with_availability(sample):
    r"""
    The decorator for sampling with client availability

    Example:
    ```python
        >>> import flgo.algorithm.fedbase
        >>> import flgo.simulator.base as ss
        >>> class Server(flgo.algorithm.fedbase.BasicServer):
        ...     @ss.with_availability
        ...     def sample(self):
        ...         ...
    ```
    """
    def sample_with_availability(self):
        available_clients = self.gv.simulator.idle_clients
        # ensure that there is at least one client to be available at the current moment
        # while len(available_clients) == 0:
        #     self.gv.clock.step()
        #     available_clients = self.gv.simulator.idle_clients
        # call the original sampling function
        selected_clients = sample(self)
        # filter the selected but unavailable clients
        effective_clients = set(selected_clients).intersection(set(available_clients))
        # return the selected and available clients (e.g. sampling with replacement should be considered here)
        self._unavailable_selected_clients = [cid for cid in selected_clients if cid not in effective_clients]
        if len(self._unavailable_selected_clients)>0:
            self.gv.logger.info('The selected clients {} are not currently available.'.format(self._unavailable_selected_clients))
        selected_clients = [cid for cid in selected_clients if cid in effective_clients]
        self.gv.simulator.set_client_state(selected_clients, 'selected')
        return selected_clients
    return sample_with_availability

# communicating phase
def with_dropout(communicate):
    r"""
    The decorator for communicating to simulate the scene where clients may drop out

    Example:
    ```python
        >>> import flgo.algorithm.fedbase
        >>> import flgo.simulator.base as ss
        >>> class Server(flgo.algorithm.fedbase.BasicServer):
        ...     @ss.with_dropout
        ...     def communicate(self,...):
        ...         ...
    ```
    """
    @functools.wraps(communicate)
    def communicate_with_dropout(self, selected_clients, mtype=0, asynchronous=False):
        if len(selected_clients) > 0:
            self.gv.simulator.update_client_connectivity(selected_clients)
            probs_drop = self.gv.simulator.get_variable(selected_clients, 'prob_drop')
            self._dropped_selected_clients = [cid for cid,prob in zip(selected_clients, probs_drop) if self.gv.simulator.random_module.rand() <= prob]
            self.gv.simulator.set_client_state(self._dropped_selected_clients, 'dropped')
            return communicate(self, [cid for cid in selected_clients if cid not in self._dropped_selected_clients], mtype, asynchronous)
        else:
            return communicate(self, selected_clients, mtype, asynchronous)
    return communicate_with_dropout

# communicating phase
def with_latency(communicate_with):
    r"""
    The decorator to simulate the scene where there are network latencies during communication

    Example:
    ```python
        >>> import flgo.algorithm.fedbase
        >>> import flgo.simulator.base as ss
        >>> class Server(flgo.algorithm.fedbase.BasicServer):
        ...     @ss.with_latency
        ...     def communicate_with(self,...):
        ...         ...
    ```
    """
    @functools.wraps(communicate_with)
    def delayed_communicate_with(self, target_id, package):
        # Calculate latency for the target client
        # Set local_movielens_recommendation model size of clients for computation cost estimation
        if 'model' in package.keys() and isinstance(package['model'], flgo.utils.fmodule.FModule):
            model_size = package['model'].count_parameters(output=False)
        else:
            model_size = 0
        self.gv.simulator.set_variable(target_id, '__model_size', model_size)
        # Set downloading package sizes for clients for downloading cost estimation
        self.gv.simulator.set_variable(target_id, '__download_package_size',size_of_package(package))
        res = communicate_with(self, target_id, package)
        if res is None: res = {}
        # Set uploading package sizes for clients for uploading cost estimation
        self.gv.simulator.set_variable(target_id, '__upload_package_size', size_of_package(res))
        # update latency of the target client according to the communication cost and computation cost
        self.gv.simulator.update_client_responsiveness([target_id])
        # Record the size of the package that may influence the value of the latency
        # Update the real-time latency of the client response
        # Get the updated latency
        latency = self.gv.simulator.get_variable(target_id, 'latency')[0]
        self.clients[target_id]._latency = latency
        res['__cid'] = target_id
        # Compute the arrival time
        res['__t'] = self.gv.clock.current_time + latency
        return res
    return delayed_communicate_with

# local_movielens_recommendation training phase
def with_completeness(train):
    r"""
    The decorator to simulate the scene where the clients may upload incomplete model updates

    Example:
    ```python
        >>> import flgo.algorithm.fedbase
        >>> import flgo.simulator.base as ss
        >>> class Client(flgo.algorithm.fedbase.BasicClient):
        ...     @ss.with_completeness
        ...     def train(self,...):
        ...         ...
    ```
    """
    @functools.wraps(train)
    def train_with_incomplete_update(self, model, *args, **kwargs):
        old_num_steps = self.num_steps
        self.num_steps = self._working_amount
        res = train(self, model, *args, **kwargs)
        self.num_steps = old_num_steps
        return res
    return train_with_incomplete_update

def with_clock(communicate):
    r"""
    The decorator to simulate the scene where there is a virtual global clock

    Example:
    ```python
        >>> import flgo.algorithm.fedbase
        >>> import flgo.simulator.base as ss
        >>> class Server(flgo.algorithm.fedbase.BasicServer):
        ...     @ss.with_clock
        ...     def communicate(self,...):
        ...         ...
    ```
    """
    def communicate_with_clock(self, selected_clients, mtype=0, asynchronous=False):
        self.gv.simulator.update_client_completeness(selected_clients)
        res = communicate(self, selected_clients, mtype, asynchronous)
        # If all the selected clients are unavailable, directly return the result without waiting.
        # Else if all the available clients have dropped out and not using asynchronous communication,  waiting for `tolerance_for_latency` time units.
        tolerance_for_latency = self.get_tolerance_for_latency()
        if not asynchronous and len(selected_clients)==0:
            if hasattr(self, '_dropped_selected_clients') and len(self._dropped_selected_clients)>0:
                self.gv.clock.step(tolerance_for_latency)
            return res
        # Convert the unpacked packages to a list of packages of each client.
        pkgs = [{key: vi[id] for key, vi in res.items()} for id in range(len(list(res.values())[0]))] if len(selected_clients)>0 else []
        if len(pkgs)>0 and pkgs[0].get('__cid', None) is None:
            for cid, pkg in zip(selected_clients, pkgs):
                pkg['__cid'] = cid
        # Put the packages from selected clients into clock only if when there are effective selected clients
        if len(selected_clients)>0:
            # Set selected clients' states as `working`
            self.gv.simulator.set_client_state(selected_clients, 'working')
            for pi in pkgs:
                self.gv.clock.put(pi, pi.get('__t', 0))
        # Receiving packages in asynchronous\synchronous way
        # Wait for client packages. If communicating in asynchronous way, the waiting time is 0.
        if asynchronous:
            # Return the currently received packages to the server
            eff_pkgs = self.gv.clock.get_until(self.gv.clock.current_time)
            eff_cids = [pkg_i['__cid'] for pkg_i in eff_pkgs]
        else:
            # Wait all the selected clients for no more than `tolerance_for_latency` time units.
            # Check if anyone had dropped out or will be overdue
            max_latency = max(self.gv.simulator.get_variable(selected_clients, 'latency'))
            any_drop, any_overdue = (hasattr(self, '_dropped_selected_clients') and len(self._dropped_selected_clients) > 0), (max_latency >  tolerance_for_latency)
            # Compute delta of time for the communication.
            delta_t = tolerance_for_latency if any_drop or any_overdue else max_latency
            # Receive packages within due
            eff_pkgs = self.gv.clock.get_until(self.gv.clock.current_time + delta_t)
            self.gv.clock.step(int(delta_t))
            # Drop the packages of overdue clients and reset their states to `idle`
            eff_cids = [pkg_i['__cid'] for pkg_i in eff_pkgs]
            self._overdue_clients = list(set([cid for cid in selected_clients if cid not in eff_cids]))
            # no additional wait for the synchronous selected clients and preserve the later packages from asynchronous clients
            if len(self._overdue_clients) > 0:
                self.gv.clock.conditionally_clear(lambda x: x['__cid'] in self._overdue_clients)
                self.gv.simulator.set_client_state(self._overdue_clients, 'idle')
            # Resort effective packages
            pkg_map = {pkg_i['__cid']: pkg_i for pkg_i in eff_pkgs}
            eff_pkgs = [pkg_map[cid] for cid in selected_clients if cid in eff_cids]
        self.gv.simulator.set_client_state(eff_cids, 'offline')
        self.received_clients = [pkg_i['__cid'] for pkg_i in eff_pkgs]
        return self.unpack(eff_pkgs)
    return communicate_with_clock
