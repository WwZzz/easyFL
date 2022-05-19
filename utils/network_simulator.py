"""This module is designed for simulating the network environment of the real world.
Decorators are written here to decorate the originnal methods owned by the central server.
Here we implement three different network heterogeneity for a FL system:
    1. with_accessibility: some clients are not available during the stage of sampling
    2. with_latency: accumulating latencies of clients and dropout the overdue clients
"""
def init_active_probability_distribution(clients):
    return

def init_dropping_probability_distribution(clients):
    return

def init_latency_amount_distribution(clients):
    return

def init_network_environment(server):
    init_active_probability_distribution(server.clients)
    init_dropping_probability_distribution(server.clients)
    init_latency_amount_distribution(server.clients)

def with_accessibility(sample):
    def sample_with_active(self, *args, **kargs):
        selected_clients = sample(self, *args, **kargs)
        selected_clients, time_access = self.wait_for_accessibility(selected_clients)
        # collect all the active clients at this round and wait for at least one client is active
        self.virtual_clock['time_access'].append(time_access)
        return selected_clients
    return sample_with_active

def with_latency(communicate):
    def communicate_under_network_latency(self, selected_clients):
        client_latencies = [self.clients[cid].get_network_latency() for cid in selected_clients]
        # drop clients if dropping out
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if not self.clients[selected_clients[i]].is_drop()]
        time_sync = min(max(client_latencies), self.TIME_LATENCY_BOUND)
        self.virtual_clock['time_sync'].append(time_sync)
        return communicate(self, self.selected_clients)
    return communicate_under_network_latency
