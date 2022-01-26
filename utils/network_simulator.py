"""This module is designed for simulating the network environment of the real world.
Decorators are written here to decorate the originnal methods owned by the central server.
Here we implement three different network heterogeneity for a FL system:
    1. deactivate_clients: some clients are not available during the stage of sampling
    2. drop_client: some clients have dropped out during the stage of the server waiting
       for clients' responses.
    3. network_latency: clients differ in the time cost of communicating
"""

def deactivate_client(sample):
    def sample_with_active(self, *args, **kargs):
        selected_clients = sample(self, *args, **kargs)
        # collect all the active clients at this round and wait for at least one client is active
        active_clients = []
        while (len(active_clients) < 1):
            active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active()]
        # drop the selected but inactive clients
        res = []
        for cid in selected_clients:
            if cid in active_clients:
                res.append(cid)
        return res
    return sample_with_active

def drop_client(communicate_with):
    def communicate_with_client_not_dropped(self, client_id):
        if self.clients[client_id].is_drop(): return None
        else: return communicate_with(self, client_id)
    return communicate_with_client_not_dropped

def under_network_latency(communicate):
    def communicate_under_network_latency(self, selected_clients):
        res = communicate(self, selected_clients)
        if selected_clients==[]:
            self.virtual_clock.append(self._max_waiting_time)
        else:
            client_latencies = [self.clients[cid].get_network_latency() for cid in selected_clients]
            self.virtual_clock.append(max(client_latencies))
        return res
    return communicate_under_network_latency




