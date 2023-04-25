import collections

import numpy as np
import torch
from flgo.utils import fmodule

from .fedbase import BasicServer
from .fedbase import BasicClient as Client
import multiprocessing as mp


class CloudServer(BasicServer):
    def __init__(self, option):
        super(CloudServer, self).__init__(option)
        self.init_algo_para({'local_num_rounds': 1})
        self.local_num_rounds = self.algo_para['local_num_rounds']
        self.edge_servers = []

    def run(self):
        """
        Running the FL symtem where the global model is trained and evaluated iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        if self.eval_interval > 0:
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.log_once()
            self.gv.logger.time_end('Eval Time Cost')
        while self.current_round <= self.num_rounds:
            self.gv.clock.step()
            # iterate
            updated = self.iterate()
            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                self.gv.logger.info("--------------Cloud Round {} & Edge Round {}--------------".format(self.current_round, self.current_round * self.local_num_rounds))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.log_once()
                    self.gv.logger.time_end('Eval Time Cost')
                # check if early stopping
                if self.gv.logger.early_stop(): break
                self.current_round += 1
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

    def iterate(self):
        """
        The standard iteration of each federated communication round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.

        Returns:
            False if the global model is not updated in this iteration
        """
        # training
        models = self.communicate([edid for edid in range(len(self.edge_servers))])['model']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models)
        return len(models) > 0

    def aggregate(self, models: list, *args, **kwargs):
        if len(models) == 0: return self.model
        local_data_vols = [es.datavol for es in self.edge_servers]
        total_data_vol = sum(local_data_vols)
        if self.aggregation_option == 'weighted_scale':
            p = [1.0 * local_data_vols[esid] / total_data_vol for esid in self.received_edge_servers]
            K = len(models)
            N = self.num_edge_servers
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.aggregation_option == 'uniform':
            return fmodule._model_average(models)
        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * local_data_vols[esid] / total_data_vol for esid in self.received_edge_servers]
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0 - sum(p)) * self.model + w
        else:
            p = [1.0 * local_data_vols[esid] / total_data_vol for esid in self.received_edge_servers]
            sump = sum(p)
            p = [pk / sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def communicate(self, selected_edge_servers, mtype=0, asynchronous=False):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.

        Args:
            selected_clients (list of int): the clients to communicate with
            mtype (anytype): type of message
            asynchronous (bool): asynchronous communciation or synchronous communcation

        Returns:
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_edge_servers = []
        received_package_buffer = {}
        communicate_edge_servers = list(set(selected_edge_servers))
        # prepare packages for edge servers
        for esid in communicate_edge_servers:
            received_package_buffer[esid] = None
        # communicate with selected edge servers
        for esid in communicate_edge_servers:
            server_pkg = self.pack(esid, mtype)
            server_pkg['__mtype__'] = mtype
            response_from_es_id = self.communicate_with(self.edge_servers[esid].id, package=server_pkg)
            packages_received_from_edge_servers.append(response_from_es_id)
        for i, esid in enumerate(communicate_edge_servers): received_package_buffer[esid] = packages_received_from_edge_servers[i]
        packages_received_from_edge_servers = [received_package_buffer[esid] for esid in selected_edge_servers if
                                          received_package_buffer[esid]]
        self.received_edge_servers = selected_edge_servers
        return self.unpack(packages_received_from_edge_servers)

    def communicate_with(self, target_id, package={}):
        return super(BasicServer, self).communicate_with(target_id, package)

    def register_edge_servers(self, edge_servers):
        self.register_objects(edge_servers, 'edge_servers')
        self.num_edge_servers = len(edge_servers)
        for esid, es in enumerate(self.edge_servers):
            es.edge_server_id = esid
        for es in self.edge_servers: es.register_server(self)

    def global_test(self, model=None, flag:str='valid'):
        if model is None: model=self.model
        all_metrics = collections.defaultdict(list)
        for es in self.edge_servers:
            es_metrics = es.global_test(model, flag)
            for met_name, met_val in es_metrics.items():
                all_metrics[met_name].extend(met_val)
        return all_metrics

    @property
    def available_clients(self):
        return None


class EdgeServer(BasicServer):
    def __init__(self, option):
        super(EdgeServer, self).__init__(option)
        self.init_algo_para({'local_num_rounds':1})
        self.local_num_rounds = self.algo_para['local_num_rounds']
        self.actions = {0: self.reply}
        self.server = None

    def run(self):
        for round in range(self.local_num_rounds):
            # iterate
            updated = self.iterate()
            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                self.current_round += 1
            # decay learning rate
            self.global_lr_scheduler(self.current_round)
        return

    def reply(self, svr_pkg):
        self.model = self.unpack_from_cloud(svr_pkg)
        self.run()
        cpkg = self.pack_to_cloud(self.model)
        return cpkg

    def unpack_from_cloud(self, packages_received_from_cloud):
        return packages_received_from_cloud['model']

    def pack_to_cloud(self, model):
        return {
            "model": model,
        }

    def register_server(self, server=None):
        r"""
        Register the server to self.server
        """
        self.register_objects([server], 'server_list')
        if server is not None:
            self.server = server

    def communicate_with(self, target_id, package={}):
        return super(BasicServer, self).communicate_with(target_id, package)

    def communicate(self, selected_clients, mtype=0, asynchronous=False):
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        # prepare packages for clients
        for cid in communicate_clients:
            received_package_buffer[cid] = None
        # communicate with selected clients
        if self.num_parallels <= 1:
            # computing iteratively
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                response_from_client_id = self.communicate_with(self.clients[client_id].id, package=server_pkg)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_parallels)
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                self.clients[client_id].update_device(self.gv.apply_for_device())
                args = (int(self.clients[client_id].id), server_pkg)
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=args))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
        for i, cid in enumerate(communicate_clients): received_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if
                                          received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

    def sample(self):
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        # full sampling with unlimited communication resources of the server
        if 'full' in self.sample_option:
            return all_clients
        # sample clients
        elif 'uniform' in self.sample_option:
            # original sample proposed by fedavg
            selected_clients = list(
                np.random.choice(all_clients, min(self.clients_per_round, len(all_clients)), replace=False)) if len(
                all_clients) > 0 else []
        elif 'md' in self.sample_option:
            # the default setting that is introduced by FedProx, where the clients are sampled with the probability in proportion to their local_movielens_recommendation data sizes
            local_data_vols = [self.clients[cid].datavol for cid in all_clients]
            total_data_vol = sum(local_data_vols)
            p = np.array(local_data_vols) / total_data_vol
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=p)) if len(
                all_clients) > 0 else []
        return selected_clients

    @property
    def datavol(self):
        return sum([c.datavol for c in self.clients])






