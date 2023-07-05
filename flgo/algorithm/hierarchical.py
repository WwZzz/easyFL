import collections
from .fedbase import BasicServer
from .fedbase import BasicClient
import torch.multiprocessing as mp
import torch
from tqdm import tqdm

class Server(BasicServer):
    def __init__(self, option):
        super(Server, self).__init__(option)
        self.sample_option = 'full'

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
                self.gv.logger.info("--------------Global Round {}--------------".format(self.current_round))
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

    def global_test(self, model=None, flag:str='valid'):
        if model is None: model=self.model
        all_metrics = collections.defaultdict(list)
        for es in self.clients:
            es_metrics = es.global_test(model, flag)
            for met_name, met_val in es_metrics.items():
                all_metrics[met_name].extend(met_val)
        return all_metrics

    def communicate(self, selected_clients, mtype=0, asynchronous=False):
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
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        # prepare packages for clients
        for client_id in communicate_clients:
            received_package_buffer[client_id] = None
        # communicate with selected clients
        if self.num_parallels <= 1:
            # computing iteratively
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                response_from_client_id = self.communicate_with(self.clients[client_id].id, package=server_pkg)
                packages_received_from_clients.append(response_from_client_id)
        else:
            self.model = self.model.to(torch.device('cpu'))
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_parallels)
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                self.clients[client_id].update_device(self.gv.apply_for_device())
                args = (self.clients[client_id].id, server_pkg)
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=args))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
            self.model = self.model.to(self.device)
            for pkg in packages_received_from_clients:
                for k,v in pkg.items():
                    if hasattr(v, 'to'):
                        try:
                            pkg[k] = v.to(self.device)
                        except:
                            continue
        for i, client_id in enumerate(communicate_clients): received_package_buffer[client_id] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if
                                          received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

class EdgeServer(BasicServer):
    def __init__(self, option):
        super(EdgeServer, self).__init__(option)
        self.num_edge_rounds = option['num_edge_rounds']
        self.actions = {0: self.reply}
        self.server = None
        self.num_clients = 0

    def reply(self, svr_pkg):
        self.model = svr_pkg['model']
        for round in tqdm(range(self.num_edge_rounds)):
            self.iterate()
            self.current_round += 1
            self.global_lr_scheduler(self.current_round)
        return {'model': self.model}

    @property
    def datavol(self):
        return sum([c.datavol for c in self.clients])

    def register_server(self, server=None):
        r"""
        Register the server to self.server
        """
        self.register_objects([server], 'server_list')
        if server is not None:
            self.server = server

    def communicate(self, selected_clients, mtype=0, asynchronous=False):
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
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        # prepare packages for clients
        for client_id in communicate_clients:
            received_package_buffer[client_id] = None
        # communicate with selected clients
        if self.num_parallels <= 1:
            # computing iteratively
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                response_from_client_id = self.communicate_with(self.clients[client_id].id, package=server_pkg)
                packages_received_from_clients.append(response_from_client_id)
        else:
            self.model = self.model.to(torch.device('cpu'))
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_parallels)
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                self.clients[client_id].update_device(self.gv.apply_for_device())
                args = (self.clients[client_id].id, server_pkg)
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=args))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
            self.model = self.model.to(self.device)
            for pkg in packages_received_from_clients:
                for k,v in pkg.items():
                    if hasattr(v, 'to'):
                        try:
                            pkg[k] = v.to(self.device)
                        except:
                            continue
        for i, client_id in enumerate(communicate_clients): received_package_buffer[client_id] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if
                                          received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

Client = BasicClient