"""This is a non-official implementation of 'Federated Learning with Buffered Asynchronous Aggregation' (http://arxiv.org/abs/2106.06639). """
from .fedasync import Server as AsyncServer
from .fedbase import BasicClient
import copy

class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'period': 1, 'k': 0, 'K': 10})
        self.tolerance_for_latency = 1000
        self.updated = True
        self.accumulate_delta = None

    def iterate(self):
        # Scheduler periodically triggers the idle clients to locally train the model
        self.selected_clients = self.sample() if (self.gv.clock.current_time % self.period) == 0 or self.gv.clock.current_time == 1 else []
        if len(self.selected_clients) > 0:
            self.gv.logger.info('Select clients {} at time {}'.format(self.selected_clients, self.gv.clock.current_time))
        # Record the timestamp of the selected clients
        #  for cid in self.selected_clients: self.client_taus[cid] = self.current_round
        # Check the currently received models
        res = self.communicate(self.selected_clients, asynchronous=True)
        received_models = res['model']
        received_client_ids = res['__cid']
        # if reveive client update
        if len(received_models) > 0:
            self.gv.logger.info('Receive new models from clients {} at time {}'.format(received_client_ids, self.gv.clock.current_time))
            flag = False
            if self.accumulate_delta == None: flag = True
            for id, model_k in enumerate(received_models):
                if flag == True:
                        self.accumulate_delta = model_k
                        flag = False
                else:
                    self.accumulate_delta += model_k
                self.k += 1
                if self.k == self.K:
                    self.accumulate_delta = self.accumulate_delta * pow(self.K, -1)
                    self.model = self.model - self.lr * self.accumulate_delta
                    self.accumulate_delta = None
                    flag = True
                    self.k = 0
            # update aggregation round and the flag `updated`
            self.current_round += 1
            self.updated = True
        return



class Client(BasicClient):

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        model_0 = copy.deepcopy(model)
        self.train(model)
        model_q = model_0 - model
        cpkg = self.pack(model_q)
        return cpkg