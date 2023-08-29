import copy
import numpy as np
import flgo.utils.fmodule as fmodule
from flgo.algorithm.fedbase import BasicServer
from flgo.algorithm.fedbase import BasicClient

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'cond': 0, 'time_budget':100, 'K':10})
        self.round_finished = True
        self.buffer = {
            'model': [],
            'round': [],
            't': [],
            'client_id':[],
        }
        self.sampling_timestamp = 0
        self.sample_option = 'uniform_available'

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'round': self.current_round, # model version
        }

    def iterate(self):
        # sampling clients to start a new round \ only listening for new coming models
        if self.round_finished:
            self.selected_clients = self.sample()
            self.sampling_timestamp = self.gv.clock.current_time
            self.round_finished = False
            res = self.communicate(self.selected_clients, asynchronous=True)
        else:
            res = self.communicate([], asynchronous=True)
        if res!={}:
            self.buffer['model'].extend(res['model'])
            self.buffer['round'].extend(res['round'])
            self.buffer['t'].extend([self.gv.clock.current_time for _ in res['model']])
            self.buffer['client_id'].extend(res['__cid'])

        if self.aggregation_condition():
            # update the global model
            stale_clients = []
            stale_rounds = []
            for cid, round in zip(self.buffer['client_id'], self.buffer['round']):
                if round<self.current_round:
                    stale_clients.append(cid)
                    stale_rounds.append(round)
            if len(stale_rounds)>=0:
                self.gv.logger.info('Receiving stale models from clients: {}'.format(stale_clients))
                self.gv.logger.info('The staleness are {}'.format([r-self.current_round for r in stale_rounds]))
                self.gv.logger.info('Averaging Staleness: {}'.format(np.mean([r-self.current_round for r in stale_rounds])))
            self.model = fmodule._model_average(self.buffer['model'])
            self.round_finished = True
            # clear buffer
            for k in self.buffer.keys(): self.buffer[k] = []
        return self.round_finished

    def aggregation_condition(self):
        if self.cond==0:
            for cid in self.selected_clients:
                if cid not in self.buffer['client_id']:
                    # aggregate only when receiving all the packages from selected clients
                    return False
            return True
        elif self.cond==1:
            # aggregate if the time budget for waiting is exhausted
            if self.gv.clock.current_time-self.sampling_timestamp>=self.time_budget or all([(cid in self.buffer['client_id']) for cid in self.selected_clients]):
                return True
            return False
        elif self.cond==2:
            # aggregate when the number of models in the buffer is larger than K
            return len(self.buffer['client_id'])>=self.K

class Client(BasicClient):
    def unpack(self, received_pkg):
        self.round = received_pkg['round']
        return received_pkg['model']

    def pack(self, model, *args, **kwargs):
        return {
            'model': model,
            'round': self.round
        }


if __name__ =='__main__':
    import flgo
    import flgo.benchmark.mnist_classification as mnist
    import os
    task = './my_task'
    if not os.path.exists(task):
        flgo.gen_task({'benchmark': mnist, 'partitioner': {'name': 'IIDPartitioner', 'para': {'num_clients': 20}}}, task)
    class algo:
        Server = Server
        Client = Client
    runner0 = flgo.init(task, algo, option={'num_rounds':10, 'algo_para':[0, 200, 0], "gpu": 0, 'proportion': 0.2, 'num_steps': 5, 'responsiveness': 'UNI-5-1000'})
    runner0.run()
    runner1 = flgo.init(task, algo, option={'num_rounds':10, 'algo_para':[1, 200, 0], "gpu": 0, 'proportion': 0.2, 'num_steps': 5, 'responsiveness': 'UNI-5-1000'})
    runner1.run()
    runner2 = flgo.init(task, algo, option={'num_rounds':10, 'algo_para':[2, 0, 10], "gpu": 0,  'proportion': 0.2, 'num_steps': 5, 'responsiveness': 'UNI-5-1000'})
    runner2.run()