from .fedbase import BasicServer
from .fedavg import Client
import utils.fflow as flw
import utils.logger.basic_logger as bl
import os
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def run(self):
        flw.logger.time_start('Total Time Cost')
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        flw.logger.time_end('Total Time Cost')
        flw.logger.log_per_round(models)
        flw.logger.save_output_as_json()
        return

class Logger(bl.Logger):
    def log_per_round(self, models=[]):
        if models == []: return
        for id, cid in enumerate(self.server.selected_clients):
            test_metric = self.server.test(models[id])
            cname = self.clients[cid].name
            for met_name, met_val in test_metric.items():
                self.output['test_' + met_name + '_' + cname].append(met_val)