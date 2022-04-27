from .fedbase import BasicServer, BasicClient
import utils.fflow as flw
import os
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

    def run(self):
        flw.logger.time_start('Total Time Cost')
        selected_clients = [_ for _ in range(self.num_clients)]
        models = self.communicate(selected_clients)['model']
        flw.logger.time_end('Total Time Cost')
        flw.logger.log(self, models)
        flw.logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        return

    def unpack(self, packages_received_from_clients):
        return [packages_received_from_clients[cid]['model'] for cid in range(self.num_clients)]

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

class MyLogger(flw.Logger):
    def log(self, server=None, models=[]):
        if server == None or models == []: return
        if len(self.output) == 0:
            self.output['meta'] = server.option
        for cid in range(server.num_clients):
            test_metric = server.test(models[cid])
            valid_metrics = server.clients[cid].test(models[cid], 'valid')
            self.output[server.clients[cid].name] = {}
            for met_name, met_val in test_metric.items():
                self.output['test_' + met_name].append(met_val)
            # calculate weighted averaging and other statistics of metrics of validation datasets across clients
            for met_name, met_val in valid_metrics.items():
                self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in
                                                                   zip(server.client_vols, met_val)]) / server.data_vol)
                self.output['mean_valid_' + met_name].append(np.mean(met_val))
                self.output['std_valid_' + met_name].append(np.std(met_val))
                self.output['local_valid_'+met_name].append(met_val[cid])