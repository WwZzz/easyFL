import utils.fflow as flw
import numpy as np

class MyLogger(flw.Logger):
    def __init__(self):
        super(MyLogger, self).__init__()

    def log(self, server=None):
        if len(self.output) == 0:
            self.output['meta'] = server.option
        test_metric = server.test()
        valid_metrics = server.test_on_clients(self.current_round, 'valid')
        train_metrics = server.test_on_clients(self.current_round, 'train')
        for met_name, met_val in test_metric.items():
            self.output['test_'+met_name].append(met_val)
        # calculate weighted averaging of metrics of training datasets across clients
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(server.client_vols, met_val)]) / server.data_vol)
        # calculate weighted averaging and other statistics of metrics of validation datasets across clients
        for met_name, met_val in valid_metrics.items():
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(server.client_vols, met_val)]) / server.data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        for key, val in self.output.items():
            if key=='meta':continue
            print(self.temp.format(key, val[-1]))

logger = MyLogger()

def main():
    # read options
    option = flw.read_option()
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()




