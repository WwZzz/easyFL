import utils.fflow as flw
import ujson
import numpy as np
import time
import os

logger = None

class Logger:
    def __init__(self):
        self.output = {}
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.start_timestamp = None
        self.time_costs = []

    def check_if_log(self, round, eval_interval=-1):
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def start_timer(self):
        self.start_timestamp = time.time()

    def end_timer(self):
        if self.start_timestamp == None:
            self.end_timestamp = None
            return None
        else:
            end = time.time()
            self.time_costs.append(end - self.start_timestamp)
            self.start_timestamp = None
            return self.time_costs[-1]

    def save(self, filepath):
        with open(filepath, 'w') as outf:
            ujson.dump(self.output, outf)

    def log(self, server):
        if self.output == {}:
            self.output = {
                "meta":server.option,
                "acc_dist":[],
                "mean_curve":[],
                "var_curve":[],
                "train_losses":[],
                "test_accs":[],
                "test_losses":[],
                "valid_accs":[],
                "client_accs":{},
            }
        test_metric, test_loss = server.test()
        valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
        train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')
        self.output['train_losses'].append(1.0*sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)])/server.data_vol)
        self.output['valid_accs'].append(valid_metrics)
        self.output['test_accs'].append(test_metric)
        self.output['test_losses'].append(test_loss)
        self.output['mean_valid_accs'].append(1.0*sum([ck * acc for ck, acc in zip(server.client_vols, valid_metrics)])/server.data_vol)
        self.output['mean_curve'].append(np.mean(valid_metrics))
        self.output['var_curve'].append(np.std(valid_metrics))
        for cid in range(server.num_clients):
            self.output['client_accs'][server.clients[cid].name]=[valid_metrics[i][cid] for i in range(len(self.output['valid_accs']))]
        print(self.temp.format("Training Loss:", self.output['train_losses'][-1]))
        print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
        print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
        print(self.temp.format("Validating Accuracy:", self.output['mean_valid_accs'][-1]))
        print(self.temp.format("Mean of Client Accuracy:", self.output['mean_curve'][-1]))
        print(self.temp.format("Std of Client Accuracy:", self.output['var_curve'][-1]))

def main():
    # read options
    option = flw.read_option()
    # set random seed
    flw.setup_seed(option['seed'])
    # init logger
    logger = Logger()
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()
    # save results as .json file
    logger.save(os.path.join('fedtask', option['task'], 'record', flw.output_filename(option, server)))

if __name__ == '__main__':
    main()



