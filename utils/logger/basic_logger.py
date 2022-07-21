import logging
import collections
import ujson
import time
import numpy as np

class Logger(logging.Logger):
    def __init__(self, *args, **kwargs):
        super(Logger, self).__init__(*args, **kwargs)
        self.output = collections.defaultdict(list)
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf = {}
        self.formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
        self.head = logging.StreamHandler()
        self.head.setLevel(logging.INFO)
        self.head.setFormatter(self.formatter)
        self.addHandler(self.head)

    def check_if_log(self, round, eval_interval=-1):
        """For evaluating every 'eval_interval' rounds, check whether to log at 'round'."""
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key=''):
        """Create a timestamp of the event 'key' starting"""
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key=''):
        """Create a timestamp that ends the event 'key' and print the time interval of the event."""
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] = time.time() - self.time_buf[key][-1]
            self.info("{:<30s}{:.4f}".format(key + ":", self.time_buf[key][-1]) + 's')

    def save_output_as_json(self, filepath):
        """Save the self.output as .json file"""
        if len(self.output) == 0: return
        self.organize_output()
        with open(filepath, 'w') as outf:
            ujson.dump(dict(self.output), outf)

    def write_var_into_output(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name == None: raise RuntimeError("Missing the name of the variable to be logged.")
        self.output[var_name].append(var_value)
        return

    def initialize(self, *args, **kwargs):
        return

    def register_variable(self, **kwargs):
        """Initialze the logger in utils.fflow.initialize()"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return

    def log_per_round(self, *args, **kwargs):
        """This method is called at the beginning of each communication round of Server.
        The round-wise operations of recording should be complemented here."""
        # calculate the testing metrics on testing dataset owned by server
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across clients
        train_metrics = self.server.test_on_clients('train')
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name + '_dist'].append(met_val)
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
        # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        valid_metrics = self.server.test_on_clients('valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_'+met_name+'_dist'].append(met_val)
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()

    def organize_output(self, *args, **kwargs):
        """This method will be called before saving self.output"""
        self.output['meta'] = self.meta
        for key in self.output.keys():
            if '_dist' in key:
                self.output[key] = self.output[key][-1]
        return

    def show_current_output(self, yes_key=['train', 'test', 'valid'], no_key=['dist']):
        for key, val in self.output.items():
            a = [(yk in key) for yk in yes_key]
            nf = [(nk not in key) for nk in no_key]
            a.extend(nf)
            if not np.any(a):
                self.info(self.temp.format(key, val[-1]))