import logging
import collections
import time
import numpy as np
import os
try:
    import ujson as json
except:
    import json

class Logger(logging.Logger):

    _LEVEL = {
        "DEBUG": logging.DEBUG,

        "INFO": logging.INFO,

        "WARNING": logging.WARNING,

        "ERROR": logging.ERROR,

        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, meta, *args, **kwargs):
        self.meta = meta
        super(Logger, self).__init__(*args, **kwargs)
        self.output = collections.defaultdict(list)
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf = {}
        self.formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')
        self.handler_list = []
        self.overwrite = not self.meta['no_overwrite']
        if not self.meta['no_log_console']:
            self.streamhandler = logging.StreamHandler()
            self.streamhandler.setFormatter(self.formatter)
            self.streamhandler.setLevel(self._LEVEL[self.meta['log_level'].upper()])
            self.addHandler(self.streamhandler)
        if self.meta['log_file']:
            log_dir = self.get_log_path()
            self.log_path = os.path.join(log_dir, self.get_time_string()+self.get_output_name('.log'))
            if not os.path.exists(self.get_log_path()):
                os.mkdir(log_dir)
            self.filehandler = logging.FileHandler(self.log_path)
            self.filehandler.setFormatter(self.formatter)
            self.filehandler.setLevel(self._LEVEL[self.meta['log_level'].upper()])
            self.addHandler(self.filehandler)


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

    def save_output_as_json(self, filepath=None):
        """Save the self.output as .json file"""
        if len(self.output) == 0: return
        self.organize_output()
        self.output_to_jsonable_dict()
        if filepath is None:
            filepath = os.path.join(self.get_output_path(),self.get_output_name())
        if not self.overwrite:
            if os.path.exists(filepath):
                with open(filepath, 'r') as inf:
                    original_record = json.loads(inf.read())
                o_keys = set(original_record.keys())
                output_keys = set(self.output.keys())
                new_keys = list(output_keys.difference(o_keys))
                for k in new_keys:
                    original_record[k] = self.output[k]
                self.output = original_record
        try:
            with open(filepath, 'w') as outf:
                json.dump(dict(self.output), outf)
        except:
            self.error('Failed to save flw.logger.output as results')

    def check_is_jsonable(self, x):
        try:
            json.dumps(x)
            return True
        except:
            return False

    def output_to_jsonable_dict(self):
        for key, value in self.output.items():
            if not self.check_is_jsonable(value):
                try:
                    self.output[key] = str(self.output[key])
                    self.warning("flw.logger.output['{}'] is not jsonable, and is automatically converted to string.".format(key))
                except:
                    del self.output[key]
                    self.warning("Automatically remove flw.logger.output['{}'] from logger, because it is not jsonable and is failed to convert into string. ".format(key))
        return

    def write_var_into_output(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name == None: raise RuntimeError("Missing the name of the variable to be logged.")
        self.output[var_name].append(var_value)
        return

    def register_variable(self, **kwargs):
        """Initialze the logger in utils.fflow.initialize()"""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return

    def show_current_output(self, yes_key=['train', 'test', 'valid'], no_key=['dist']):
        for key, val in self.output.items():
            a = [(yk in key) for yk in yes_key]
            nf = [(nk not in key) for nk in no_key]
            if np.all(nf) and np.any(a):
                self.info(self.temp.format(key, val[-1]))

    def get_output_name(self, suffix='.json'):
        if not hasattr(self, 'meta'): raise NotImplementedError('logger has no attr named "meta"')
        header = "{}_".format(self.meta["algorithm"])
        if hasattr(self, 'server'):
            for para, pv in self.server.algo_para.items():
                header = header + para + "{}_".format(pv)
        else:
            if self.meta['algo_para'] is not None:
                header = header + 'algopara_'+'|'.join([str(p) for p in self.meta['algo_para']])

        output_name = header + "M{}_R{}_B{}_".format(self.meta['model'], self.meta['num_rounds'], self.meta['batch_size'])
        if self.meta['num_steps']<0:
            output_name = output_name + ("E{}_".format(self.meta['num_epochs']))
        else:
            output_name = output_name + ("K{}_".format(self.meta['num_steps']))

        output_name = output_name + "LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_NET{}_CMP{}".format(
                          self.meta['learning_rate'],
                          self.meta['proportion'],
                          self.meta['seed'],
                          self.meta['lr_scheduler'] + self.meta['learning_rate_decay'],
                          self.meta['weight_decay'],
                          self.meta['network_config'],
                          self.meta['computing_config']
                      ) + suffix
        return output_name

    def get_output_path(self):
        if not hasattr(self, 'meta'): raise NotImplementedError('logger has no attr named "meta"')
        return os.path.join('fedtask', self.meta['task'], 'record')

    def get_log_path(self):
        if not hasattr(self, 'meta'): raise NotImplementedError('logger has no attr named "meta"')
        return os.path.join('fedtask', self.meta['task'], 'log')

    def get_time_string(self):
        return time.strftime('%Y-%m-%d-%H-%M-%S')

    def initialize(self, *args, **kwargs):
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