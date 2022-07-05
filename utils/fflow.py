import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import utils.fmodule
import ujson
import time
import collections
import utils.systemic_simulator as ss

sample_list=['uniform', 'md']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam']
logger = None

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;', type=str, default='mnist_classification_cnum100_dist0_skew0_seed0')
    parser.add_argument('--algorithm', help='name of algorithm;', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')
    parser.add_argument('--pretrain', help='the path of the pretrained model parameter created by torch.save;', type=str, default='')
    # methods of server side for sampling and aggregating
    parser.add_argument('--sample', help='methods for sampling clients', type=str, choices=sample_list, default='md')
    parser.add_argument('--aggregate', help='methods for aggregating models', type=str, choices=agg_list, default='uniform')
    # hyper-parameters of training in server side
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=20)
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.2)
    parser.add_argument('--learning_rate_decay', help='learning rate decay for the training process;', type=float, default=0.998)
    parser.add_argument('--weight_decay', help='weight decay for the training process', type=float, default=0)
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    # hyper-parameters of local training
    parser.add_argument('--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument('--num_steps', help='the number of local steps, which dominate num_epochs when setting num_steps>0', type=int, default=-1)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch size when clients trainset on data;', type=float, default='64')
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)
    # realistic machine config
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int)
    parser.add_argument('--server_with_cpu', help='seed for random initialization;', action="store_true", default=False)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument('--num_threads', help="the number of threads in the clients computing session", type=int, default=1)
    parser.add_argument('--num_workers', help='the number of workers of DataLoader', type=int, default=0)
    parser.add_argument('--test_batch_size', help='the batch_size used in testing phase;', type=int, default=512)
    parser.add_argument('--project', help='name of this project for wandb;', type=str, default='unknown_project')
    # the simulating systemic configuration of clients that helps constructing the heterogeity in the network condition & computing power
    parser.add_argument('--network_config', help="configuration of the availability of clients", type=str, default = 'ideal')
    parser.add_argument('--computing_config', help="configuration of the computing power of clients", type=str, default = 'ideal')
    # algorithm-dependent hyper-parameters
    parser.add_argument('--algo_para', help='algorithm-dependent hyper-parameters', nargs='*', type=float)

    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)

def initialize(option):
    # init fedtask
    print("init fedtask...", end='')
    # dynamical initializing the configuration with the benchmark
    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    # init gpu
    gpus = option['gpu']
    utils.fmodule.dev_list = [torch.device('cpu')] if len(gpus)==0 else [torch.device('cuda:{}'.format(gpu_id)) for gpu_id in gpus]
    utils.fmodule.dev_manager = utils.fmodule.get_device()
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    # The Model is defined in bmk_model_path as default, whose filename is option['model'] and the classname is 'Model'
    # If an algorithm change the backbone for a task, a modified model should be defined in the path 'algorithm/method_name.py', whose classname is option['model']
    try:
        utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    except ModuleNotFoundError:
        utils.fmodule.Model = getattr(importlib.import_module('.'.join(['algorithm', option['algorithm']])), option['model'])
    if not option['server_with_cpu']:
        model = utils.fmodule.Model().to(utils.fmodule.dev_list[0])
    else:
        model = utils.fmodule.Model()
    # init the model that owned by the server (e.g. the model trained in the server-side)
    try:
        utils.fmodule.SvrModel = getattr(importlib.import_module(bmk_model_path), 'SvrModel')
    except:
        utils.fmodule.SvrModel = utils.fmodule.Model
    # init the model that owned by the client (e.g. the personalized model whose type may be different from the global model)
    try:
        utils.fmodule.CltModel = getattr(importlib.import_module(bmk_model_path), 'CltModel')
    except:
        utils.fmodule.CltModel = utils.fmodule.Model
    # load pre-trained model
    try:
        if option['pretrain'] != '':
            model.load_state_dict(torch.load(option['pretrain'])['model'])
    except:
        print("Invalid Model Configuration.")
        exit(1)
    # read federated task by TaskPipe
    TaskPipe = getattr(importlib.import_module(bmk_core_path), 'TaskPipe')
    train_datas, valid_datas, test_data, client_names = TaskPipe.load_task(os.path.join('fedtask', option['task']))
    num_clients = len(client_names)
    print("done")

    # init client
    print('init clients...', end='')
    client_path = '%s.%s' % ('algorithm', option['algorithm'])
    Client=getattr(importlib.import_module(client_path), 'Client')
    clients = [Client(option, name=client_names[cid], train_data=train_datas[cid], valid_data=valid_datas[cid]) for cid in range(num_clients)]
    print('done')

    # init server
    print("init server...", end='')
    server_path = '%s.%s' % ('algorithm', option['algorithm'])
    server = getattr(importlib.import_module(server_path), 'Server')(option, model, clients, test_data = test_data)
    # init virtual systemic configuration including network state and the distribution of computing power
    ss.init_systemic_config(server, option)
    # init logger
    try:
        Logger = getattr(importlib.import_module(server_path), 'MyLogger')
    except AttributeError:
        Logger = DefaultLogger
    global logger
    logger = Logger()
    print('done')
    return server

def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para,pv in server.algo_para.items(): header = header + para + "{}_".format(pv)
    output_name = header + "M{}_R{}_B{}_".format(option['model'], option['num_rounds'],option['batch_size'])+ \
                  ("E{}".format(option['num_epochs']) if option['num_steps']<0 else "K{}".format(option['num_steps']))+\
                  "LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_NET{}_CMP{}_.json".format(
                      option['learning_rate'],
                      option['proportion'],
                      option['seed'],
                      option['lr_scheduler']+option['learning_rate_decay'],
                      option['weight_decay'],
                      option['network_config'],
                      option['computing_config']
                  )
    return output_name

class Logger:
    def __init__(self):
        self.output = collections.defaultdict(list)
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf={}

    def check_if_log(self, round, eval_interval=-1):
        """For evaluating every 'eval_interval' rounds, check whether to log at 'round'."""
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key = ''):
        """Create a timestamp of the event 'key' starting"""
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key = ''):
        """Create a timestamp that ends the event 'key' and print the time interval of the event."""
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] =  time.time() - self.time_buf[key][-1]
            print("{:<30s}{:.4f}".format(key+":", self.time_buf[key][-1]) + 's')

    def save(self, filepath):
        """Save the self.output as .json file"""
        if len(self.output)==0: return
        with open(filepath, 'w') as outf:
            ujson.dump(dict(self.output), outf)
            
    def write(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name==None: raise RuntimeError("Missing the name of the variable to be logged.")
        self.output[var_name].append(var_value)
        return

    def log(self, server=None):
        pass

class DefaultLogger(Logger):
    def __init__(self):
        super(DefaultLogger, self).__init__()

    def log(self, server=None, current_round=-1):
        if len(self.output) == 0:
            self.output['meta'] = server.option
        test_metric = server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        # calculate weighted averaging of metrics of training datasets across clients
        train_metrics = server.test_on_clients(self.current_round, 'train')
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(server.local_data_vols, met_val)]) / server.total_data_vol)
        # calculate weighted averaging and other statistics of metrics of validation datasets across clients
        valid_metrics = server.test_on_clients(self.current_round, 'valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(server.local_data_vols, met_val)]) / server.total_data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        for key, val in self.output.items():
            if key == 'meta': continue
            print(self.temp.format(key, val[-1]))
