import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import ujson
import utils.fmodule

sample_list=['uniform', 'md']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam']

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;', type=str, default='mnist_client100_dist0_beta0_noise0')
    parser.add_argument('--method', help='name of method;', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')
    # methods of server side for sampling and aggregating
    parser.add_argument('--sample', help='methods for sampling clients', type=str, choices=sample_list, default='md')
    parser.add_argument('--aggregate', help='methods for aggregating models', type=str, choices=agg_list, default='uniform')
    parser.add_argument('--learning_rate_decay', help='learning rate decay for the training process;', type=float, default=0.998)
    parser.add_argument('--weight_decay', help='weight decay for the training process', type=float, default=0)
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    # hyper-parameters of training in server side
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=20)
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.2)
    # hyper-parameters of local training
    parser.add_argument('--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.2)
    parser.add_argument('--batch_size', help='batch size when clients trainset on data;', type=int, default=10)
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)
    # controlling
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--gpu', help='GPU ID, -1 for CPU', type=int, default=-1)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument('--num_threads', help="the number of threads in the clients computing session", type=int, default=1)
    parser.add_argument('--train_rate', help="the validtion dataset rate of each client's dataet", type=float, default=1)
    parser.add_argument('--drop', help="controlling the dropout of clients after being selected in each communication round according to distribution Beta(drop,1)", type=float, default=0)
    # hyper-parameters of different methods
    parser.add_argument('--learning_rate_lambda', help='η for λ in afl', type=float, default=0)
    parser.add_argument('--q', help='q in q-fedavg', type=float, default='0.0')
    parser.add_argument('--epsilon', help='ε in fedmgda+', type=float, default='0.0')
    parser.add_argument('--eta', help='global learning rate in fedmgda+', type=float, default='1.0')
    parser.add_argument('--tau', help='the length of recent history gradients to be contained in FedFAvg', type=int, default=0)
    parser.add_argument('--alpha', help='proportion of clients keeping original direction in FedFV/alpha in fedFA', type=float, default='0.0')
    parser.add_argument('--beta', help='beta in FedFA',type=float, default='1.0')
    parser.add_argument('--gamma', help='gamma in FedFA', type=float, default='0')
    parser.add_argument('--mu', help='mu in fedprox', type=float, default='0.1')
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
    bmk = option['task'][:option['task'].find('_')]
    model_path = '%s.%s.%s.%s' % ('benchmark', bmk, 'model', option['model'])
    utils.fmodule.device = torch.device('cuda:{}'.format(option['gpu']) if torch.cuda.is_available() and option['gpu'] != -1 else 'cpu')
    utils.fmodule.lossfunc = getattr(importlib.import_module(model_path), 'Loss')()
    utils.fmodule.Optim = getattr(importlib.import_module('torch.optim'), option['optimizer'])
    utils.fmodule.Model = getattr(importlib.import_module(model_path), 'Model')
    task_path = os.path.join('fedtask', option['task'], 'task.json')
    try:
        with open(task_path, 'r') as taskfile:
            task = ujson.load(taskfile)
    except FileNotFoundError:
        print("Task {} not found.".format(option['task']))
        exit()

    meta = task['meta']
    client_names = [name for name in task['clients'].keys()]
    train_data = [task['clients'][key]['dtrain'] for key in task['clients'].keys()]
    valid_data = [task['clients'][key]['dvalid'] for key in task['clients'].keys()]
    test_data = task['dtest']
    print("done")

    # init client
    print('init clients...', end='')
    client_path = '%s.%s' % ('method', option['method'])
    Client=getattr(importlib.import_module(client_path), 'Client')
    # the probability of dropout obey distribution beta(drop, 1). The larger 'drop' is, the more possible for a device to drop
    client_drop_rates = np.random.beta(option['drop']+0.00001,1,meta['num_clients'])
    clients = [Client(option, name = client_names[cid], data_train_dict = train_data[cid], data_val_dict = valid_data[cid], train_rate = option['train_rate'], drop_rate = client_drop_rates[cid]) for cid in range(meta['num_clients'])]
    print('done')

    # init server
    print("init server...", end='')
    server_path = '%s.%s' % ('method', option['method'])
    server = getattr(importlib.import_module(server_path), 'Server')(option, utils.fmodule.Model().to(utils.fmodule.device), clients, dtest = test_data)
    print('done')
    return server

def output_filename(option, server):
    header = "{}_".format(option["method"])
    for para in server.paras_name: header = header + para + "{}_".format(option[para])
    output_name = header + "M{}_R{}_B{}_E{}_LR{:.4f}_P{:.2f}_S{}_T{:.2f}_LD{:.3f}_WD{:.3f}_DR{:.2f}_.json".format(
        option['model'],
        option['num_rounds'],
        option['batch_size'],
        option['num_epochs'],
        option['learning_rate'],
        option['proportion'],
        option['seed'],
        option['train_rate'],
        option['lr_scheduler']+option['learning_rate_decay'],
        option['weight_decay'],
        option['drop'])
    return output_name




