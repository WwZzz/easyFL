import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import json
import task.modelfuncs

sample_list=['uniform', 'prob']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam']

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--method', help='name of method;', type=str, default='fedavg')
    parser.add_argument('--dataset', help='name of dataset;', type=str, default='mnist')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')
    # methods of server side for sampling and aggregating
    parser.add_argument('--sample', help='methods for sampling clients', type=str, choices=sample_list, default='uniform')
    parser.add_argument('--aggregate', help='methods for aggregating models', type=str, choices=agg_list, default='uniform')
    # hyper-parameters of training in server side
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=10)
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.1)
    # hyper-parameters of training in client side
    parser.add_argument('--num_epochs', help='number of epochs when clients train on data;', type=int, default=1)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=10)
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)
    # controlling
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--gpu', help='GPU ID, -1 for CPU', type=int, default=1)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=-1)
    # hyper-parameters of different methods
    parser.add_argument('--learning_rate_lambda', help='η for λ in afl', type=float, default=0)
    parser.add_argument('--q', help='q in q-fedavg', type=float, default='0.0')
    parser.add_argument('--epsilon', help='ε in fedmgda+', type=float, default='0.0')
    parser.add_argument('--tau', help='the length of recent history gradients to be contained in FedFAvg', type=int, default=0)
    parser.add_argument('--alpha', help='proportion of clients keeping original direction in FedFV/alpha in fedFA', type=float, default='0.0')
    parser.add_argument('--beta', help='beta in FedFA',type=float, default='1.0')
    parser.add_argument('--gamma', help='gamma in FedFA', type=float, default='0')
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
    # init model
    print("init model...",end='')
    model_path = '%s.%s.%s' % ('task', option['dataset'], option['model'])
    task.modelfuncs.device = torch.device('cuda:{}'.format(option['gpu']) if torch.cuda.is_available() and option['gpu'] != -1 else 'cpu')
    task.modelfuncs.lossfunc = getattr(importlib.import_module(model_path), 'Loss')()
    task.modelfuncs.optim = getattr(importlib.import_module('torch.optim'), option['optimizer'])
    model = getattr(importlib.import_module(model_path), 'Model')().to(task.modelfuncs.device)
    print('ok')
    #init dataset
    print("init dataset...", end='')
    train_path = os.path.join('task', option['dataset'], 'data', 'train')
    test_path = os.path.join('task', option['dataset'], 'data', 'test')
    client_names = []
    train_data = {}
    test_data = {}
    train_files = os.listdir(train_path)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_path, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        client_names.extend(cdata['users'])
        train_data.update(cdata['user_data'])
    test_files = os.listdir(test_path)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_path, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    client_names = list(train_data.keys())
    print("ok")
    # init client
    print('init clients...', end='')
    clients=[]
    client_path = '%s.%s' % ('method', option['method'])
    Client=getattr(importlib.import_module(client_path), 'Client')
    for cname in client_names:
        clients.append(Client(option, cname, train_data[cname], test_data[cname]))
    print('ok')
    # init server
    print("init server...", end='')
    server_path = '%s.%s' % ('method', option['method'])
    server = getattr(importlib.import_module(server_path), 'Server')(option, model, clients)
    print('ok')
    return server

def output_filename(option, server):
    filename = "{}".format(option["method"])
    for para in server.paras_name:
        filename = filename + "_" + para + "{}".format(option[para])
    filename = filename + "_r{}_b{}_e{}_lr{}_p{}_seed{}.json".format(option['num_rounds'], option['batch_size'],
                                                                     option['num_epochs'],
                                                                     option['learning_rate'], option['proportion'],
                                                                     option['seed'])
    return filename




