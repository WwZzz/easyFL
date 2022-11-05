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
import utils.system_simulator as ss
import logging

sample_list=['uniform', 'md', 'full']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam', 'RMSprop', 'Adagrad']
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
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    parser.add_argument('--early_stop', help='stop training if there is no improvement for no smaller than the maximum rounds', type=int, default=-1)
    # hyper-parameters of local training
    parser.add_argument('--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument('--num_steps', help='the number of local steps, which dominate num_epochs when setting num_steps>0', type=int, default=-1)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch size when clients trainset on data;', type=float, default='64')
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)
    parser.add_argument('--weight_decay', help='weight decay for the training process', type=float, default=0)
    # realistic machine config
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int)
    parser.add_argument('--server_with_cpu', help='seed for random initialization;', action="store_true", default=False)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument('--cross_validation', help='shuffle each local train_data and valid_data', action="store_true", default=False)
    parser.add_argument('--train_on_all', help='use both train_data and valid_data to train the model;', action="store_true", default=False)
    parser.add_argument('--num_threads', help="the number of threads in the clients computing session", type=int, default=1)
    parser.add_argument('--num_workers', help='the number of workers of DataLoader', type=int, default=0)
    parser.add_argument('--test_batch_size', help='the batch_size used in testing phase;', type=int, default=512)
    # the simulating systemic configuration of clients and the server that helps constructing the heterogeity in the network condition & computing power
    parser.add_argument('--availability', help="client availability mode", type=str, default = 'IDL')
    parser.add_argument('--connectivity', help="client connectivity mode", type=str, default = 'IDL')
    parser.add_argument('--completeness', help="client completeness mode", type=str, default = 'IDL')
    parser.add_argument('--timeliness', help="client response timeliness mode", type=str, default='IDL')
    # algorithm-dependent hyper-parameters
    parser.add_argument('--algo_para', help='algorithm-dependent hyper-parameters', nargs='*', type=float)
    # logger setting
    parser.add_argument('--logger', help='the Logger in utils.logger.logger_name will be loaded', type=str, default='basic_logger')
    parser.add_argument('--log_level', help='the level of logger', type=str, default='INFO')
    parser.add_argument('--log_file', help='bool controls whether log to file and default value is False', action="store_true", default=False)
    parser.add_argument('--no_log_console', help='bool controls whether log to screen and default value is True', action="store_true", default=False)
    parser.add_argument('--no_overwrite', help='bool controls whether to overwrite the old result', action="store_true", default=False)

    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def initialize(option):
    # init logger from 1) Logger in algorithm/fedxxx.py, 2) Logger in utils/logger/logger_name.py 3) Logger in utils/logger/basic_logger.py
    logger_order = {'{}Logger'.format(option['algorithm']):'%s.%s' % ('algorithm', option['algorithm']),option['logger']:'.'.join(['utils', 'logger', option['logger']]),'basic_logger':'.'.join(['utils', 'logger', 'basic_logger'])}
    global logger
    for log_name, log_path in logger_order.items():
        try:
            Logger = getattr(importlib.import_module(log_path), 'Logger')
            break
        except:
            continue
    logger = Logger(meta=option, name=log_name, level=option['log_level'])
    logger.info('Using Logger in `{}`'.format(log_path))
    logger.info("Initializing fedtask: {}".format(option['task']))
    # benchmark information
    bmk_name = option['task'][:option['task'].find('cnum')-1]
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    # read federated task by TaskPipe
    # init partitioned dataset
    TaskPipe = getattr(importlib.import_module(bmk_core_path), 'TaskPipe')
    TaskPipe.set_option(option['cross_validation'], option['train_on_all'])
    train_datas, valid_datas, test_data, client_names = TaskPipe.load_task(os.path.join('fedtask', option['task']))
    # init model
    try:
        utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
        logger.info('Using model `{}` in `{}` as the globally shared model.'.format(option['model'], bmk_model_path))
    except ModuleNotFoundError:
        utils.fmodule.Model = getattr(importlib.import_module('.'.join(['algorithm', option['algorithm']])), option['model'])
        logger.info('Using model `{}` in `{}` as the globally shared model.'.format(option['model'],'.'.join(['algorithm', option['algorithm']])))
    # init the model that owned by the server (e.g. the model trained in the server-side)
    try:
        utils.fmodule.SvrModel = getattr(importlib.import_module(bmk_model_path), 'SvrModel')
        logger.info('The server keeps the `SvrModel` in `{}`'.format(bmk_model_path))
    except:
        try:
            utils.fmodule.SvrModel = getattr(importlib.import_module('.'.join(['algorithm', option['algorithm']])), 'SvrModel')
            logger.info('The server keeps the `SvrModel` in `{}`'.format('.'.join(['algorithm', option['algorithm']])))
        except:
            utils.fmodule.SvrModel = None
            logger.info('No server-specific model is used.')

    # init the model that owned by the client (e.g. the personalized model whose type may be different from the global model)
    try:
        utils.fmodule.CltModel = getattr(importlib.import_module(bmk_model_path), 'CltModel')
        logger.info('Clients keep the `CltModel` in `{}`'.format(bmk_model_path))
    except:
        try:
            utils.fmodule.CltModel = getattr(importlib.import_module('.'.join(['algorithm', option['algorithm']])), 'CltModel')
            logger.info('Clients keep the `CltModel` in `{}`'.format('.'.join(['algorithm', option['algorithm']])))
        except:
            utils.fmodule.CltModel = None
            logger.info('No client-specific model is used.')
    # init devices
    gpus = option['gpu']
    utils.fmodule.dev_list = [torch.device('cpu')] if gpus is None else [torch.device('cuda:{}'.format(gpu_id)) for gpu_id in gpus]
    utils.fmodule.dev_manager = utils.fmodule.get_device()
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    logger.info('Initializing devices: '+','.join([str(dev) for dev in utils.fmodule.dev_list])+' will be used for this running.')
    # The Model is defined in bmk_model_path as default, whose filename is option['model'] and the classname is 'Model'
    # If an algorithm change the backbone for a task, a modified model should be defined in the path 'algorithm/method_name.py', whose classname is option['model']
    if not option['server_with_cpu']:
        model = utils.fmodule.Model().to(utils.fmodule.dev_list[0])
    else:
        model = utils.fmodule.Model()
    # load pre-trained model
    try:
        if option['pretrain'] != '':
            model.load_state_dict(torch.load(option['pretrain'])['model'])
            logger.info('The pretrained model parameters in {} will be loaded'.format(option['pretrain']))
    except:
        logger.warn("Invalid Model Configuration.")
        exit(1)

    # init client
    num_clients = len(client_names)
    client_path = '%s.%s' % ('algorithm', option['algorithm'])
    logger.info('Initializing Clients: '+'{} clients of `{}` being created.'.format(num_clients, client_path+'.Client'))
    Client=getattr(importlib.import_module(client_path), 'Client')
    clients = [Client(option, name=client_names[cid], train_data=train_datas[cid], valid_data=valid_datas[cid]) for cid in range(num_clients)]
    for cid, c in enumerate(clients): c.id = cid
    # init server
    server_path = '%s.%s' % ('algorithm', option['algorithm'])
    logger.info('Initializing Server: '+'1 server of `{}` being created.'.format(server_path + '.Server'))
    server_module = importlib.import_module(server_path)
    server = getattr(server_module, 'Server')(option, model, clients, test_data = test_data)

    # init virtual systemic configuration including network state and the distribution of computing power
    logger.info('Initializing Systemic Heterogeneity: '+'Availability {}'.format(option['availability']))
    logger.info('Initializing Systemic Heterogeneity: '+'Connectivity {}'.format(option['connectivity']))
    logger.info('Initializing Systemic Heterogeneity: '+'Completeness {}'.format(option['completeness']))
    logger.info('Initializing Systemic Heterogeneity: '+'Timeliness {}'.format(option['timeliness']))

    ss.init_system_environment(server, option)
    logger.register_variable(server=server, clients=clients, meta=option, clock=ss.clock)
    logger.initialize()
    logger.info('Ready to start.')
    return server

def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para,pv in server.algo_para.items(): header = header + para + "{}_".format(pv)
    output_name = header + "M{}_R{}_B{}_".format(option['model'], option['num_rounds'],option['batch_size'])+ \
                  ("E{}_".format(server.clients[0].epochs)) + \
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