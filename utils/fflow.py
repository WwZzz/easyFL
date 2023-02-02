import numpy as np
import argparse
import random
import paddle
import os.path
import importlib
import os

import system_simulator.base
import utils.fmodule
import config as cfg

sample_list=['uniform', 'md', 'full', 'uniform_available', 'md_available', 'full_available']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam', 'RMSprop', 'Adagrad']
logger = None

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    paddle.seed(12+seed)
    # torch.cuda.manual_seed_all(123+seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;', type=str, default='B-mnist_classification_P-IID_N-100_S-0')
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
    parser.add_argument('--train_holdout', help='the rate of holding out the validation dataset from all the local training datasets', type=float, default=0.1)
    parser.add_argument('--test_holdout', help='the rate of holding out the validation dataset from the training datasets', type=float, default=0.0)
    parser.add_argument('--num_threads', help="the number of threads in the clients computing session", type=int, default=1)
    parser.add_argument('--num_workers', help='the number of workers of DataLoader', type=int, default=0)
    parser.add_argument('--test_batch_size', help='the batch_size used in testing phase;', type=int, default=512)
    # the simulating systemic configuration of clients and the server that helps constructing the heterogeity in the network condition & computing power
    parser.add_argument('--simulator', help='name of system simulator', type=str, default='default_simulator')
    parser.add_argument('--availability', help="client availability mode", type=str, default = 'IDL')
    parser.add_argument('--connectivity', help="client connectivity mode", type=str, default = 'IDL')
    parser.add_argument('--completeness', help="client completeness mode", type=str, default = 'IDL')
    parser.add_argument('--responsiveness', help="client responsiveness mode", type=str, default='IDL')
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

def init_logger(option):
    # init logger from 1) Logger in algorithm/fedxxx.py, 2) Logger in utils/logger/logger_name.py 3) Logger in utils/logger/basic_logger.py
    loading_priority = {
        '{}Logger'.format(option['algorithm']): '%s.%s' % ('algorithm', option['algorithm']),
        option['logger']: '.'.join(['experiment', 'logger', option['logger']]),
        'BasicLogger': '.'.join(['experiment', 'logger', 'basic_logger'])
    }
    Logger = None
    for log_name, log_path in loading_priority.items():
        try:
            Logger = getattr(importlib.import_module(log_path), 'Logger')
            break
        except:
            continue
    logger = Logger(option=option, name=log_name, level =option['log_level'])
    logger.info('Using {} in `{}`'.format(log_name, log_path))
    return logger

def init_taskcore(option):
    cfg.logger.info("Initializing task core of  {}".format(option['task']))
    bmk_name = option['task'][2:option['task'].find('_P-')]
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    pipe = getattr(importlib.import_module(bmk_core_path), 'TaskPipe')(option['task'])
    class_calculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    return pipe, class_calculator

def init_device(option):
    dev_list = [paddle.device.set_device('cpu')] if option['gpu'] is None else [paddle.device.set_device('gpu:{}'.format(gpu_id)) for gpu_id in option['gpu']]
    dev_manager = utils.fmodule.get_device()
    cfg.logger.info('Initializing devices: '+','.join([str(dev) for dev in dev_list])+' will be used for this running.')
    return dev_list, dev_manager

def init_model(option):
    bmk_name = option['task'][2:option['task'].find('_P-')]
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    model_classes = { 'Model': None, 'SvrModel': None, 'CltModel': None}
    for model_class in model_classes:
        loading_priority = {
            bmk_model_path: model_class,
            '.'.join(['algorithm', option['algorithm']]): option['model'] if model_class=='Model' else model_class,
        }
        for model_path, model_name in loading_priority.items():
            try:
                model_classes[model_class] = getattr(importlib.import_module(model_path), model_name)
                break
            except:
                continue
        if model_classes[model_class] is not None: cfg.logger.info('Global model {} in {} was loaded.'.format(model_class, model_path))
        else: cfg.logger.info('No {} is being used.'.format(model_class))
    return model_classes['Model'], model_classes['SvrModel'], model_classes['CltModel']

def init_system_environment(objects, option):
    # init virtual systemic configuration including network state and the distribution of computing power
    cfg.logger.info('Use `{}` as the system simulator'.format(option['simulator']))
    system_simulator.base.random_seed_gen = system_simulator.base.seed_generator(option['seed'])
    cfg.clock = system_simulator.base.ElemClock()
    simulator = getattr(importlib.import_module('.'.join(['system_simulator', option['simulator']])), 'StateUpdater')(objects, option)
    cfg.state_updater = simulator
    cfg.clock.register_state_updater(simulator)

def initialize(option):
    # init logger
    cfg.logger = init_logger(option)
    # init task pipe and task
    task_pipe, cfg.TaskCalculator = init_taskcore(option)
    # init devices
    cfg.dev_list, cfg.dev_manager = init_device(option)
    # init model
    cfg.Model, cfg.SvrModel, cfg.CltModel = init_model(option)
    # load federated task through pipe and init objects (i.e. server and clients)
    objects = task_pipe.load_task(option)
    for ob in objects: ob.initialize()
    # init virtual system environment
    init_system_environment(objects, option)
    # finally prepare for logger
    cfg.logger.register_variable(coordinator=objects[0], participants=objects[1:], option=option, clock=cfg.clock)
    cfg.logger.initialize()
    cfg.logger.info('Ready to start.')
    return objects[0]