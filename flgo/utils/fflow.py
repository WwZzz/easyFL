import numpy as np
import torch
import os.path
try:
    import ujson as json
except:
    import json
import flgo.system_simulator
import flgo.system_simulator.default_simulator
import flgo.system_simulator.base
import flgo.utils.fmodule
import flgo.experiment.logger.simple_logger
import flgo.algorithm
import flgo.tmp
import argparse
import importlib
import random
import os
import yaml
import tempfile

sample_list=['uniform', 'md', 'full', 'uniform_available', 'md_available', 'full_available']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam', 'RMSprop', 'Adagrad']
logger = None

class GlobalVariable:
    def __init__(self):
        self.logger = None
        self.simulator = None
        self.clock = None
        self.Model = None
        self.SvrModel = None
        self.CltModel = None
        self.dev_list = None
        self.dev_manager = None
        self.TaskCalculator = None
        self.TaskPipe = None

    def create_devmanager(self):
        if self.dev_list is None: return None
        crt_dev = 0
        while True:
            yield self.dev_list[crt_dev]
            crt_dev = (crt_dev + 1) % len(self.dev_list)

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def read_option():
    parser = argparse.ArgumentParser()
    """Training Options"""
    # basic settings
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
    # algorithm-dependent hyper-parameters
    parser.add_argument('--algo_para', help='algorithm-dependent hyper-parameters', nargs='*', type=float)

    """Environment Options"""
    # the ratio of the amount of the data used to train
    parser.add_argument('--train_holdout', help='the rate of holding out the validation dataset from all the local training datasets', type=float, default=0.1)
    parser.add_argument('--test_holdout', help='the rate of holding out the validation dataset from the training datasets', type=float, default=0.0)
    # realistic machine config
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int)
    parser.add_argument('--server_with_cpu', help='seed for random initialization;', action="store_true", default=False)
    parser.add_argument('--num_threads', help="the number of threads in the clients computing session", type=int, default=1)
    parser.add_argument('--num_workers', help='the number of workers of DataLoader', type=int, default=0)
    parser.add_argument('--test_batch_size', help='the batch_size used in testing phase;', type=int, default=512)

    """Simulator Options"""
    # the simulating systemic configuration of clients and the server that helps constructing the heterogeity in the network condition & computing power
    parser.add_argument('--simulator', help='name of system simulator', type=str, default='default_simulator')
    parser.add_argument('--availability', help="client availability mode", type=str, default = 'IDL')
    parser.add_argument('--connectivity', help="client connectivity mode", type=str, default = 'IDL')
    parser.add_argument('--completeness', help="client completeness mode", type=str, default = 'IDL')
    parser.add_argument('--responsiveness', help="client responsiveness mode", type=str, default='IDL')

    """Logger Options"""
    # logger setting
    parser.add_argument('--logger', help='the Logger in utils.logger.logger_name will be loaded', type=str, default='basic_logger')
    parser.add_argument('--log_level', help='the level of logger', type=str, default='INFO')
    parser.add_argument('--log_file', help='bool controls whether log to file and default value is False', action="store_true", default=False)
    parser.add_argument('--no_log_console', help='bool controls whether log to screen and default value is True', action="store_true", default=False)
    parser.add_argument('--no_overwrite', help='bool controls whether to overwrite the old result', action="store_true", default=False)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)

    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    for key in option.keys():
        if option[key] is None:
            option[key]=[]
    return option

def clear_temp():
    tmppath = '/'.join(flgo.tmp.__file__.split('/')[:-1])
    files = os.listdir(tmppath)
    for file in files:
        os.remove(os.path.join(tmppath, file))

def gen_task(config, task_path='', rawdata_path='', seed=0):
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # load configuration
    with open(config) as f:
        gen_option = yaml.load(f, Loader=yaml.FullLoader)
    if 'para' not in gen_option['benchmark'].keys(): gen_option['benchmark']['para'] = {}
    if 'partitioner' in gen_option.keys() and 'para' not in gen_option['partitioner'].keys(): gen_option['partitioner']['para'] = {}
    if 'partitioner' in gen_option.keys() and 'name' not in gen_option['partitioner'].keys() and 'para' in gen_option['partitioner'].keys():
        gen_option['benchmark']['para'].update(gen_option['partitioner']['para'])
    # init generator
    if rawdata_path!='': gen_option['benchmark']['para']['rawdata_path']=rawdata_path
    task_generator = getattr(importlib.import_module('.'.join(['flgo','benchmark', gen_option['benchmark']['name'], 'core'])), 'TaskGenerator')(**gen_option['benchmark']['para'])
    # create partitioner for generator if specified
    if 'partitioner' in gen_option.keys():
        partitioner = getattr(importlib.import_module('.'.join(['flgo','benchmark', 'toolkits', 'partition'])), gen_option['partitioner']['name'])(**gen_option['partitioner']['para'])
        task_generator.register_partitioner(partitioner)
        partitioner.register_generator(task_generator)
    else:
        partitioner = None
    # generate federated task
    task_generator.generate()
    # save the generated federated benchmark
    # initialize task pipe
    if task_path=='': task_path = os.path.join('.', task_generator.task_name)
    task_pipe = getattr(importlib.import_module('.'.join(['flgo','benchmark', gen_option['benchmark']['name'], 'core'])), 'TaskPipe')(task_path)
    # check if task already exists
    if task_pipe.task_exists():
        raise FileExistsError('Task {} already exists.'.format(task_path))
    try:
        # create task architecture
        task_pipe.create_task_architecture()
        # save meta infomation
        task_pipe.save_info(task_generator)
        # save task
        task_pipe.save_task(task_generator)
        print('Task {} has been successfully generated.'.format(task_generator.task_name))
    except Exception as e:
        print(e)
        task_pipe.remove_task()
        print("Failed to saving splited dataset.")
    # save visualization
    try:
        visualize_func = getattr(importlib.import_module('.'.join(['flgo','benchmark', gen_option['benchmark']['name']])),'visualize')
        visualize_func(task_generator, partitioner)
        task_pipe.save_figure()
    except:
        pass

def init(task, algorithm, option, model_name='', Logger=flgo.experiment.logger.simple_logger.Logger, simulator=flgo.system_simulator.default_simulator, scene='horizontal'):
    # init option
    default_option = read_option()
    for op_key in option:
        if op_key in default_option.keys():
            op_type = type(default_option[op_key])
            if op_type == type(option[op_key]):
                default_option[op_key] = option[op_key]
            else:
                if op_type is list:
                    default_option[op_key]=list(option[op_key]) if hasattr(option[op_key], '__iter__') else [option[op_key]]
                elif op_type is tuple:
                    default_option[op_key] = tuple(option[op_key]) if hasattr(option[op_key], '__iter__') else (option[op_key])
                else:
                    default_option[op_key] = op_type(option[op_key])
    option = default_option
    setup_seed(seed=option['seed'])
    option['task'] = task
    option['algorithm'] = (algorithm.__name__).split('.')[-1]

    # init task info
    if not os.path.exists(task):
        raise FileExistsError("Fedtask '{}' doesn't exist. Please generate the specified task by flgo.create_task().")
    with open(os.path.join(task, 'info'), 'r') as inf:
        task_info = json.load(inf)
    benchmark = task_info['benchmark']
    if model_name=='': model_name = getattr(importlib.import_module('.'.join(['flgo','benchmark',benchmark])), 'default_model')
    option['model'] = model_name

    try:
        # init multiprocess
        torch.multiprocessing.set_start_method('spawn')
        torch.multiprocessing.set_sharing_strategy('file_system')
    except:
        pass

    # create global variable
    gv = GlobalVariable()

    # init logger
    gv.logger = Logger(task=task, option=option, name=str(Logger), level=option['log_level'])

    # init device
    gv.dev_list = [torch.device('cpu')] if option['gpu'] is None else [torch.device('cuda:{}'.format(gpu_id)) for gpu_id in option['gpu']]
    gv.dev_manager = gv.create_devmanager()
    gv.logger.info('Initializing devices: '+','.join([str(dev) for dev in gv.dev_list])+' will be used for this running.')

    # init task
    core_module = '.'.join(['flgo','benchmark',benchmark, 'core'])
    gv.TaskPipe = getattr(importlib.import_module(core_module), 'TaskPipe')
    task_pipe = gv.TaskPipe(task)
    gv.TaskCalculator = getattr(importlib.import_module(core_module), 'TaskCalculator')
    task_data = task_pipe.load_data(option)

    # init model
    if model_name=='': model_name = getattr(importlib.import_module('.'.join(['flgo','benchmark',benchmark])), 'default_model')
    benchmark_model_module = '.'.join(['flgo','benchmark',benchmark, 'model', model_name])
    model_classes = { 'Model': None, 'SvrModel': None, 'CltModel': None}
    for model_class in model_classes:
        loading_priority = {
            benchmark_model_module: model_class,
            algorithm: model_name if model_class=='Model' else model_class,
        }
        for model_path, model_name in loading_priority.items():
            try:
                model_classes[model_class] = getattr(importlib.import_module(model_path), model_name)
                break
            except:
                continue
        if model_classes[model_class] is not None: gv.logger.info('Global model {} in {} was loaded.'.format(model_class, model_path))
        else: gv.logger.info('No {} is being used.'.format(model_class))
    gv.Model, gv.SvrModel, gv.CltModel = model_classes['Model'], model_classes['SvrModel'], model_classes['CltModel']

    # init objects
    obj_class = [c for c in dir(algorithm) if not c.startswith('__')]
    tmp = []
    for c in obj_class:
        try:
            C = getattr(algorithm, c)
            setattr(C, 'gv', gv)
            tmp.append(c)
        except:
            continue
    objects = task_pipe.generate_objects(option, algorithm, scene=scene)
    task_pipe.distribute(task_data, objects)
    for ob in objects: ob.initialize()
    for c in tmp:
        C = getattr(algorithm, c)
        delattr(C, 'gv')

    # init virtual system environment
    gv.logger.info('Use `{}` as the system simulator'.format(simulator))
    flgo.system_simulator.base.random_seed_gen = flgo.system_simulator.base.seed_generator(option['seed'])
    gv.clock = flgo.system_simulator.base.ElemClock()
    gv.state_updater = getattr(simulator, 'StateUpdater')(objects, option)
    gv.clock.register_state_updater(state_updater=gv.state_updater)

    gv.logger.register_variable(coordinator=objects[0], participants=objects[1:], option=option, clock=gv.clock)
    gv.logger.initialize()
    gv.logger.info('Ready to start.')

    # register global variables for objects
    for ob in objects:
        ob.gv = gv
    gv.state_updater.gv = gv
    gv.clock.gv = gv
    gv.logger.gv = gv
    return objects[0]
