import collections
import shutil
import sys
import copy
import multiprocessing
import time
import itertools
import argparse
import importlib
import random
import os
import os.path
import warnings
from typing import *
try:
    import ujson as json
except:
    import json
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable

import numpy as np
import torch
try:
    import yaml
except ModuleNotFoundError:
    warnings.warn("Module pyyaml is not installed. The configuration cannot be loaded by .yml file.")

import flgo.simulator
import flgo.simulator.default_simulator
import flgo.simulator.base
import flgo.utils.fmodule
import flgo.experiment.logger.simple_logger
import flgo.experiment.logger.tune_logger
import flgo.experiment.logger.vertical_logger
import flgo.experiment.logger.dec_logger
import flgo.experiment.logger.hier_logger
import flgo.experiment.logger
import flgo.experiment.device_scheduler
from flgo.simulator.base import BasicSimulator
import flgo.benchmark.partition
import flgo.benchmark.toolkits.partition
import flgo.algorithm

sample_list=['uniform', 'md', 'full', 'uniform_available', 'md_available', 'full_available'] # sampling options for the default sampling method in flgo.algorihtm.fedbase
agg_list=['uniform', 'weighted_scale', 'weighted_com'] # aggregation options for the default aggregating method in flgo.algorihtm.fedbase
optimizer_list=['SGD', 'Adam', 'RMSprop', 'Adagrad'] # supported optimizers
default_option_dict = {'pretrain': '', 'sample': 'md', 'aggregate': 'uniform', 'num_rounds': 20, 'proportion': 0.2, 'learning_rate_decay': 0.998, 'lr_scheduler': -1, 'early_stop': -1, 'num_epochs': 5, 'num_steps': -1, 'learning_rate': 0.1, 'batch_size': 64.0, 'optimizer': 'SGD', 'clip_grad':0.0,'momentum': 0.0, 'weight_decay': 0.0, 'num_edge_rounds':5, 'algo_para': [], 'train_holdout': 0.1, 'test_holdout': 0.0, 'local_test':False,'seed': 0, 'gpu': [], 'server_with_cpu': False, 'num_parallels': 1, 'num_workers': 0, 'pin_memory':False,'test_batch_size': 512,'pin_memory':False ,'simulator': 'default_simulator', 'availability': 'IDL', 'connectivity': 'IDL', 'completeness': 'IDL', 'responsiveness': 'IDL', 'logger': 'basic_logger', 'log_level': 'INFO', 'log_file': False, 'no_log_console': False, 'no_overwrite': False, 'eval_interval': 1}

class GlobalVariable:
    """This class is to create a shared space for sharing variables across
    different parties for each runner"""

    def __init__(self):
        self.logger = None
        self.simulator = None
        self.clock = None
        self.dev_list = None
        self.TaskCalculator = None
        self.TaskPipe = None
        self.crt_dev = 0

    def apply_for_device(self):
        r"""
        Apply for a new device from currently available ones (i.e. devices in self.dev_list)

        Returns:
            GPU device (i.e. torch.device)
        """
        if self.dev_list is None: return None
        dev = self.dev_list[self.crt_dev]
        self.crt_dev = (self.crt_dev + 1) % len(self.dev_list)
        return dev

def setup_seed(seed):
    r"""
    Fix all the random seed used in numpy, torch and random module

    Args:
        seed (int): the random seed
    """
    if seed <0:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        seed = -seed
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)

def read_option_from_command():
    r"""
    Generate running-time configurations for flgo.init with default values from command lines

    Returns:
        a dict of option (i.e. configuration)
    """

    parser = argparse.ArgumentParser()
    """Training Options"""
    # basic settings
    # methods of server side for sampling and aggregating
    parser.add_argument('--sample', help='methods for sampling clients', type=str, choices=sample_list, default='uniform')
    parser.add_argument('--aggregate', help='methods for aggregating models', type=str, choices=agg_list, default='other')
    # hyper-parameters of training in server side
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=20)
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.2)
    parser.add_argument('--learning_rate_decay', help='learning rate decay for the training process;', type=float, default=0.998)
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    parser.add_argument('--early_stop', help='stop training if there is no improvement for no smaller than the maximum rounds', type=int, default=-1)
    # hyper-parameters of local_movielens_recommendation training
    parser.add_argument('--num_epochs', help='number of epochs when clients locally train the model on data;', type=int, default=5)
    parser.add_argument('--num_steps', help='the number of local steps, which dominate num_epochs when setting num_steps>0', type=int, default=-1)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch size', type=float, default='64')
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--clip_grad', help='clipping gradients if the max norm of gradients ||g|| > clip_norm > 0', type=float, default=0.0)
    parser.add_argument('--momentum', help='momentum of local training', type=float, default=0.0)
    parser.add_argument('--weight_decay', help='weight decay of local training', type=float, default=0.0)
    parser.add_argument('--num_edge_rounds', help='number of edge rounds in hierFL', type=int, default=5)
    # algorithm-dependent hyper-parameters
    parser.add_argument('--algo_para', help='algorithm-dependent hyper-parameters', nargs='*', type=float)

    """Environment Options"""
    # the ratio of the amount of the data used to train
    parser.add_argument('--train_holdout', help='the rate of holding out the validation dataset from all the local training datasets', type=float, default=0.1)
    parser.add_argument('--test_holdout', help='the rate of holding out the validation dataset from the testing datasets owned by the server', type=float, default=0.0)
    parser.add_argument('--local_test', help='if this term is set True and train_holdout>0, (0.5*train_holdout) of data will be set as client.test_data.', action="store_true", default=False)
    # realistic machine config
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--gpu', nargs='*', help='GPU IDs and empty input is equal to using CPU', type=int)
    parser.add_argument('--server_with_cpu', help='the model parameters will be stored in the memory if True', action="store_true", default=False)
    parser.add_argument('--num_parallels', help="the number of parallels in the clients computing session", type=int, default=1)
    parser.add_argument('--num_workers', help='the number of workers of DataLoader', type=int, default=0)
    parser.add_argument('--pin_memory', help='pin_memory of DataLoader', action="store_true", default=False)
    parser.add_argument('--test_batch_size', help='the batch_size used in testing phase;', type=int, default=512)

    """Simulator Options"""
    # the simulating systemic configuration of clients and the server that helps constructing the heterogeity in the network condition & computing power
    parser.add_argument('--availability', help="client availability mode", type=str, default = 'IDL')
    parser.add_argument('--connectivity', help="client connectivity mode", type=str, default = 'IDL')
    parser.add_argument('--completeness', help="client completeness mode", type=str, default = 'IDL')
    parser.add_argument('--responsiveness', help="client responsiveness mode", type=str, default='IDL')

    """Logger Options"""
    # logger setting
    parser.add_argument('--log_level', help='the level of logger', type=str, default='INFO')
    parser.add_argument('--log_file', help='bool controls whether log to file and default value is False', action="store_true", default=False)
    parser.add_argument('--no_log_console', help='bool controls whether log to screen and default value is True', action="store_true", default=False)
    parser.add_argument('--no_overwrite', help='bool controls whether to overwrite the old result', action="store_true", default=False)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    try: option = vars(parser.parse_known_args()[0])
    except IOError as msg: parser.error(str(msg))
    for key in option.keys():
        if option[key] is None:
            option[key]=[]
    return option

def load_configuration(config={}):
    r"""
    Load configurations from .yml file or dict.

    Args:
        config (dict|str): the configurations

    Returns:
        a dict of option (i.e. configuration)
    """
    if type(config) is str and config.endswith('.yml'):
        with open(config) as f:
            option = yaml.load(f, Loader=yaml.FullLoader)
        return option
    elif type(config) is dict:
        return config
    else:
        raise TypeError('The input config should be either a dict or a filename.')

def gen_benchmark_from_file(benchmark:str, config_file:str, target_path='.',data_type:str='cv', task_type:str='classification'):
    r"""
        Create customized benchmarks from configurations. The configuration is a .py file that describes the datasets and the model,
        where there must exist a function named `get_model` and a variable `train_data`. `val_data` and test_data are two optional
        variables in the configuration.
    Args:
        benchmark (str): the name of the benchmark
        config_file (str): the path of the configuration file
        target_path: (str): the path to store the benchmark
        data_type (str): the type of dataset that should be in the list ['cv', 'nlp', 'graph', 'rec', 'series', 'tabular']
        task_type (str): the type of the task (e.g. classification, regression...)
    Returns:
        bmk_module (str): the module name of the generated benchmark
    """
    if not os.path.exists(config_file): raise FileNotFoundError('File {} not found.'.format(config_file))
    target_path = os.path.abspath(target_path)
    bmk_path = os.path.join(target_path, benchmark)
    if os.path.exists(bmk_path): raise FileExistsError('Benchmark {} already exists'.format(bmk_path))
    temp_path = os.path.join(flgo.benchmark.path, 'toolkits', data_type, task_type, 'temp')
    if not os.path.exists(temp_path):
        raise NotImplementedError('There is no support to automatically generation of {}.{}. More other types are comming soon...'.format(data_type, task_type))
    else:
        shutil.copytree(temp_path, bmk_path)
    shutil.copyfile(config_file, os.path.join(bmk_path, 'config.py'))
    bmk_module = '.'.join(os.path.relpath(bmk_path, os.getcwd()).split(os.path.sep))
    return bmk_module

def gen_benchmark(benchmark:str, config_file:str, target_path='.',data_type:str='cv', task_type:str='classification'):
    r"""
        Create customized benchmarks from configurations. The configuration is a .py file that describes the datasets and the model,
        where there must exist a function named `get_model` and a variable `train_data`. `val_data` and test_data are two optional
        variables in the configuration.
    Args:
        benchmark (str): the name of the benchmark
        config_file (str): the path of the configuration file
        target_path: (str): the path to store the benchmark
        data_type (str): the type of dataset that should be in the list ['cv', 'nlp', 'graph', 'rec', 'series', 'tabular']
        task_type (str): the type of the task (e.g. classification, regression...)
    Returns:
        bmk_module (str): the module name of the generated benchmark
    """
    if not os.path.exists(config_file): raise FileNotFoundError('File {} not found.'.format(config_file))
    target_path = os.path.abspath(target_path)
    bmk_path = os.path.join(target_path, benchmark)
    if os.path.exists(bmk_path): raise FileExistsError('Benchmark {} already exists'.format(bmk_path))
    temp_path = os.path.join(flgo.benchmark.path, 'toolkits', data_type, task_type, 'temp')
    if not os.path.exists(temp_path):
        raise NotImplementedError('There is no support to automatically generation of {}.{}. More other types are comming soon...'.format(data_type, task_type))
    else:
        shutil.copytree(temp_path, bmk_path)
    shutil.copyfile(config_file, os.path.join(bmk_path, 'config.py'))
    bmk_module = '.'.join(os.path.relpath(bmk_path, os.getcwd()).split(os.path.sep))
    return bmk_module

def gen_decentralized_benchmark(benchmark:str, config_file:str, target_path = '.', data_type:str='cv', task_type:str='classification'):
    r"""
        Create customized benchmarks from configurations. The configuration is a .py file that describes the datasets and the model,
        where there must exist a function named `get_model` and a variable `train_data`. `val_data` and test_data are two optional
        variables in the configuration.
    Args:
        benchmark (str): the name of the benchmark
        config_file (str): the path of the configuration file
        target_path: (str): the path to store the benchmark
        data_type (str): the type of dataset that should be in the list ['cv', 'nlp', 'graph', 'rec', 'series', 'tabular']
        task_type (str): the type of the task (e.g. classification, regression...)
    Returns:
        bmk_module (str): the module name of the generated benchmark
    """
    if not os.path.exists(config_file): raise FileNotFoundError('File {} not found.'.format(config_file))
    target_path = os.path.abspath(target_path)
    bmk_path = os.path.join(target_path, benchmark)
    if os.path.exists(bmk_path): raise FileExistsError('Benchmark {} already exists'.format(bmk_path))
    temp_path = os.path.join(flgo.benchmark.path, 'toolkits', data_type, task_type, 'dec_temp')
    if not os.path.exists(temp_path):
        raise NotImplementedError('There is no support to automatically generation of {}.{}. More other types are comming soon...'.format(data_type, task_type))
    else:
        shutil.copytree(temp_path, bmk_path)
    shutil.copyfile(config_file, os.path.join(bmk_path, 'config.py'))
    bmk_module = '.'.join(os.path.relpath(bmk_path, os.getcwd()).split(os.path.sep))
    return bmk_module

def gen_hierarchical_benchmark(benchmark:str, config_file:str, target_path = '.', data_type:str='cv', task_type:str='classification'):
    r"""
        Create customized benchmarks from configurations. The configuration is a .py file that describes the datasets and the model,
        where there must exist a function named `get_model` and a variable `train_data`. `val_data` and test_data are two optional
        variables in the configuration.
    Args:
        benchmark (str): the name of the benchmark
        config_file (str): the path of the configuration file
        target_path: (str): the path to store the benchmark
        data_type (str): the type of dataset that should be in the list ['cv', 'nlp', 'graph', 'rec', 'series', 'tabular']
        task_type (str): the type of the task (e.g. classification, regression...)
    Returns:
        bmk_module (str): the module name of the generated benchmark
    """
    if not os.path.exists(config_file): raise FileNotFoundError('File {} not found.'.format(config_file))
    target_path = os.path.abspath(target_path)
    bmk_path = os.path.join(target_path, benchmark)
    if os.path.exists(bmk_path): raise FileExistsError('Benchmark {} already exists'.format(bmk_path))
    temp_path = os.path.join(flgo.benchmark.path, 'toolkits', data_type, task_type, 'hier_temp')
    if not os.path.exists(temp_path):
        raise NotImplementedError('There is no support to automatically generation of {}.{}. More other types are comming soon...'.format(data_type, task_type))
    else:
        shutil.copytree(temp_path, bmk_path)
    shutil.copyfile(config_file, os.path.join(bmk_path, 'config.py'))
    bmk_module = '.'.join(os.path.relpath(bmk_path, os.getcwd()).split(os.path.sep))
    return bmk_module

def gen_task_from_para(benchmark, bmk_para:dict={}, Partitioner=None, par_para:dict={}, task_path: str= '', rawdata_path:str= '', seed:int=0):
    r"""
    Generate a federated task according to the parameters of this function. The formats and meanings of the inputs are listed as below:

    Args:
        benchmark (package|str): the benchmark package or the module path of it
        bmk_para (dict): the customized parameter dict of the method TaskGenerator.__init__() of the benchmark
        Partitioner (flgo.benchmark.toolkits.partition.BasicPartitioner|str): the class of the Partitioner or the name of the Partitioner that was realized in flgo.benchmark.toolkits.partition
        par_para (dict): the customized parameter dict of the method Partitioner.__init__()
        task_path (str): the path to store the generated task
        rawdata_path (str): where the raw data will be downloaded\stored
        seed (int): the random seed used to generate the task

    Example:
    ```python
        >>> import flgo
        >>> import flgo.benchmark.mnist_classification as mnist
        >>> from flgo.benchmark.toolkits.partition import IIDPartitioner
        >>> # GENERATE TASK BY PASSING THE MODULE OF BENCHMARK AND THE CLASS OF THE PARTITIOENR
        >>> flgo.gen_task_from_para(benchmark=mnist, Partitioner = IIDPartitioner, par_para={'num_clients':100}, task_path='./mnist_gen_by_para1')
        >>> # GENERATE THE SAME TASK BY PASSING THE STRING
        >>> flgo.gen_task_from_para(benchmark='flgo.benchmark.mnist_classification', Partitioner='IIDPartitioner', par_para={'num_clients':100}, task_path='./mnist_gen_by_para2')
    ```
    """
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    torch.manual_seed(12+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if type(benchmark) is str: benchmark = importlib.import_module(benchmark)
    if not hasattr(benchmark, '__path__'): raise RuntimeError("benchmark should be a package or the path of a package")
    if Partitioner is not None:
        if type(Partitioner) is str:
            if Partitioner in globals().keys(): Partitioner = eval(Partitioner)
            else: Partitioner = getattr(flgo.benchmark.toolkits.partition, Partitioner)
        partitioner = Partitioner(**par_para)
    else:
        try:
            if hasattr(benchmark, 'default_partitioner'):
                Partitioner = getattr(benchmark, 'default_partitioner')
                default_partition_para = getattr(benchmark, 'default_partition_para') if hasattr(benchmark, 'default_partition_para') else {}
                partitioner = Partitioner(**default_partition_para)
            else:
                partitioner = None
        except:
            partitioner = None
    if rawdata_path!='': bmk_para['rawdata_path']=rawdata_path
    bmk_core = benchmark.core
    task_generator = getattr(bmk_core, 'TaskGenerator')(**bmk_para)
    if partitioner is not None:
        task_generator.register_partitioner(partitioner)
        partitioner.register_generator(task_generator)
    task_generator.generate()
    # save the generated federated benchmark
    # initialize task pipe
    if task_path=='': task_path = os.path.join('.', task_generator.task_name)
    task_pipe = getattr(bmk_core, 'TaskPipe')(task_path)
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
        print('Task {} has been successfully generated.'.format(task_pipe.task_path))
    except Exception as e:
        print(e)
        task_pipe.remove_task()
        print("Failed to saving splited dataset.")
    # save visualization
    try:
        visualize_func = getattr(benchmark,'visualize')
        visualize_func(task_generator, partitioner, task_path)
    except:
        pass

def gen_task(config={}, task_path:str= '', rawdata_path:str= '', seed:int=0):
    r"""
    Generate a federated task that is specified by the benchmark information and the partition information, where the generated task will be stored in the task_path and the raw data will be downloaded into the rawdata_path.

        config (dict || str): configuration is either a dict contains parameters or a filename of a .yml file
        task_path (str): where the generated task will be stored
        rawdata_path (str): where the raw data will be downloaded\stored
        seed (int): the random seed used to generate the task

    Example:
    ```python
        >>> import flgo
        >>> config = {'benchmark':{'name':'flgo.benchmark.mnist_classification'}, 'partitioner':{'name':'IIDParitioner', 'para':{'num_clients':100}}}
        >>> flgo.gen_task(config, './my_mnist_iid')
        >>> # The task will be stored as `my_mnist_iid` in the current working dictionary
    ```
    """
    # setup random seed
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    torch.manual_seed(12+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # load configuration
    gen_option = load_configuration(config)
    if type(gen_option['benchmark']) is not dict: gen_option['benchmark']={'name':gen_option['benchmark']}
    if 'para' not in gen_option['benchmark'].keys(): gen_option['benchmark']['para'] = {}
    if 'partitioner' in gen_option.keys():
        if not isinstance(gen_option['partitioner'], dict):
            gen_option['partitioner'] = {'name': gen_option['partitioner'], 'para':{}}
        # update parameters of partitioner
        if 'para' not in gen_option['partitioner'].keys():
            gen_option['partitioner']['para'] = {}
        else:
            if 'name' not in gen_option['partitioner'].keys():
                gen_option['benchmark']['para'].update(gen_option['partitioner']['para'])
    # init generator
    if rawdata_path!='': gen_option['benchmark']['para']['rawdata_path']=rawdata_path
    if type(gen_option['benchmark']['name']) is str:
        bmk_core = importlib.import_module('.'.join([gen_option['benchmark']['name'], 'core']))
    elif hasattr(gen_option['benchmark']['name'], '__path__'):
        bmk_core = importlib.import_module('.core', gen_option['benchmark']['name'].__name__)
    else:
        raise RuntimeError("The value of parameter config['benchmark']['name'] should be either a string or a python package.")
    task_generator = getattr(bmk_core, 'TaskGenerator')(**gen_option['benchmark']['para'])
    bmk_module = importlib.import_module(gen_option['benchmark']['name']) if type(
        gen_option['benchmark']['name']) is str else gen_option['benchmark']['name']
    # create partitioner for generator if specified
    if 'partitioner' in gen_option.keys() and 'name' in gen_option['partitioner'].keys():
        Partitioner = gen_option['partitioner']['name']
        if type(Partitioner) is str:
            if Partitioner in globals().keys(): Partitioner = eval(Partitioner)
            else: Partitioner = getattr(flgo.benchmark.partition, Partitioner)
        partitioner = Partitioner(**gen_option['partitioner']['para'])
        task_generator.register_partitioner(partitioner)
        partitioner.register_generator(task_generator)
    else:
        try:
            if hasattr(bmk_module, 'default_partitioner'):
                Partitioner = getattr(bmk_module, 'default_partitioner')
                default_partition_para = getattr(bmk_module, 'default_partition_para') if hasattr(bmk_module, 'default_partition_para') else {}
                partitioner = Partitioner(**default_partition_para)
                task_generator.register_partitioner(partitioner)
                partitioner.register_generator(task_generator)
            else:
                partitioner = None
        except:
            partitioner = None
    # generate federated task
    task_generator.generate()
    # save the generated federated benchmark
    # initialize task pipe
    if task_path=='': task_path = os.path.join('.', task_generator.task_name)
    task_pipe = getattr(bmk_core, 'TaskPipe')(task_path)
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
        print('Task {} has been successfully generated.'.format(task_pipe.task_path))
    except Exception as e:
        print(e)
        task_pipe.remove_task()
        print("Failed to saving splited dataset.")
    # save visualization
    if hasattr(bmk_module, 'visualize'):
        try:
            visualize_func = getattr(bmk_module,'visualize')
            visualize_func(task_generator, partitioner, task_path)
        except Exception as e:
            print('Warning: Failed to visualize the partitioned result where there exists error {}'.format(e))

def init(task: str, algorithm, option = {}, model=None, Logger: flgo.experiment.logger.BasicLogger = None, Simulator: BasicSimulator=flgo.simulator.DefaultSimulator, scene='horizontal'):
    r"""
    Initialize a runner in FLGo, which is to optimize a model on a specific task (i.e. IID-mnist-of-100-clients) by the selected federated algorithm.

    Args:
        task (str): the dictionary of the federated task
        algorithm (module|class): the algorithm will be used to optimize the model in federated manner, which must contain pre-defined attributions (e.g. algorithm.Server and algorithm.Client for horizontal federated learning)
        option (dict|str): the configurations of training, environment, algorithm, logger and simulator
        model (module|class): the model module that contains two methods: model.init_local_module(object) and model.init_global_module(object)
        Logger (flgo.experiment.logger.BasicLogger): the class of the logger inherited from flgo.experiment.logger.BasicLogger
        Simulator (flgo.simulator.base.BasicSimulator): the class of the simulator inherited from flgo.simulator.BasicSimulator
        scene (str): 'horizontal' or 'vertical' in current version of FLGo

    Returns:
        runner: the object instance that has the method runner.run()

    Example:
    ```python
        >>> import flgo
        >>> from flgo.algorithm import fedavg
        >>> from flgo.experiment.logger.simple_logger import SimpleLogger
        >>> # create task 'mnist_iid' by flgo.gen_task('gen_config.yml', 'mnist_iid') if there exists no such task
        >>> if os.path.exists('mnist_iid'): flgo.gen_task({'benchmark':{'name':'flgo.benchmark.mnist_classification'}, 'partitioner':{'name':'IIDPartitioner','para':{'num_clients':100}}}, 'mnist_iid')
        >>> # create runner
        >>> fedavg_runner = flgo.init('mnist_iid', algorithm=fedavg, option = {'num_rounds':20, 'gpu':[0], 'learning_rate':0.1})
        >>> fedavg_runner.run()
        ... # the training will start after runner.run() was called, and the running-time results will be recorded by Logger into the task dictionary
    ```
    """

    # init option
    option = load_configuration(option)
    default_option = read_option_from_command()
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
        else:
            default_option[op_key] = option[op_key]
    option = default_option
    setup_seed(seed=option['seed'])
    option['task'] = task
    option['algorithm'] = (algorithm.__name__).split('.')[-1]
    # option['server_with_cpu'] = True if (option['num_parallels']>1 and len(option['gpu'])>1) else option['server_with_cpu']
    # init task info
    if not os.path.exists(task):
        raise FileExistsError("Fedtask '{}' doesn't exist. Please generate the specified task by flgo.gen_task().")
    with open(os.path.join(task, 'info'), 'r') as inf:
        task_info = json.load(inf)
    benchmark = task_info['benchmark']
    if model== None:
        bmk_module = importlib.import_module(benchmark)
        if hasattr(bmk_module, 'default_model'):
            model = getattr(bmk_module, 'default_model')
        else:
            model = algorithm
    option['model'] = (model.__name__).split('.')[-1]
    # create global variable
    gv = GlobalVariable()
    # init logger
    if Logger is None:
        if scene=='horizontal':
            Logger = flgo.experiment.logger.simple_logger.SimpleLogger
        elif scene=='vertical':
            Logger = flgo.experiment.logger.vertical_logger.VerticalLogger
        elif scene=='decentralized':
            Logger = flgo.experiment.logger.dec_logger.DecLogger
        elif scene=='hierarchical':
            Logger = flgo.experiment.logger.hier_logger.HierLogger
    gv.logger = Logger(task=task, option=option, name=str(id(gv))+str(Logger), level=option['log_level'])
    # init device
    gv.dev_list = [torch.device('cpu')] if (option['gpu'] is None or len(option['gpu'])==0) else [torch.device('cuda:{}'.format(gpu_id)) for gpu_id in option['gpu']]
    gv.logger.info('Initializing devices: '+','.join([str(dev) for dev in gv.dev_list])+' will be used for this running.')
    # init task
    gv.logger.info('BENCHMARK:\t{}'.format(benchmark))
    gv.logger.info('TASK:\t\t\t{}'.format(task))
    gv.logger.info('MODEL:\t\t{}'.format(model.__name__))
    gv.logger.info('ALGORITHM:\t{}'.format(option['algorithm']))
    core_module = '.'.join([benchmark, 'core'])
    gv.TaskPipe = getattr(importlib.import_module(core_module), 'TaskPipe')
    task_pipe = gv.TaskPipe(task)
    gv.TaskCalculator = getattr(importlib.import_module(core_module), 'TaskCalculator')
    task_data = task_pipe.load_data(option)

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
    # init simulator
    if scene=='horizontal':
        for c in obj_class:
            if 'Client' in c:
                class_client = getattr(algorithm, c)
                class_client.train = flgo.simulator.base.with_completeness(class_client.train)
            elif 'Server' in c:
                class_server = getattr(algorithm, c)
                class_server.sample = flgo.simulator.base.with_availability(class_server.sample)
                class_server.communicate_with = flgo.simulator.base.with_latency(class_server.communicate_with)
                class_server.communicate = flgo.simulator.base.with_dropout(class_server.communicate)
    objects = task_pipe.generate_objects(option, algorithm, scene=scene)
    obj_classes = collections.defaultdict(int)
    for obj in objects: obj_classes[obj.__class__]+=1
    creating_str = []
    for k,v in obj_classes.items(): creating_str.append("{} {}".format(v, k))
    creating_str = ', '.join(creating_str)
    gv.logger.info('SCENE:\t\t{} FL with '.format(scene)+creating_str)
    task_pipe.distribute(task_data, objects)

    # init model
    if hasattr(model, 'init_local_module'):
        for object in objects:
            model.init_local_module(object)
    if hasattr(model, 'init_global_module'):
        for object in objects:
            model.init_global_module(object)
    if hasattr(model, 'init_dataset'):
        for object in objects:
            model.init_dataset(object)

    # init communicator
    gv.communicator = flgo.VirtualCommunicator(objects)

    for ob in objects: ob.initialize()

    # init virtual system environment
    gv.logger.info('SIMULATOR:\t{}'.format(str(Simulator)))
    # flgo.simulator.base.random_seed_gen = flgo.simulator.base.seed_generator(option['seed'])

    gv.clock = flgo.simulator.base.ElemClock()
    gv.simulator = Simulator(objects, option) if scene == 'horizontal' else None
    if gv.simulator is not None: gv.simulator.initialize()
    gv.clock.register_simulator(simulator=gv.simulator)
    gv.logger.register_variable(coordinator=objects[0], participants=objects[1:], option=option, clock=gv.clock, scene=scene, objects = objects, simulator=Simulator.__name__ if scene == 'horizontal' else 'None')
    if scene=='horizontal':
        gv.logger.register_variable(server=objects[0], clients=objects[1:])
    gv.logger.initialize()
    gv.logger.info('Ready to start.')

    # register global variables for objects
    for c in tmp:
        try:
            C = getattr(algorithm, c)
            delattr(C, 'gv')
        except:
            continue
    for ob in objects:
        ob.gv = gv
    if gv.simulator is not None:
        gv.simulator.gv = gv
    gv.clock.gv = gv
    gv.logger.gv = gv
    return objects[0]

def _call_by_process(task, algorithm_name,  opt, model_name, Logger, Simulator, scene, send_end):
    r"""
    This function is used to create a seperate child process.

    Args:
        task (str): the path of the task
        algorithm_name (str): the module name of algorithm
        opt (dict): option
        model_name (str): the module name of model
        Logger (flgo.experiment.logger.BasicLogger): the class of the logger
        Simulator (flgo.simulator.base.BasicSimulator): the class of the simulator inherited from flgo.simulator.BasicSimulator
        scene (str): horizontal or vertical
        send_end (connection.Connection): the return of multiprocess.Pipe(...) that is used to pass data to the parent process
    """

    pid = os.getpid()
    sys.stdout = open(os.devnull, 'w')
    if model_name is None: model = None
    else:
        try:
            model = importlib.import_module(model_name)
        except:
            model = model_name
    try:
        algorithm = importlib.import_module(algorithm_name)
    except:
        algorithm = algorithm_name
    try:
        runner = flgo.init(task, algorithm, model=model, option=opt, Logger=Logger, Simulator=Simulator, scene=scene)
        runner.run()
        res = (os.path.join(runner.gv.logger.get_output_path(), runner.gv.logger.get_output_name()), pid)
        send_end.send(res)
    except Exception as e:
        s = 'Process {} exits with error:" {}". '.format(pid, str(e))
        res = (opt, s, pid)
        send_end.send(res)

def tune(task: str, algorithm, option: dict = {}, model=None, Logger: flgo.experiment.logger.BasicLogger = flgo.experiment.logger.tune_logger.TuneLogger, Simulator: BasicSimulator=flgo.simulator.DefaultSimulator, scene='horizontal', scheduler=None):
    """
        Tune hyper-parameters for the specific (task, algorithm, model) in parallel.
        Args:
            task (str): the dictionary of the federated task
            algorithm (module|class): the algorithm will be used to optimize the model in federated manner, which must contain pre-defined attributions (e.g. algorithm.Server and algorithm.Client for horizontal federated learning)
            option (dict): the dict whose values should be of type list to construct the combinations
            model (module|class): the model module that contains two methods: model.init_local_module(object) and model.init_global_module(object)
            Logger (class): the class of the logger inherited from flgo.experiment.logger.BasicLogger
            Simulator (class): the class of the simulator inherited from flgo.simulator.BasicSimulator
            scene (str): 'horizontal' or 'vertical' in current version of FLGo
            scheduler (instance of flgo.experiment.device_scheduler.BasicScheduler): GPU scheduler that schedules GPU by checking their availability
        """
    # generate combinations of hyper-parameters
    if 'gpu' in option.keys():
        device_ids = option['gpu']
        option.pop('gpu')
        if not isinstance(device_ids, Iterable): device_ids = [device_ids]
    else:
        device_ids = [-1]
    keys = list(option.keys())
    for k in keys: option[k] = [option[k]] if (not isinstance(option[k], Iterable) or isinstance(option[k], str)) else option[k]
    para_combs = [para_comb for para_comb in itertools.product(*(option[k] for k in keys))]
    options = [{k:v for k,v in zip(keys, paras)} for paras in para_combs]
    for op in options:op['log_file'] = True
    if scheduler is None:
        scheduler = flgo.experiment.device_scheduler.BasicScheduler(device_ids)
    outputs = run_in_parallel(task, algorithm, options,model, devices=device_ids, Logger=Logger, Simulator=Simulator, scene=scene, scheduler=scheduler)
    optimal_idx = int(np.argmin([min(output['val_loss']) for output in outputs]))
    optimal_para = options[optimal_idx]
    print("The optimal combination of hyper-parameters is:")
    print('-----------------------------------------------')
    for k,v in optimal_para.items():
        if k=='gpu': continue
        print("{}\t|{}".format(k,v))
    print('-----------------------------------------------')
    op_round = np.argmin(outputs[optimal_idx]['val_loss'])
    if 'eval_interval' in option.keys(): op_round = option['eval_interval']*op_round
    print('The minimal validation loss occurs at the round {}'.format(op_round))

def run_in_parallel(task: str, algorithm, options:list = [], model=None, devices = [], Logger:flgo.experiment.logger.BasicLogger = flgo.experiment.logger.simple_logger.SimpleLogger, Simulator=flgo.simulator.DefaultSimulator, scene='horizontal', scheduler = None):
    """
    Run different groups of hyper-parameters for one task and one algorithm in parallel.

    Args:
        task (str): the dictionary of the federated task
        algorithm (module|class): the algorithm will be used to optimize the model in federated manner, which must contain pre-defined attributions (e.g. algorithm.Server and algorithm.Client for horizontal federated learning)
        options (list): the configurations of different groups of hyper-parameters
        model (module|class): the model module that contains two methods: model.init_local_module(object) and model.init_global_module(object)
        devices (list): the list of IDs of devices
        Logger (class): the class of the logger inherited from flgo.experiment.logger.BasicLogger
        Simulator (class): the class of the simulator inherited from flgo.simulator.BasicSimulator
        scene (str): 'horizontal' or 'vertical' in current version of FLGo
        scheduler (instance of flgo.experiment.device_scheduler.BasicScheduler): GPU scheduler that schedules GPU by checking their availability

    Returns:
        the returns of _call_by_process
    """
    try:
        # init multiprocess
        torch.multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    except:
        pass
    if model is None:
        model_name = None
    else:
        if not hasattr(model, '__module__') and hasattr(model, '__name__'):
            model_name = model.__name__
        else:
            model_name = model
    algorithm_name = algorithm.__name__ if (not hasattr(algorithm, '__module__') and hasattr(algorithm, '__name__')) else algorithm
    option_state = {oid:{'p':None, 'completed':False, 'output':None, 'option_in_queue':False, 'recv':None, } for oid in range(len(options))}
    if scheduler is None: scheduler = flgo.experiment.device_scheduler.BasicScheduler(devices)
    while True:
        for oid in range(len(options)):
            opt = options[oid]
            if option_state[oid]['p'] is None:
                if not option_state[oid]['completed']:
                    available_device = scheduler.get_available_device(opt)
                    if available_device is None: continue
                    else:
                        opt['gpu'] = available_device
                        recv_end, send_end = multiprocessing.Pipe(False)
                        option_state[oid]['p'] = multiprocessing.Process(target=_call_by_process, args=(task, algorithm_name, opt, model_name, Logger, Simulator, scene, send_end))
                        option_state[oid]['recv'] = recv_end
                        option_state[oid]['p'].start()
                        scheduler.add_process(option_state[oid]['p'].pid)
                        print('Process {} was created for args {}'.format(option_state[oid]['p'].pid,(task, algorithm_name, opt, model_name, Logger, Simulator, scene)))
            else:
                if option_state[oid]['p'].exitcode is not None:
                    tmp = option_state[oid]['recv'].recv()
                    scheduler.remove_process(tmp[-1])
                    try:
                        option_state[oid]['p'].terminate()
                    except:
                        pass
                    option_state[oid]['p'] = None
                    if len(tmp)==2:
                        option_state[oid]['completed'] = True
                        option_state[oid]['output'] = tmp[0]
                    else:
                        print(tmp[1])
        if all([v['completed'] for v in option_state.values()]):break
        time.sleep(1)
    res = []
    for oid in range(len(options)):
        rec_path = option_state[oid]['output']
        with open(rec_path, 'r') as inf:
            s_inf = inf.read()
            rec = json.loads(s_inf)
        res.append(rec)
    return res

def multi_init_and_run(runner_args:list, devices = [], scheduler=None):
    r"""
    Create multiple runners and run in parallel

    Args:
        runner_args (list): each element in runner_args should be either a dict or a tuple or parameters
        devices (list): a list of gpu id
        scheduler (flgo.experiment.device_scheduler.BasicScheduler(...)): GPU scheduler

    Returns:
        a list of output results of runners

    Example:
    ```python
        >>> from flgo.algorithm import fedavg, fedprox, scaffold
        >>> # create task 'mnist_iid' by flgo.gen_task if there exists no such task
        >>> task='./mnist_iid'
        >>> if os.path.exists(task): flgo.gen_task({'benchmark':{'name':'flgo.benchmark.mnist_classification'}, 'partitioner':{'name':'IIDPartitioner','para':{'num_clients':100}}}, task)
        >>> algos = [fedavg, fedprox, scaffold]
        >>> flgo.multi_init_and_run([{'task':task, 'algorithm':algo} for algo in algos], devices=[0])
    ```
    """
    if len(runner_args)==0:return
    args = []
    if type(runner_args[0]) is dict:
        keys = ['task', 'algorithm', 'option', 'model', 'Logger', 'Simulator', 'scene']
        for a in runner_args:
            tmp = collections.defaultdict(lambda:None, a)
            if tmp['task'] is None or tmp['algorithm'] is None:
                raise RuntimeError("keyword 'task' or 'algorithm' is of NoneType")
            algorithm = tmp['algorithm']
            tmp['algorithm'] = algorithm.__name__ if (not hasattr(algorithm, '__module__') and hasattr(algorithm, '__name__')) else algorithm
            if tmp['option'] is None:
                tmp['option'] = default_option_dict
            else:
                option = tmp['option']
                default_option = read_option_from_command()
                for op_key in option:
                    if op_key in default_option.keys():
                        op_type = type(default_option[op_key])
                        if op_type == type(option[op_key]):
                            default_option[op_key] = option[op_key]
                        else:
                            if op_type is list:
                                default_option[op_key] = list(option[op_key]) if hasattr(option[op_key],
                                                                                         '__iter__') else [
                                    option[op_key]]
                            elif op_type is tuple:
                                default_option[op_key] = tuple(option[op_key]) if hasattr(option[op_key],
                                                                                          '__iter__') else (
                                option[op_key])
                            else:
                                default_option[op_key] = op_type(option[op_key])
                tmp['option'] = default_option
            if tmp['model'] is None:
                model_name = None
            else:
                if not hasattr(tmp['model'], '__module__') and hasattr(tmp['model'], '__name__'):
                    model_name = tmp['model'].__name__
                else:
                    model_name = tmp['model']
            tmp['model'] = model_name
            if tmp['Logger'] is None:
                tmp['Logger'] = flgo.experiment.logger.simple_logger.SimpleLogger
            algorithm_name = tmp['algorithm'].__name__ if (not hasattr(tmp['algorithm'], '__module__') and hasattr(tmp['algorithm'], '__name__')) else tmp['algorithm']
            if tmp['Simulator'] is None:
                tmp['Simulator'] = flgo.simulator.DefaultSimulator
            if tmp['scene'] is None:
                tmp['scene'] = 'horizontal'
            args.append([tmp[k] for k in keys])
    elif type(runner_args[0]) is tuple or type(runner_args[0]) is list:
        for a in runner_args:
            if len(a)<2: raise RuntimeError('the args of runner should at least contain task and algorithm.')
            default_args = [None, None, default_option_dict, None, flgo.experiment.logger.simple_logger.SimpleLogger, flgo.simulator.DefaultSimulator, 'horizontal']
            for aid in range(len(a)):
                if aid==0:
                    default_args[aid] = a[aid]
                if aid==1:
                    algorithm = a[aid]
                    algorithm_name = algorithm.__name__ if (not hasattr(algorithm, '__module__') and hasattr(algorithm, '__name__')) else algorithm
                    default_args[aid] = algorithm_name
                elif aid==2:
                    option = a[aid]
                    default_option = read_option_from_command()
                    for op_key in option:
                        if op_key in default_option.keys():
                            op_type = type(default_option[op_key])
                            if op_type == type(option[op_key]):
                                default_option[op_key] = option[op_key]
                            else:
                                if op_type is list:
                                    default_option[op_key] = list(option[op_key]) if hasattr(option[op_key],
                                                                                             '__iter__') else [
                                        option[op_key]]
                                elif op_type is tuple:
                                    default_option[op_key] = tuple(option[op_key]) if hasattr(option[op_key],
                                                                                              '__iter__') else (
                                        option[op_key])
                                else:
                                    default_option[op_key] = op_type(option[op_key])
                    default_args[aid] = default_option
                elif aid==3:
                    model = a[aid]
                    if model is None:
                        model_name = None
                    else:
                        if not hasattr(model, '__module__') and hasattr(model, '__name__'):
                            model_name = model.__name__
                        else:
                            model_name = model
                    default_args[aid] = model_name
                else:
                    default_args[aid] = a[aid]

    runner_state = {rid: {'p': None, 'completed': False, 'output': None, 'runner_in_queue': False, 'recv': None, } for
                    rid in range(len(args))}
    if scheduler is None: scheduler = flgo.experiment.device_scheduler.BasicScheduler(devices)
    while True:
        for rid in range(len(args)):
            current_arg = args[rid]
            if runner_state[rid]['p'] is None:
                if not runner_state[rid]['completed']:
                    available_device = scheduler.get_available_device(current_arg)
                    if available_device is None:
                        continue
                    else:
                        list_current_arg = copy.deepcopy(current_arg)
                        list_current_arg[2]['gpu'] = available_device
                        recv_end, send_end = multiprocessing.Pipe(False)
                        list_current_arg.append(send_end)
                        runner_state[rid]['p'] = multiprocessing.Process(target=_call_by_process, args=tuple(list_current_arg))
                        runner_state[rid]['recv'] = recv_end
                        runner_state[rid]['p'].start()
                        scheduler.add_process(runner_state[rid]['p'].pid)
                        print('Process {} was created for args {}'.format(runner_state[rid]['p'].pid,current_arg))
            else:
                if runner_state[rid]['p'].exitcode is not None:
                    tmp = runner_state[rid]['recv'].recv()
                    scheduler.remove_process(tmp[-1])
                    try:
                        runner_state[rid]['p'].terminate()
                    except:
                        pass
                    runner_state[rid]['p'] = None
                    if len(tmp) == 2:
                        runner_state[rid]['completed'] = True
                        runner_state[rid]['output'] = tmp[0]
                    else:
                        print(tmp[1])
        if all([v['completed'] for v in runner_state.values()]): break
        time.sleep(1)
    res = []
    for rid in range(len(runner_state)):
        rec_path = runner_state[rid]['output']
        with open(rec_path, 'r') as inf:
            s_inf = inf.read()
            rec = json.loads(s_inf)
        res.append(rec)
    return res

def convert_model(get_model:Callable, model_name='anonymous', scene:str='horizontal'):
    r"""
    Convert an existing model into a model that can be loaded in flgo.
    Args:
        get_model (Callable): this function will return a model of type torch.nn.Module when it is called
        model_name (str): the name of the model
        scene (str): the FL scene

    Returns:
        res_model: the model can be used in flgo.init(..., model=res_model, ...)
    """
    class DecoratedModel(flgo.utils.fmodule.FModule):
        def __init__(self):
            super().__init__()
            self.model = get_model()

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    if scene=='horizontal':
        class AnonymousModel:
            __name__ = model_name

            @classmethod
            def init_global_module(self, object):
                if 'Server' in object.__class__.__name__:
                    object.model = DecoratedModel().to(object.device)

            @classmethod
            def init_local_module(self, object):
                pass

    elif scene=='decentralized':
        class AnonymousModel:
            __name__ = model_name

            @classmethod
            def init_local_module(self, object):
                if 'Client' in object.__class__.__name__:
                    object.model = DecoratedModel().to(object.device)

            @classmethod
            def init_global_module(self, object):
                pass
    else:
        raise NotImplementedError('The current version only support converting model for horizontalFL and DecentralizedFL.')
    return AnonymousModel()

def module2fmodule(Model):
    """
    Convert a class of torch.nn.Module into class flgo.utils.fmodule.FModule
    Args:
        Model (class): a class inherited from torch.nn.Module

    Returns:
        TempModule (class): The same class but additionally inheriting from flgo.utils.fmodule.FModule

    """
    class TempFModule(Model, flgo.utils.fmodule.FModule):
        def __init__(self, *args, **kwargs):
            super(TempFModule, self).__init__(*args, **kwargs)
    return TempFModule

def set_data_root(data_root:str=None):
    """
    Set the root of data that stores all the raw data automatically
    Args:
        data_root (str): the path of a directory whose default value is None and will be set to os.getcwd() as default.
    Returns:
    """
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'benchmark', '__init__.py')
    default_root = os.path.abspath(os.path.join(flgo.benchmark.path, 'RAW_DATA'))
    if data_root is None and os.path.abspath(flgo.benchmark.data_root)!=default_root:
        crt_root = default_root.strip()
        root_name = '"'+default_root.strip()+'"'
    elif data_root == 'cwd':
        crt_root = os.path.abspath(os.getcwd())
        root_name = 'os.getcwd()'
    else:
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        if not os.path.isdir(data_root):
            raise TypeError('data_root must be a dir')
        crt_root = os.path.abspath(data_root).strip()
        root_name = '"'+crt_root+'"'
    with open(file_path, 'r', encoding=sys.getfilesystemencoding()) as inf:
        lines = inf.readlines()
        idx = -1
        for i,line in enumerate(lines):
            if line.find('data_root')>-1:
                idx = i
                break
        if idx>0:
            lines[idx] = "data_root = "+ root_name
    with open(file_path, 'w', encoding=sys.getfilesystemencoding()) as outf:
        outf.writelines(lines)
    flgo.benchmark.data_root = crt_root
    print('Data root directory has successfully been changed to {}'.format(crt_root))
    return



