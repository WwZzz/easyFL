import copy
import time
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable
import numpy as np
import torch
import os.path
import json
import flgo.system_simulator
import flgo.system_simulator.default_simulator
import flgo.system_simulator.base
import flgo.utils.fmodule
import flgo.experiment.logger.simple_logger
import flgo.experiment.logger.tune_logger
import flgo.experiment.logger
from flgo.system_simulator.base import BasicSimulator
import flgo.benchmark.toolkits.partition
import flgo.algorithm
import itertools
import argparse
import importlib
import random
import os
import yaml
import queue

sample_list=['uniform', 'md', 'full', 'uniform_available', 'md_available', 'full_available']
agg_list=['uniform', 'weighted_scale', 'weighted_com']
optimizer_list=['SGD', 'Adam', 'RMSprop', 'Adagrad']

class GlobalVariable:
    """this class is to create a buffer space for sharing variables across different parties for each runner respectively in a single machine"""
    def __init__(self):
        self.logger = None
        self.simulator = None
        self.clock = None
        self.dev_list = None
        self.TaskCalculator = None
        self.TaskPipe = None
        self.crt_dev = 0

    def apply_for_device(self):
        """apply for a new device from currently available ones (i.e. devices in self.dev_list)"""
        if self.dev_list is None: return None
        dev = self.dev_list[self.crt_dev]
        self.crt_dev = (self.crt_dev + 1) % len(self.dev_list)
        return dev

def setup_seed(seed):
    """fix all the random seed used in numpy, torch and random module"""
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def read_option_from_command():
    """load configuration for flgo.init from command lines"""
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
    parser.add_argument('--num_parallels', help="the number of parallels in the clients computing session", type=int, default=1)
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

def load_configuration(config={}):
    """load configuration for yml file or dict"""
    if type(config) is str and config.endswith('.yml'):
        with open(config) as f:
            option = yaml.load(f, Loader=yaml.FullLoader)
        return option
    elif type(config) is dict:
        return config

def gen_task_by_para(benchmark, bmk_para:dict={}, Partitioner=None, par_para:dict={}, task_path: str='', rawdata_path:str='', seed:int=0):
    r"""
    Generate a federated task according to the parameters of this function. The formats and meanings of the inputs are listed as below:
    :param
        benchmark (python package || str): the benchmark package or the module path of it
        bmk_para (dict): the customized parameter dict of the method TaskGenerator.__init__() of the benchmark
        Partitioner (class || str): the class of the Partitioner or the name of the Partitioner that was realized in flgo.benchmark.toolkits.partition
        par_para (dict): the customized parameter dict of the method Partitioner.__init__()
        task_path (str): the path to store the generated task
        rawdata_path (str): where the raw data will be downloaded\stored
        seed (int): the random seed used to generate the task
    :return

    Example:
        >>> import flgo
        >>> import flgo.benchmark.mnist_classification as mnist
        >>> from flgo.benchmark.toolkits.partition import IIDPartitioner
        >>> # GENERATE TASK BY PASSING THE MODULE OF BENCHMARK AND THE CLASS OF THE PARTITIOENR
        >>> flgo.gen_task_by_para(benchmark=mnist, Partitioner = IIDPartitioner, par_para={'num_clients':100}, task_path='./mnist_gen_by_para1')
        >>> # GENERATE THE SAME TASK BY PASSING THE STRING
        >>> flgo.gen_task_by_para(benchmark='flgo.benchmark.mnist_classification', Partitioner='IIDPartitioner', par_para={'num_clients':100}, task_path='./mnist_gen_by_para2')
    """
    if type(benchmark) is str: benchmark = importlib.import_module(benchmark)
    if not hasattr(benchmark, '__path__'): raise RuntimeError("benchmark should be a package or the path of a package")
    if Partitioner is not None:
        if type(Partitioner) is str:
            if Partitioner in globals().keys(): Partitioner = eval(Partitioner)
            else: Partitioner = getattr(flgo.benchmark.toolkits.partition, Partitioner)
        partitioner = Partitioner(**par_para)
    else: partitioner = None
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
        print('Task {} has been successfully generated.'.format(task_generator.task_name))
    except Exception as e:
        print(e)
        task_pipe.remove_task()
        print("Failed to saving splited dataset.")
    # save visualization
    try:
        visualize_func = getattr(benchmark,'visualize')
        visualize_func(task_generator, partitioner)
        task_pipe.save_figure()
    except:
        pass

def gen_task_by_config(config={}, task_path:str='', rawdata_path:str='', seed:int=0):
    r"""
    Generate a federated task that is specified by the benchmark information and the partition information, where the generated task will be stored in the task_path and the raw data will be downloaded into the rawdata_path.
    :param
        config (dict || str): configuration is either a dict contains parameters or a filename of a .yml file
        task_path (str): where the generated task will be stored
        rawdata_path (str): where the raw data will be downloaded\stored
        seed (int): the random seed used to generate the task
    :return

    Example:
        >>> import flgo
        >>> config = {'benchmark':{'name':'flgo.benchmark.mnist_classification'}, 'partitioner':{'name':'IIDParitioner', 'para':{'num_clients':100}}}
        >>> flgo.gen_task(config, './my_mnist_iid')
        >>> # The task will be stored as `my_mnist_iid` in the current working dictionary
    """
    # setup random seed
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # load configuration
    gen_option = load_configuration(config)
    if 'para' not in gen_option['benchmark'].keys(): gen_option['benchmark']['para'] = {}
    if 'partitioner' in gen_option.keys():
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
        bmk_core = gen_option['benchmark']['name'].core
    else:
        raise RuntimeError("The value of parameter config['benchmark']['name'] should be either a string or a python package.")
    task_generator = getattr(bmk_core, 'TaskGenerator')(**gen_option['benchmark']['para'])
    # create partitioner for generator if specified
    if 'partitioner' in gen_option.keys() and 'name' in gen_option['partitioner'].keys():
        Partitioner = gen_option['partitioner']['name']
        if type(Partitioner) is str:
            if Partitioner in globals().keys(): Partitioner = eval(Partitioner)
            else: Partitioner = getattr(flgo.benchmark.toolkits.partition, Partitioner)
        partitioner = Partitioner(**gen_option['partitioner']['para'])
        task_generator.register_partitioner(partitioner)
        partitioner.register_generator(task_generator)
    else:
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
        print('Task {} has been successfully generated.'.format(task_generator.task_name))
    except Exception as e:
        print(e)
        task_pipe.remove_task()
        print("Failed to saving splited dataset.")
    # save visualization
    try:
        visualize_func = getattr(importlib.import_module(gen_option['benchmark']['name']),'visualize')
        visualize_func(task_generator, partitioner)
        task_pipe.save_figure()
    except:
        pass

def init(task: str, algorithm, option = {}, model=None, Logger: flgo.experiment.logger.BasicLogger = flgo.experiment.logger.simple_logger.SimpleLogger, Simulator: BasicSimulator=flgo.system_simulator.DefaultSimulator, scene='horizontal'):
    r"""
    Initialize a runner in FLGo, which is to optimize a model on a specific task (i.e. IID-mnist-of-100-clients) by the selected federated algorithm.
    :param
        task (str): the dictionary of the federated task
        algorithm (module || class): the algorithm will be used to optimize the model in federated manner, which must contain pre-defined attributions (e.g. algorithm.Server and algorithm.Client for horizontal federated learning)
        option (dict || str): the configurations of training, environment, algorithm, logger and simulator
        model (module || class): the model module that contains two methods: model.init_local_module(object) and model.init_global_module(object)
        Logger (class): the class of the logger inherited from flgo.experiment.logger.BasicLogger
        Simulator (class): the class of the simulator inherited from flgo.system_simulator.BasicSimulator
        scene (str): 'horizontal' or 'vertical' in current version of FLGo
    :return
        runner: the object instance that has the method runner.run()

    Example:
        >>> import flgo
        >>> from flgo.algorithm import fedavg
        >>> from flgo.experiment.logger.simple_logger import SimpleLogger
        >>> # create task 'mnist_iid' by flgo.gen_task('gen_config.yml', 'mnist_iid') if there exists no such task
        >>> if os.path.exists('mnist_iid'): flgo.gen_task('gen_config.yml', 'mnist_iid')
        >>> # create runner
        >>> fedavg_runner = flgo.init('mnist_iid', algorithm=fedavg, option = {'num_rounds':20, 'gpu':[0], 'learning_rate':0.1})
        >>> fedavg_runner.run()
        ... # the training will start after runner.run() was called, and the running-time results will be recorded by Logger into the task dictionary
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
    option = default_option
    setup_seed(seed=option['seed'])
    option['task'] = task
    option['algorithm'] = (algorithm.__name__).split('.')[-1]

    # init task info
    if not os.path.exists(task):
        raise FileExistsError("Fedtask '{}' doesn't exist. Please generate the specified task by flgo.gen_task().")
    with open(os.path.join(task, 'info'), 'r') as inf:
        task_info = json.load(inf)
    benchmark = task_info['benchmark']
    if model== None: model = getattr(importlib.import_module(benchmark), 'default_model')
    option['model'] = (model.__name__).split('.')[-1]

    try:
        # init multiprocess
        torch.multiprocessing.set_start_method('spawn')
        torch.multiprocessing.set_sharing_strategy('file_system')
    except:
        pass

    # create global variable
    gv = GlobalVariable()
    # init logger
    gv.logger = Logger(task=task, option=option, name=str(id(gv))+str(Logger), level=option['log_level'])

    # init device
    gv.dev_list = [torch.device('cpu')] if (option['gpu'] is None or len(option['gpu'])==0) else [torch.device('cuda:{}'.format(gpu_id)) for gpu_id in option['gpu']]
    gv.logger.info('Initializing devices: '+','.join([str(dev) for dev in gv.dev_list])+' will be used for this running.')
    # init task
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
    objects = task_pipe.generate_objects(option, algorithm, scene=scene)
    task_pipe.distribute(task_data, objects)
    for c in tmp:
        C = getattr(algorithm, c)
        delattr(C, 'gv')
    # init model
    if hasattr(model, 'init_local_module'):
        for object in objects:
            model.init_local_module(object)
    if hasattr(model, 'init_global_module'):
        for object in objects:
            model.init_global_module(object)

    # init communicator
    gv.communicator = flgo.VirtualCommunicator(objects)

    for ob in objects: ob.initialize()

    # init virtual system environment
    gv.logger.info('Use `{}` as the system simulator'.format(str(Simulator)))
    flgo.system_simulator.base.random_seed_gen = flgo.system_simulator.base.seed_generator(option['seed'])
    gv.clock = flgo.system_simulator.base.ElemClock()
    gv.simulator = Simulator(objects, option)
    gv.clock.register_simulator(simulator=gv.simulator)

    gv.logger.register_variable(coordinator=objects[0], participants=objects[1:], option=option, clock=gv.clock, scene=scene)
    gv.logger.initialize()
    gv.logger.info('Ready to start.')

    # register global variables for objects
    for ob in objects:
        ob.gv = gv
    gv.simulator.gv = gv
    gv.clock.gv = gv
    gv.logger.gv = gv
    return objects[0]

def _call_by_process(task, algorithm_name,  opt, model_name, Logger, Simulator, scene):
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
        return runner.gv.logger.output
    except Exception as e:
        print(e)
        return (opt, e)

def get_available_device(device_ids):
    # dev_handlers = [pynvml.nvmlDeviceGetHandleByIndex(dev_id) for dev_id in device_ids]
    return random.choice(device_ids)

def tune(task: str, algorithm, option: dict = {}, model=None, Logger: flgo.experiment.logger.BasicLogger = flgo.experiment.logger.tune_logger.TuneLogger, Simulator: BasicSimulator=flgo.system_simulator.DefaultSimulator, scene='horizontal'):
    """
        Tune hyper-parameters for one task and one algorithm in parallel.
        :param
            task (str): the dictionary of the federated task
            algorithm (module || class): the algorithm will be used to optimize the model in federated manner, which must contain pre-defined attributions (e.g. algorithm.Server and algorithm.Client for horizontal federated learning)
            option (dict): the dict whose values should be of type list to construct the combinations
            model (module || class): the model module that contains two methods: model.init_local_module(object) and model.init_global_module(object)
            Logger (class): the class of the logger inherited from flgo.experiment.logger.BasicLogger
            Simulator (class): the class of the simulator inherited from flgo.system_simulator.BasicSimulator
            scene (str): 'horizontal' or 'vertical' in current version of FLGo
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
    # allocate gpu to different configurations
    crt_dev_idx = 0
    for op in options:
        op['gpu'] = [device_ids[crt_dev_idx]]
        crt_dev_idx = (crt_dev_idx+1)%len(device_ids)
        op['log_file'] = True
        # op['no_log_console'] = True
    outputs = run_in_parallel(task, algorithm, options,model, devices=device_ids, Logger=Logger, Simulator=Simulator, scene=scene)
    optimal_idx = int(np.argmin([min(output['valid_loss']) for output in outputs]))
    optimal_para = options[optimal_idx]
    print("The optimal combination of hyper-parameters is:")
    print('-----------------------------------------------')
    for k,v in optimal_para.items():
        if k=='gpu': continue
        print("{}\t|{}".format(k,v))
    print('-----------------------------------------------')
    op_round = np.argmin(outputs[optimal_idx]['valid_loss'])
    if 'eval_interval' in option.keys(): op_round = option['eval_interval']*op_round
    print('The minimal validation loss occurs at the round {}'.format(op_round))

def run_in_parallel(task: str, algorithm, options:list = [], model=None, devices = [], Logger:flgo.experiment.logger.BasicLogger = flgo.experiment.logger.simple_logger.SimpleLogger, Simulator=flgo.system_simulator.DefaultSimulator, scene='horizontal'):
    """
    Run different groups of hyper-parameters for one task and one algorithm in parallel.
    :param
        task (str): the dictionary of the federated task
        algorithm (module || class): the algorithm will be used to optimize the model in federated manner, which must contain pre-defined attributions (e.g. algorithm.Server and algorithm.Client for horizontal federated learning)
        options (list): the configurations of different groups of hyper-parameters
        model (module || class): the model module that contains two methods: model.init_local_module(object) and model.init_global_module(object)
        devices (list): the list of IDs of devices
        Logger (class): the class of the logger inherited from flgo.experiment.logger.BasicLogger
        Simulator (class): the class of the simulator inherited from flgo.system_simulator.BasicSimulator
        scene (str): 'horizontal' or 'vertical' in current version of FLGo
    :return
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
    mp = torch.multiprocessing.Pool(len(options))
    x = [mp.apply_async(_call_by_process, args=(task, algorithm_name, opt, model_name, Logger, Simulator, scene)) for opt in options]
    outputs = [None for _ in x]
    option_to_be_run = queue.Queue(len(options))
    completed = [False for _ in options]
    option_in_queue = [False for _ in options]
    while True:
        for i,xi in enumerate(x):
            try:
                completed[i] = xi.successful()
                tmp = xi.get()
                if not option_in_queue[i] and isinstance(tmp, tuple):
                    completed[i] = False
                    option_in_queue[i] = True
                    option_to_be_run.put((i, options[i]))
                elif outputs[i] is None or type(outputs[i]) is tuple:
                    outputs[i] = tmp
            except:
                continue
        if any(completed) and not option_to_be_run.empty():
            available_device = get_available_device(devices)
            if available_device is not None:
                i, opt = option_to_be_run.get()
                opt['gpu'] = available_device
                completed[i] = False
                option_in_queue[i] = False
                x[i] = mp.apply_async(_call_by_process, args=(task, algorithm_name, opt, model_name, Logger, Simulator, scene))
        # print('-------------------------------------------------')
        # for i in range(len(res)):
        #     if res[i]:
        #         print(str(options[i])+' is finished')
        #     else:
        #         print(str(options[i])+' is running')
        if option_to_be_run.empty() and all(completed):
            break
        time.sleep(1)
    mp.close()
    mp.join()
    return outputs