import argparse
import importlib
import random
import numpy as np
import os
import yaml

def set_random_seed(self, seed=0):
    """Set random seed"""
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_option():
    """basic parameters for generating federated task"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration for generating federated benchmark;', type=str, default='gen_config.yml')
    parser.add_argument('--seed', help='random seed;', type=int, default=0)
    parser.add_argument('--visualize', help='the function of visualizing the partitioned results', type=str,  default='')
    option = parser.parse_args()
    option = vars(option)
    return option

def load_configuration(config):
    with open(config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if 'para' not in cfg['benchmark'].keys(): cfg['benchmark']['para'] = {}
    if 'partitioner' in cfg.keys() and 'para' not in cfg['partitioner'].keys(): cfg['partitioner']['para'] = {}
    if 'partitioner' in cfg.keys() and 'name' not in cfg['partitioner'].keys() and 'para' in cfg['partitioner'].keys():
        cfg['benchmark']['para'].update(cfg['partitioner']['para'])
    return cfg

def initialize_generator(bmk_option):
    TaskGenerator = getattr(importlib.import_module('.'.join(['benchmark', bmk_option['name'], 'core'])), 'TaskGenerator')
    task_generator = TaskGenerator(**bmk_option['para'])
    return task_generator

def initialize_partitioner(pt_option):
    Partitioner = getattr(importlib.import_module('.'.join(['benchmark', 'toolkits', 'partition'])), pt_option['name'])
    partitioner = Partitioner(**pt_option['para'])
    return partitioner

def initialize_pipe(bmk_option, task_name):
    task_pipe = getattr(importlib.import_module('.'.join(['benchmark', bmk_option['name'], 'core'])), 'TaskPipe')(task_name)
    return task_pipe

if __name__ == '__main__':
    # augments
    option = read_option()
    set_random_seed(option['seed'])
    cfg = load_configuration(option['config'])
    # create task generator
    task_generator = initialize_generator(cfg['benchmark'])
    # create partitioner for generator if specified
    if 'partitioner' in cfg.keys():
        partitioner = initialize_partitioner(cfg['partitioner'])
        task_generator.register_partitioner(partitioner)
        partitioner.register_generator(task_generator)
    else:
        partitioner = None
    # generate federated task
    task_generator.generate()
    # save the generated federated benchmark
    # initialize task pipe
    task_pipe = initialize_pipe(cfg['benchmark'], task_generator.task_name+'_S-{}'.format(option['seed']))
    # check if task already exists
    if task_pipe.task_exists():
        raise FileExistsError('Task {} already exists.'.format(task_generator.task_name))
    try:
        # create task architecture
        task_pipe.create_task_architecture()
        # save task
        task_pipe.save_task(task_generator)
        print('Task {} has been successfully generated.'.format(task_generator.task_name))
    except Exception as e:
        print(e)
        task_pipe.remove_task()
        print("Failed to saving splited dataset.")
    # save visualization
    if option['visualize'] != '':
        visualize_func = getattr(importlib.import_module('.'.join(['benchmark', 'toolkits', 'visualization'])),
                                 option['visualize'])
        visualize_func(task_generator, partitioner)
        task_pipe.save_figure()