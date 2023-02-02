import importlib
import shutil
from abc import ABCMeta, abstractmethod
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import ujson

TASKROOT_PATH = './fedtask'

class AbstractTaskGenerator(metaclass=ABCMeta):
    @abstractmethod
    def load_data(self, *args, **kwarg):
        """Load the original data into memory that can be partitioned"""
        pass

    @abstractmethod
    def partition(self, *args, **kwarg):
        """Partition the loaded data into subsets of data owned by clients
        and the test data owned by the server
        """
        pass

    @abstractmethod
    def generate(self, *args, **kwarg):
        """Load and partition the data, and then generate the necessary
        information about the federated task (e.g. path, partition way, ...)"""
        pass


class AbstractTaskPipe(metaclass=ABCMeta):
    @abstractmethod
    def save_task(self, *args, **kwargs):
        """Save a federated task created by TaskGenerator as a static file on the disk"""
        pass

    @abstractmethod
    def load_task(self, *args, **kwargs):
        """Load a federated task from disk"""
        pass


class AbstractTaskCalculator(metaclass=ABCMeta):
    @abstractmethod
    def to_device(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_dataloader(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, model, data, *args, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, model, data, *args, **kwargs):
        pass

    @abstractmethod
    def get_optimizer(self, model, *args, **kwargs):
        pass


class BasicTaskGenerator(AbstractTaskGenerator):
    def __init__(self, benchmark, rawdata_path):
        """
        :param benchmark: the name of the ML task to be converted
        :param rawdata_path: the dictionary of the original dataset
        """
        # basic attribution
        self.benchmark = benchmark
        self.rawdata_path = rawdata_path
        # optional attribution
        self.partitioner = None
        self.train_data = None
        self.test_data = None
        self.task_name = None
        self.para = {}

    def generate(self, *args, **kwarg):
        """The whole process to generate federated task. """
        # load data
        self.load_data()
        # partition
        self.partition()
        # generate task name
        self.task_name = self.get_task_name()
        return

    def load_data(self, *args, **kwargs):
        """Download and load dataset into memory."""
        raise NotImplementedError

    def partition(self, *args, **kwargs):
        """Partition the data"""
        return

    def register_partitioner(self, partitioner=None):
        self.partitioner = partitioner

    def init_para(self, para_list=None):
        pnames = list(self.para.keys())
        if para_list is not None:
            for i, pv in enumerate(para_list):
                pname = pnames[i]
                try:
                    self.para[pname] = type(self.para[pname])(pv)
                except:
                    self.para[pname] = pv
        for pname, pv in self.para.items():
            self.__setattr__(pname, pv)
        return

    def get_task_name(self):
        return '_'.join(['B-' + self.benchmark, 'P-' + str(self.partitioner), 'N-' + str(self.partitioner.num_clients)])


class BasicTaskPipe(AbstractTaskPipe):
    TaskDataset = None

    def __init__(self, task_name):
        self.task_name = task_name
        self.task_path = os.path.join(TASKROOT_PATH, self.task_name)

    def save_task(self, generator):
        # Construct `feddata` and store it into the disk for recover the partitioned datasets again from it
        raise NotImplementedError

    def load_data(self, running_time_option) -> dict:
        # Load the data and process it to the format that can be distributed to different objects
        raise NotImplementedError

    def generate_objects(self, running_time_option) -> list:
        # Generate the virtual objects (i.e. coordinators and participants) in the FL system
        raise NotImplementedError

    def load_task(self, running_time_option, *args, **kwargs):
        with open(os.path.join(self.task_path, 'data.json'), 'r') as inf:
            self.feddata = ujson.load(inf)
        task_data = self.load_data(running_time_option)
        objects = self.generate_objects(running_time_option)
        self.distribute(task_data, objects)
        return objects

    def distribute(self, task_data: dict, objects: list):
        for ob in objects:
            ob_data = task_data[ob.name]
            for data_name, data in ob_data.items():
                ob.set_data(data, data_name)

    def split_dataset(self, dataset, p=0.0):
        if p == 0: return dataset, None
        s1 = int(len(dataset) * p)
        s2 = len(dataset) - s1
        return torch.utils.data.random_split(dataset, [s2, s1])

    def task_exists(self):
        """Check whether the task already exists."""
        return os.path.exists(self.task_path)

    def remove_task(self):
        "remove the task"
        if self.task_exists():
            shutil.rmtree(self.task_path)
        return

    def create_task_architecture(self):
        """Create the directories of the task."""
        if not self.task_exists():
            os.mkdir(self.task_path)
            os.mkdir(os.path.join(self.task_path, 'record'))
        else:
            raise FileExistsError("federated task {} already exists!".format(self.task_name))

    def save_figure(self):
        plt.savefig(os.path.join(self.task_path, 'res.png'))

    def gen_client_names(self, num_clients):
        return [('Client{:0>' + str(len(str(num_clients))) + 'd}').format(i) for i in range(num_clients)]


class BasicTaskCalculator(AbstractTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        self.device = device
        self.optimizer_name = optimizer_name
        self.criterion = None
        self.DataLoader = None

    def to_device(self, data, *args, **kwargs):
        return NotImplementedError

    def get_dataloader(self, *args, **kwargs):
        return NotImplementedError

    def test(self, model, data, *args, **kwargs):
        return NotImplementedError

    def compute_loss(self, model, data, *args, **kwargs):
        return NotImplementedError

    def get_optimizer(self, model=None, lr=0.1, weight_decay=0, momentum=0):
        OPTIM = getattr(importlib.import_module('torch.optim'), self.optimizer_name)
        filter_fn = filter(lambda p: p.requires_grad, model.parameters())
        if self.optimizer_name.lower() == 'sgd':
            return OPTIM(filter_fn, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.optimizer_name.lower() in ['adam', 'rmsprop', 'adagrad']:
            return OPTIM(filter_fn, lr=lr, weight_decay=weight_decay)
        else:
            raise RuntimeError("Invalid Optimizer.")


class HorizontalTaskPipe(BasicTaskPipe):
    def generate_objects(self, running_time_option):
        # init clients
        client_path = '%s.%s' % ('algorithm', running_time_option['algorithm'])
        Client = getattr(importlib.import_module(client_path), 'Client')
        clients = [Client(running_time_option) for _ in range(len(self.feddata['client_names']))]
        for cid, c in enumerate(clients):
            c.id = cid
            c.name = self.feddata['client_names'][cid]
        # init server
        server_path = '%s.%s' % ('algorithm', running_time_option['algorithm'])
        server_module = importlib.import_module(server_path)
        server = getattr(server_module, 'Server')(running_time_option)
        server.name = 'server'
        server.register_clients(clients)
        for c in clients: c.register_server(server)
        objects = [server]
        objects.extend(clients)
        return objects


class XYHorizontalTaskPipe(HorizontalTaskPipe):
    """
    This pipe is for supervised learning where each sample contains a feature $x_i$ and a label $y_i$
     that can be indexed by $i$.
    To use this pipe, it's necessary to set the attribute `test_data` of the generator to be a dict like:
        {'x': [...], 'y':[...]}
    and the attribute `local_datas` to be a list of the above dict that means the local data owned by clients:
        [{'x':[...], 'y':[...]}, ..., ]
    """
    TaskDataset = torch.utils.data.TensorDataset

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server': {'data': generator.test_data}}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid]}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)

    def load_data(self, running_time_option) -> dict:
        test_data = self.feddata['server']['data']
        test_data = self.TaskDataset(torch.tensor(test_data['x']), torch.tensor(test_data['y']))
        local_datas = [self.TaskDataset(torch.tensor(self.feddata[cname]['data']['x']),
                                        torch.tensor(self.feddata[cname]['data']['y'])) for cname in
                       self.feddata['client_names']]
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        for key in self.feddata['server'].keys():
            if key == 'data':
                continue
            task_data['server'][key] = self.feddata['server'][key]
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = local_datas[cid]
            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train': cdata_train, 'valid': cdata_valid}
            for key in self.feddata[cname]:
                if key == 'data':
                    continue
                task_data[cname][key] = self.feddata[cname][key]
        return task_data

# class IDXHorizontalTaskPipe(HorizontalTaskPipe):
#     TaskDataset = torch.utils.data.Subset
#     def save_task(self, generator):
#         client_names = self.gen_client_names(len(generator.local_datas))
#         feddata = {'client_names': client_names, 'server': {'data': generator.test_data}}
#         pass
