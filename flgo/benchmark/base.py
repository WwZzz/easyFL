import importlib
import shutil
from abc import ABCMeta, abstractmethod
import random
import os
try:
    import ujson as json
except:
    import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

class AbstractTaskGenerator(metaclass=ABCMeta):
    r"""
    Abstract Task Generator
    """
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
    r"""
    Abstract Task Pipe
    """
    @abstractmethod
    def save_task(self, *args, **kwargs):
        """Save a federated task created by TaskGenerator as a static file on the disk"""
        pass

    @abstractmethod
    def load_task(self, *args, **kwargs):
        """Load a federated task from disk"""
        pass


class AbstractTaskCalculator(metaclass=ABCMeta):
    r"""
    Abstract Task Calculator
    """
    @abstractmethod
    def to_device(self, *args, **kwargs):
        """Put the data into the gpu device"""
        pass

    @abstractmethod
    def get_dataloader(self, *args, **kwargs):
        """Return a data loader that splits the input data into batches"""
        pass

    @abstractmethod
    def test(self, model, data, *args, **kwargs):
        """Evaluate the model on the data"""
        pass

    @abstractmethod
    def compute_loss(self, model, data, *args, **kwargs):
        """Compute the loss of the model on the data to complete the forward process"""
        pass

    @abstractmethod
    def get_optimizer(self, model, *args, **kwargs):
        """Return the optimizer on the parameters of the model"""
        pass

class BasicTaskGenerator(AbstractTaskGenerator):
    r"""
        Load the original dataset and partition the
        original dataset into local data
    """
    def __init__(self, benchmark:str, rawdata_path:str):
        """
        Args:
            benchmark (str): the name of the federated task
            rawdata_path (str): the dictionary of the original dataset
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
        self.additional_option = {}
        self.train_additional_option = {}
        self.test_additional_option = {}

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
        """Partition the data into different local datasets"""
        return

    def register_partitioner(self, partitioner=None):
        """Register the partitioner as self's data partitioner"""
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
        r"""
        Create the default name of the task
        """
        if not hasattr(self.partitioner, 'num_parties') and hasattr(self.partitioner, 'num_clients'):
            self.partitioner.num_parties = self.partitioner.num_clients
        else: self.partitioner.num_parties = 'unknown'
        return '_'.join(['B-' + self.benchmark, 'P-' + str(self.partitioner), 'N-' + str(self.partitioner.num_parties)])

class BasicTaskPipe(AbstractTaskPipe):
    r"""
    Store the partition information of TaskGenerator into the disk
    when generating federated tasks.

    Load the original dataset and the partition information to
    create the federated scenario when optimizing models
    """
    TaskDataset = None

    def __init__(self, task_path):
        r"""
        Args:
            task_path (str): the path of the federated task
        """
        self.task_path = task_path
        if os.path.exists(os.path.join(self.task_path, 'data.json')):
            with open(os.path.join(self.task_path, 'data.json'), 'r') as inf:
                self.feddata = json.load(inf)

    def save_task(self, generator):
        """Construct `feddata` and store it into the disk for recovering
        the partitioned datasets again from it"""
        raise NotImplementedError

    def load_data(self, running_time_option) -> dict:
        """Load the data and process it to the format that can be distributed
        to different objects"""
        raise NotImplementedError

    def generate_objects(self, running_time_option, algorithm, scene='horizontal') -> list:
        r"""
        Generate the virtual objects (i.e. coordinators and participants)
        in the FL system

        Args:
            running_time_option (dict): the option (i.e. configuration)
            algorithm (module|class): algorithm
            scene (str): horizontal or vertical
        """
        if scene=='horizontal':
            # init clients
            Client = algorithm.Client
            clients = [Client(running_time_option) for _ in range(len(self.feddata['client_names']))]
            for cid, c in enumerate(clients):
                c.id = cid
                c.name = self.feddata['client_names'][cid]
            # init server
            server = algorithm.Server(running_time_option)
            server.name = 'server'
            server.id = -1
            # bind clients and server
            server.register_clients(clients)
            for c in clients: c.register_server(server)
            # return objects as list
            objects = [server]
            objects.extend(clients)
        elif scene=='vertical':
            PassiveParty = algorithm.PassiveParty
            ActiveParty = algorithm.ActiveParty
            objects = []
            for pid, pname in enumerate(self.feddata['party_names']):
                is_active = self.feddata[pname]['data']['with_label']
                obj = ActiveParty(running_time_option) if is_active else PassiveParty(running_time_option)
                obj.id = pid
                obj.name = pname
                objects.append(obj)
            for party in objects:
                party.register_objects(objects)
        return objects

    def save_info(self, generator):
        r"""
        Save the basic information of the generated task into the disk
        """
        info = {'benchmark': '.'.join(generator.__module__.split('.')[:-1])}
        info['scene'] = generator.scene if hasattr(generator, 'scene') else 'unknown'
        info['num_clients'] = generator.num_clients if hasattr(generator, 'num_clients') else (generator.num_parties if hasattr(self, 'num_parties') else 'unknown')
        with open(os.path.join(self.task_path, 'info'), 'w') as outf:
            json.dump(info, outf)

    def load_task(self, running_time_option, *args, **kwargs):
        r"""
        Load the generated task into disk and create objects in the federated
        scenario.
        """
        task_data = self.load_data(running_time_option)
        objects = self.generate_objects(running_time_option)
        self.distribute(task_data, objects)
        return objects

    def distribute(self, task_data: dict, objects: list):
        r"""
        Distribute the loaded local datasets to different objects in
        the federated scenario
        """
        for ob in objects:
            ob_data = task_data[ob.name]
            for data_name, data in ob_data.items():
                ob.set_data(data, data_name)

    def split_dataset(self, dataset, p=0.0):
        r"""
        Split the dataset into two parts.

        Args:
            dataset (torch.utils.data.Dataset): the dataset to be splitted
            p (float): the ratio of the splitting

        Returns:
            The two split parts
        """
        if p == 0: return dataset, None
        s1 = int(len(dataset) * p)
        s2 = len(dataset) - s1
        return torch.utils.data.random_split(dataset, [s2, s1])

    def task_exists(self):
        r"""
        Check whether the task already exists.

        Returns:
            True if the task already exists
        """
        return os.path.exists(self.task_path)

    def remove_task(self):
        r"""Remove this task"""
        if self.task_exists():
            shutil.rmtree(self.task_path)
        return

    def create_task_architecture(self):
        """Create the directories of the task."""
        if not self.task_exists():
            os.mkdir(self.task_path)
            os.mkdir(os.path.join(self.task_path, 'record'))
            os.mkdir(os.path.join(self.task_path, 'log'))
        else:
            raise FileExistsError("federated task {} already exists!".format(self.task_path))

    def gen_client_names(self, num_clients):
        r"""
        Generate the names of clients

        Returns:
            a list of strings
        """
        return [('Client{:0>' + str(len(str(num_clients))) + 'd}').format(i) for i in range(num_clients)]


class BasicTaskCalculator(AbstractTaskCalculator):
    r"""
    Support task-specific computation when optimizing models, such
    as putting data into device, computing loss, evaluating models,
    and creating the data loader
    """

    def __init__(self, device, optimizer_name='sgd'):
        r"""
        Args:
            device (torch.device): device
            optimizer_name (str): the name of the optimizer
        """
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
        r"""
        Create optimizer of the model parameters

        Args:
            model (torch.nn.Module): model
            lr (float): learning rate
            weight_decay (float): the weight_decay coefficient
            momentum (float): the momentum coefficient

        Returns:
            the optimizer
        """
        OPTIM = getattr(importlib.import_module('torch.optim'), self.optimizer_name)
        filter_fn = filter(lambda p: p.requires_grad, model.parameters())
        if self.optimizer_name.lower() == 'sgd':
            return OPTIM(filter_fn, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif self.optimizer_name.lower() in ['adam', 'rmsprop', 'adagrad']:
            return OPTIM(filter_fn, lr=lr, weight_decay=weight_decay)
        else:
            raise RuntimeError("Invalid Optimizer.")


# class HorizontalTaskPipe(BasicTaskPipe):
#     def generate_objects(self, running_time_option):
#         # init clients
#         client_path = '%s.%s' % ('algorithm', running_time_option['algorithm'])
#         Client = getattr(importlib.import_module(client_path), 'Client')
#         clients = [Client(running_time_option) for _ in range(len(self.feddata['client_names']))]
#         for cid, c in enumerate(clients):
#             c.id = cid
#             c.name = self.feddata['client_names'][cid]
#         # init server
#         server_path = '%s.%s' % ('algorithm', running_time_option['algorithm'])
#         server_module = importlib.import_module(server_path)
#         server = getattr(server_module, 'Server')(running_time_option)
#         server.name = 'server'
#         server.register_clients(clients)
#         for c in clients: c.register_server(server)
#         objects = [server]
#         objects.extend(clients)
#         return objects


class XYHorizontalTaskPipe(BasicTaskPipe):
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
            json.dump(feddata, outf)

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
