import importlib
import shutil
from abc import ABCMeta, abstractmethod
import os
from typing import Callable
try:
    import ujson as json
except:
    import json
import torch
import random
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
        original dataset into local_movielens_recommendation data
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
        self.val_data = None
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
        return

    def partition(self, *args, **kwargs):
        """Partition the data into different local_movielens_recommendation datasets"""
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
        elif scene=='hierarchical':
            server = algorithm.Server(running_time_option)
            server.id = -1
            server.name = 'server'
            edge_servers = [algorithm.EdgeServer(running_time_option) for _ in range(self.feddata['num_edge_servers'])]
            for sid in range(len(edge_servers)):
                edge_servers[sid].id = sid
                edge_servers[sid].name = 'edge_server'+str(sid)
                edge_servers[sid].clients = []
            server.register_clients(edge_servers)
            clients = [algorithm.Client(running_time_option) for _ in range(len(self.feddata['client_names']))]
            edge_server_clients = [[] for _ in edge_servers]
            for cid, c in enumerate(clients):
                c.id = cid+len(edge_servers)
                c.name = self.feddata['client_names'][cid]
                edge_server_clients[self.feddata['client_group'][c.name]].append(c)
            for edge_server, client_set in zip(edge_servers,edge_server_clients):
                edge_server.register_clients(client_set)
            objects = [server]
            objects.extend(edge_servers)
            objects.extend(clients)
        elif scene=='decentralized':
            # init clients
            Client = algorithm.Client
            clients = [Client(running_time_option) for _ in range(len(self.feddata['client_names']))]
            for cid, c in enumerate(clients):
                c.id = cid
                c.name = self.feddata['client_names'][cid]
            # init topology of clients
            topology = self.feddata['topology']
            for c in clients:
                c.topology = topology
            adjacent = self.feddata['adjacent']
            for cid,c in enumerate(clients):
                c.clients = [clients[k] for k,nid in enumerate(adjacent[cid]) if nid>0]
            # init protocol
            protocol = algorithm.Protocol(running_time_option)
            protocol.name = 'protocol'
            # bind clients and server
            protocol.clients = clients
            # return objects as list
            objects = [protocol]
            objects.extend(clients)
        elif scene=='parallel_horizontal':
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
            # return objects as list
            objects = [server]
            objects.extend(clients)
        elif scene=='real_hserver':
            server = algorithm.Server(running_time_option)
            server.name = 'server'
            objects = [server]
        elif scene=='real_hclient':
            client = algorithm.Client(running_time_option)
            objects = [client]
        return objects

    def save_info(self, generator):
        r"""
        Save the basic information of the generated task into the disk
        """
        info = {'benchmark': '.'.join(generator.__module__.split('.')[:-1])}
        info['scene'] = generator.scene if hasattr(generator, 'scene') else 'unknown'
        info['num_clients'] = generator.num_clients if hasattr(generator, 'num_clients') else (generator.num_parties if hasattr(self, 'num_parties') else 'unknown')
        try:
            info['bmk_path'] = os.path.dirname(generator.__module__.__file__)
        except:
            pass
        if hasattr(generator, 'partitioner'):
            info['partitioner'] = str(generator.partitioner)
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
        Distribute the loaded local_movielens_recommendation datasets to different objects in
        the federated scenario
        """
        for ob in objects:
            if ob.name in task_data.keys():
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
        if s1==0:
            return dataset, None
        elif s2==0:
            return None, dataset
        else:
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
        self.collect_fn = None

    def to_device(self, data, *args, **kwargs):
        return NotImplementedError

    def get_dataloader(self, dataset, batch_size=64, *args, **kwargs):
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

    def set_criterion(self, criterion:Callable)->None:
        self.criterion = criterion

    def set_collect_fn(self, collect_fn:Callable)->None:
        self.collect_fn = collect_fn

class XYHorizontalTaskPipe(BasicTaskPipe):
    """
    This pipe is for supervised learning where each sample contains a feature $x_i$ and a label $y_i$
     that can be indexed by $i$.
    To use this pipe, it's necessary to set the attribute `test_data` of the generator to be a dict like:
        {'x': [...], 'y':[...]}
    and the attribute `local_datas` to be a list of the above dict that means the local_movielens_recommendation data owned by clients:
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
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        for key in self.feddata['server'].keys():
            if key == 'data':
                continue
            task_data['server'][key] = self.feddata['server'][key]
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = local_datas[cid]
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['local_test'] and cdata_val is not None:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train': cdata_train, 'val': cdata_val, 'test': cdata_test}
            for key in self.feddata[cname]:
                if key == 'data':
                    continue
                task_data[cname][key] = self.feddata[cname][key]
        return task_data

class XHorizontalTaskPipe(BasicTaskPipe):
    TaskDataset = torch.utils.data.TensorDataset
    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server': {'data': generator.test_data}}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid]}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)

    def load_data(self, running_time_option) -> dict:
        test_data = self.feddata['server']['data']
        test_data = self.TaskDataset(torch.tensor(test_data['x']))
        local_datas = [self.TaskDataset(torch.tensor(self.feddata[cname]['data']['x']), ) for cname in self.feddata['client_names']]
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        for key in self.feddata['server'].keys():
            if key == 'data':
                continue
            task_data['server'][key] = self.feddata['server'][key]
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = local_datas[cid]
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['local_test'] and cdata_val is not None:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train': cdata_train, 'val': cdata_val, 'test':cdata_test}
            for key in self.feddata[cname]:
                if key == 'data':
                    continue
                task_data[cname][key] = self.feddata[cname][key]
        return task_data

class FromDatasetGenerator(BasicTaskGenerator):
    r"""
    This generator will do:
    1. Directly create train_data and test_data from input;
    2. Convert the train_data into the scheme that can be partitioned by Partitioner if necessary.
    """
    def __init__(self, benchmark, train_data, val_data=None, test_data=None):
        super(FromDatasetGenerator, self).__init__(benchmark=benchmark, rawdata_path='')
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def generate(self, *args, **kwarg):
        self.partition()

    def partition(self, *args, **kwargs):
        self.train_data = self.prepare_data_for_partition()
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)

    def prepare_data_for_partition(self):
        """Transform the attribution self.train_data into the format that can be received by partitioner"""
        return self.train_data

class FromDatasetPipe(BasicTaskPipe):
    TaskDataset = None
    def __init__(self, task_path, train_data, val_data = None, test_data=None):
        super(FromDatasetPipe, self).__init__(task_path)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

class DecentralizedFromDatasetPipe(FromDatasetPipe):
    TaskDataset = None
    def __init__(self, task_path, train_data, val_data = None, test_data=None):
        super(DecentralizedFromDatasetPipe, self).__init__(task_path, train_data, val_data, test_data)

    def save_topology(self, feddata:{}):
        client_names = feddata['client_names']
        feddata['topology'] = self.topology
        num_clients = len(client_names)
        if self.adjacent is None:
            if self.topology == 'mesh':
                feddata['adjacent'] = [[1.0 for __ in range(num_clients)] for _ in range(num_clients)]
                for i in range(num_clients):
                    feddata['adjacent'][i][i] = 0.0
            elif self.topology == 'line':
                feddata['adjacent'] = [[0.0 for __ in range(num_clients)] for _ in range(num_clients)]
                for i in range(1, num_clients):
                    feddata['adjacent'][i][((i - 1) % num_clients)] = 1.0
            elif self.topology == 'ring':
                feddata['adjacent'] = [[0.0 for __ in range(num_clients)] for _ in range(num_clients)]
                for i in range(num_clients):
                    feddata['adjacent'][i][((i - 1) % num_clients)] = 1.0
            elif self.topology == 'random':
                feddata['adjacent'] = [[0.0 for __ in range(num_clients)] for _ in range(num_clients)]
                p = 0.5
                for i in range(num_clients):
                    for k in range(num_clients):
                        if i == k: continue
                        if random.random() < p:
                            feddata['adjacent'][i][k] = 1.0
        return feddata

class HierFromDatasetGenerator(FromDatasetGenerator):
    def partition(self, *args, **kwargs):
        self.train_data = self.prepare_data_for_partition()
        self.local_datas = self.partitioner(self.train_data)
        self.num_edge_servers = len(self.local_datas)
        self.num_clients = sum([len(d) for d in self.local_datas])

class HierFromDatasetPipe(FromDatasetPipe):
    def create_feddata(self, generator):
        num_clients = sum([len(d) for d in generator.local_datas])
        client_names = self.gen_client_names(num_clients)
        feddata = {'client_names': client_names}
        feddata['client_group'] = {}
        k = 0
        client_data = {cname:{} for cname in client_names}
        for gid in range(len(generator.local_datas)):
            group_data = generator.local_datas[gid]
            for i in range(len(group_data)):
                feddata['client_group'][client_names[k+i]] = gid
                client_data[client_names[k+i]]['data'] = generator.local_datas[gid][i]
            k+=len(group_data)
        feddata['num_edge_servers'] = len(generator.local_datas)
        feddata.update(client_data)
        return feddata

    def save_task(self, generator):
        feddata = self.create_feddata(generator)
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return