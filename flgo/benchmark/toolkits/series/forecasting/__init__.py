import math

import flgo
import numpy
import torch
import os
import flgo.benchmark.base as fbb
from torch.utils.data import Subset
import torch.nn.functional as F
try:
    import ujson as json
except:
    import json
class BuiltinClassGenerator(fbb.BasicTaskGenerator):
    r"""
    Generator for the time series dataset.

    Args:
        benchmark (str): the name of the benchmark
        rawdata_path (str): the path storing the raw data
        builtin_class (class): dataset class
    """
    def __init__(self, benchmark, rawdata_path, builtin_class):
        super(BuiltinClassGenerator, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.additional_option = {}
        self.train_additional_option = {}
        self.test_additional_option = {}

    def load_data(self):
        # load the datasets
        train_default_init_para = {'root': self.rawdata_path, 'train': True}
        test_default_init_para = {'root': self.rawdata_path, 'train': False}
        train_default_init_para.update(self.additional_option)
        train_default_init_para.update(self.train_additional_option)
        test_default_init_para.update(self.additional_option)
        test_default_init_para.update(self.test_additional_option)
        # if 'kwargs' not in self.builtin_class.__init__.__annotations__:
        #     train_pop_key = [k for k in train_default_init_para.keys() if
        #                      k not in self.builtin_class.__init__.__annotations__]
        #     test_pop_key = [k for k in test_default_init_para.keys() if
        #                     k not in self.builtin_class.__init__.__annotations__]
        #     for k in train_pop_key: train_default_init_para.pop(k)
        #     for k in test_pop_key: test_default_init_para.pop(k)
        # init datasets
        self.train_data = self.builtin_class(**train_default_init_para)
        self.test_data = self.builtin_class(**test_default_init_para)

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)

class BuiltinClassPipe(fbb.BasicTaskPipe):
    r"""
    TaskPipe for the time series dataset.

    Args:
        task_path (str): the path of the task
        builtin_class (class): dataset class
    """
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, perturbation=None, pin_memory=False):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices
            self.perturbation = {idx:p for idx, p in zip(indices, perturbation)} if perturbation is not None else None
            self.pin_memory = pin_memory
            if not self.pin_memory:
                self.X = None
                self.Y = None
            else:
                self.X = torch.stack([self.dataset[i][0] for i in self.indices])
                self.Y = torch.stack([self.dataset[i][1] for i in self.indices])

        def __getitem__(self, idx):
            if self.X is not None:
                if self.perturbation is None:
                    return self.X[idx], self.Y[idx]
                else:
                    return self.X[idx]+self.perturbation[self.indices[idx]], self.Y[idx]
            else:
                if self.perturbation is None:
                    if isinstance(idx, list):
                        return self.dataset[[self.indices[i] for i in idx]]
                    return self.dataset[self.indices[idx]]
                else:
                    return self.dataset[self.indices[idx]][0] + self.perturbation[self.indices[idx]],  self.dataset[self.indices[idx]][1]

    def __init__(self, task_path, buildin_class):
        """
        Args:
            task_path (str): the path of the task
            builtin_class (class): dataset class
        """
        super(BuiltinClassPipe, self).__init__(task_path)
        self.builtin_class = buildin_class

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))),  'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option, 'train_additional_option':generator.train_additional_option, 'test_additional_option':generator.test_additional_option,}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        if hasattr(generator.partitioner, 'local_perturbation'): feddata['local_perturbation'] = generator.partitioner.local_perturbation
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_default_init_para = {'root': self.feddata['rawdata_path'], 'train': True}
        test_default_init_para = {'root': self.feddata['rawdata_path'], 'train': False}
        if 'additional_option' in self.feddata.keys():
            train_default_init_para.update(self.feddata['additional_option'])
            test_default_init_para.update(self.feddata['additional_option'])
        if 'train_additional_option' in self.feddata.keys(): train_default_init_para.update(self.feddata['train_additional_option'])
        if 'test_additional_option' in self.feddata.keys(): test_default_init_para.update(self.feddata['test_additional_option'])
        # if 'kwargs' not in self.builtin_class.__init__.__annotations__:
        #     train_pop_key = [k for k in train_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
        #     test_pop_key = [k for k in test_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
        #     for k in train_pop_key: train_default_init_para.pop(k)
        #     for k in test_pop_key: test_default_init_para.pop(k)
        train_data = self.builtin_class(**train_default_init_para)
        test_data = self.builtin_class(**test_default_init_para)
        test_data = self.TaskDataset(test_data, list(range(len(test_data))), None, running_time_option['pin_memory'])
        # rearrange data for server
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if  local_perturbation[cid] is None else [torch.tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], cpert, running_time_option['pin_memory'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        return task_data

class FromDatasetGenerator(fbb.FromDatasetGenerator):
    def __init__(self, seq_len=None, train_data=None, val_data=None, test_data=None):
        super(FromDatasetGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)
        self.seq_len = seq_len
        if self.seq_len is not None:
            if self.train_data is not None and hasattr(self.train_data, 'set_len'):
                self.train_data.set_len(*seq_len)
            if self.test_data is not None and hasattr(self.test_data, 'set_len'):
                self.test_data.set_len(*seq_len)
            if self.val_data is not None and hasattr(self.val_data, 'set_len'):
                self.val_data.set_len(*seq_len)

class FromDatasetPipe(fbb.FromDatasetPipe):
    TaskDataset = Subset
    def __init__(self, task_path, train_data=None, val_data=None, test_data=None):
        super(FromDatasetPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        if hasattr(generator, 'seq_len') and generator.seq_len is not None:
            feddata['seq_len'] = generator.seq_len
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def split_dataset(self, dataset, p=0.0):
        if p == 0: return dataset, None
        s1 = int(len(dataset) * p)
        s2 = len(dataset) - s1
        if s1==0:
            return dataset, None
        elif s2==0:
            return None, dataset
        else:
            all_idx = list(range(len(dataset)))
            d1_idx = all_idx[:s2]
            d2_idx = all_idx[s2:]
            return Subset(dataset, d1_idx), Subset(dataset, d2_idx)

    def load_data(self, running_time_option) -> dict:
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data
        seq_len = self.feddata.get('seq_len', None)
        if seq_len is not None:
            if hasattr(self.train_data, 'set_len'):
                self.train_data.set_len(*seq_len)
            if hasattr(self.val_data, 'set_len'):
                self.val_data.set_len(*seq_len)
            if hasattr(self.test_data, 'set_len'):
                self.test_data.set_len(*seq_len)
        # rearrange data for server
        if val_data is None:
            server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        else:
            server_data_test = test_data
            server_data_val = val_data
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        return task_data


class GeneralCalculator(fbb.BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.MSELoss()
        self.DataLoader = torch.utils.data.DataLoader

    def to_device(self, data):
        if len(data)==2:
            return data[0].to(self.device), data[1].to(self.device)
        elif len(data)==4:
            x,y = data[0].to(self.device), data[1].to(self.device)
            if isinstance(data[2], torch.Tensor):
                return x,y, data[2].to(self.device), data[3].to(self.device)
            else:
                return x,y, data[2],data[3]

    def compute_loss(self, model, data):
        data = self.to_device(data)
        if len(data)==2:
            x, y = data
            ypred = model(x)
        elif len(data)==4:
            x, y, xmark, ymark = data
            ypred = model(x, xmark, ymark)
        loss = self.criterion(ypred, y)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        """
        Metric = [mean_accuracy, mean_loss]

        Args:
            model:
            dataset:
            batch_size:
        Returns: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size == -1: batch_size = len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        mse = 0.0
        mae = 0.0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            if len(batch_data) == 2:
                x, y = batch_data
                outputs = model(x)
            elif len(batch_data) == 4:
                x, y, xmark, ymark = batch_data
                outputs = model(x, xmark, ymark)
            batch_mse = self.criterion(outputs, y).item()
            batch_mae = F.l1_loss(outputs, y).item()
            mse += batch_mse * len(batch_data[-1])
            mae += batch_mae * len(batch_data[-1])
        return {'mse': mse / len(dataset), 'mae': mae / len(dataset)}


