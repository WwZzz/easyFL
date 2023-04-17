import math

import flgo
import numpy
import torch

from flgo.benchmark.base import *

class BuiltinClassGenerator(BasicTaskGenerator):
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

class BuiltinClassPipe(BasicTaskPipe):
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
            builtin_class (class): class in torchvision.datasets
            transform (torchvision.transforms.*): the transform
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
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if  local_perturbation[cid] is None else [torch.tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], cpert, running_time_option['pin_memory'])
            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_valid, cdata_test = self.split_dataset(cdata_valid, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train':cdata_train, 'valid':cdata_valid, 'test': cdata_test}
        return task_data

class GeneralCalculator(BasicTaskCalculator):
    r"""
    Calculator for the time series dataset.

    Args:
        device (torch.device): device
        optimizer_name (str): the name of the optimizer
    """
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.MSELoss()
        self.DataLoader = torch.utils.data.DataLoader

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.to_device(data)
        outputs = model(tdata[0])
        loss = self.criterion(outputs, tdata[1])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        """
        Metric = [rse, corr]

        Args:
            model:
            dataset:
            batch_size:
        Returns: [rse, corr]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        predict = None
        Y = None
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0])
            if predict is None:
                predict = outputs
                Y = batch_data[1]
            else:
                predict = torch.cat((predict, outputs))
                Y = torch.cat((Y, batch_data[1]))
        return {'rse': self.RSE(predict, Y), 'corr': self.CORR(predict, Y)}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

    def RSE(self, predict, Ytest):
        mean = Ytest.mean(dim=0)
        sum1 = ((Ytest - predict)**2).sum().item()
        sum2 = ((Ytest - mean)**2).sum().item()
        return math.sqrt(sum1/sum2)

    def CORR(self, predict, Ytest):
        mean_1 = predict.mean(dim=0)
        mean_2 = Ytest.mean(dim=0)
        corr = ((Ytest - mean_2) * (predict - mean_1)).sum(dim=1) / torch.sqrt(((Ytest - mean_2)**2) * ((predict - mean_1)**2)).sum(dim=1)
        return corr.mean().item()
