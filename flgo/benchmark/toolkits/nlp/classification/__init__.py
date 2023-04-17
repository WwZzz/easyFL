from typing import Any
from collections.abc import Callable
# from ...cv.classification import GeneralCalculator
from torch.utils.data import random_split, Subset
from torchtext.data.functional import to_map_style_dataset
from flgo.benchmark.base import BasicTaskCalculator, BasicTaskGenerator, BasicTaskPipe
import os
try:
    import ujson as json
except:
    import json
import torch

class DataPipeGenerator(BasicTaskGenerator):
    def __init__(self, benchmark:str, rawdata_path:str, build_datapipes:Callable):
        super(DataPipeGenerator, self).__init__(benchmark, rawdata_path)
        self.build_datapipes = build_datapipes
        self.additional_option = {}
        self.train_additional_option = {}
        self.test_additional_option = {}

    def load_data(self, *args, **kwargs):
        # load train datapipe and convert it to train dataset
        train_options = self.additional_option.copy()
        train_options.update(self.train_additional_option)
        train_dp = self.build_datapipes(**train_options)
        train_dp = train_dp.map(lambda x: {'feature': x[0], 'label': x[1]})
        train_dp = train_dp.add_index('index')
        train_dp = train_dp.map(lambda x: (x['index'], (x['feature'], x['label'])))
        self.train_data = train_dp.to_map_datapipe()

    def partition(self, *args, **kwargs):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)

class DataPipeTaskPipe(BasicTaskPipe):
    TaskDataset = Subset
    def __init__(self, task_path, build_datapipes):
        super(DataPipeTaskPipe, self).__init__(task_path)
        self.build_datapipes = build_datapipes

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option, 'train_additional_option':generator.train_additional_option, 'test_additional_option':generator.test_additional_option, }
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        def feat2dict(x):
            return {'Feat_'+str(k):v for k,v in enumerate(x)}

        def fdict2tuple(x):
            return (x['__index__'], tuple(x.values()))
        # load train datapipe and convert it to train dataset
        train_options = self.feddata['additional_option'].copy()
        train_options.update(self.feddata['train_additional_option'])
        train_dp = self.build_datapipes(**train_options)
        train_data = to_map_style_dataset(train_dp)
        # load test datapipe and convert it to test dataset
        test_options = self.feddata['additional_option'].copy()
        test_options.update(self.feddata['train_additional_option'])
        test_dp = self.build_datapipes(**test_options)
        test_data = to_map_style_dataset(test_dp)
        # rearrange data for server
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            if running_time_option['train_holdout'] > 0:
                cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
                if running_time_option['local_test']:
                    cdata_valid, cdata_test = self.split_dataset(cdata_valid, 0.5)
                else:
                    cdata_test = None
            else:
                cdata_train = cdata
                cdata_valid, cdata_test = None, None
            task_data[cname] = {'train': cdata_train, 'valid': cdata_valid, 'test': cdata_test}
        return task_data

class GeneralCalculator(BasicTaskCalculator):
    r"""
    Calculator for the dataset in torchvision.datasets.

    Args:
        device (torch.device): device
        optimizer_name (str): the name of the optimizer
    """
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = torch.utils.data.DataLoader

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        label, text, offsets = self.to_device(data)
        outputs = model(text, offsets)
        loss = self.criterion(outputs, label)
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
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[1], batch_data[2])
            batch_mean_loss = self.criterion(outputs, batch_data[0]).item()
            # y_pred = outputs.data.max(1, keepdim=True)[1]
            # correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
            num_correct += (outputs.argmax(1)==batch_data[0]).sum().item()
            total_loss += batch_mean_loss * len(batch_data[-1])
        return {'accuracy': 1.0*num_correct/len(dataset), 'loss':total_loss/len(dataset)}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=self.collect_fn)
