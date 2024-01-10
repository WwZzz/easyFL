from __future__ import annotations

import random
from typing import Any
import numpy as np
import torch.utils.data
import json
from flgo.benchmark.base import *
from torch.utils.data import DataLoader

def collate_fn(batch):
    enum_data = {k:[d[k] for d in batch] for k in batch[0]['__enum__']}
    unenum_data = {k:batch[0][k] for k in batch[0].keys() if k not in batch[0]['__enum__']}
    enum_data.update(unenum_data)
    return enum_data

class BuiltinClassGenerator(BasicTaskGenerator):
    def __init__(self, benchmark, rawdata_path, builtin_class, transform=None):
        super(BuiltinClassGenerator, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.transform = transform
        self.additional_option = {}
        self.train_additional_option = {}
        self.test_additional_option = {}

    def load_data(self, *args, **kwargs):
        # load the datasets
        train_default_init_para = {'root': self.rawdata_path, 'download':True, 'train':True, 'transform':self.transform}
        test_default_init_para = {'root': self.rawdata_path, 'download':False, 'train':False, 'transform':self.transform}
        train_default_init_para.update(self.additional_option)
        train_default_init_para.update(self.train_additional_option)
        test_default_init_para.update(self.additional_option)
        test_default_init_para.update(self.test_additional_option)
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            train_pop_key = [k for k in train_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            test_pop_key = [k for k in test_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in train_pop_key: train_default_init_para.pop(k)
            for k in test_pop_key: test_default_init_para.pop(k)
        # init datasets
        self.train_data = self.builtin_class(**train_default_init_para)
        self.test_data = self.builtin_class(**test_default_init_para)
        # handle train_data to triple (user, item, rating)
        self.train_data = [tuple(self.train_data[idx][k] for k in self.train_data.enum_data) for idx in range(len(self.train_data))]

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)

class BuiltinClassPipe(BasicTaskPipe):
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, index):
            self.enum_data = dataset.enum_data
            self.unenum_data = dataset.unenum_data
            self.num_users = dataset.num_users
            self.num_items = dataset.num_items
            for attr in self.enum_data:
                setattr(self, attr, getattr(dataset, attr)[index])
            for attr in self.unenum_data:
                setattr(self, attr, getattr(dataset, attr))

        def __len__(self):
            if len(self.enum_data)>0:
                return len(getattr(self, self.enum_data[0]))
            else:
                return 0

        def __getitem__(self, item):
            data = {attr: getattr(self, attr)[item] for attr in self.enum_data}
            data.update({attr: getattr(self, attr) for attr in self.unenum_data})
            data.update({'__enum__': self.enum_data})
            return data

    def __init__(self, task_path, buildin_class, transform=None):
        """
        Args:
            task_path (str): the path of the task
            builtin_class (class): class in torchvision.datasets
            transform (torchvision.transforms.*): the transform
        """
        super(BuiltinClassPipe, self).__init__(task_path)
        self.builtin_class = buildin_class
        self.transform = transform

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))),  'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option, 'train_additional_option':generator.train_additional_option, 'test_additional_option':generator.test_additional_option, }
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        train_default_init_para = {'root': self.feddata['rawdata_path'], 'download': True, 'train': True,
                                   'transform': self.transform}
        test_default_init_para = {'root': self.feddata['rawdata_path'], 'download': True, 'train': False,
                                  'transform': self.transform}
        train_default_init_para.update(self.feddata['additional_option'])
        train_default_init_para.update(self.feddata['train_additional_option'])
        test_default_init_para.update(self.feddata['additional_option'])
        test_default_init_para.update(self.feddata['test_additional_option'])
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            train_pop_key = [k for k in train_default_init_para.keys() if
                             k not in self.builtin_class.__init__.__annotations__]
            test_pop_key = [k for k in test_default_init_para.keys() if
                            k not in self.builtin_class.__init__.__annotations__]
            for k in train_pop_key: train_default_init_para.pop(k)
            for k in test_pop_key: test_default_init_para.pop(k)
        # init datasets
        self.train_data = self.builtin_class(**train_default_init_para)
        self.test_data = self.builtin_class(**test_default_init_para)
        test_idxs = list(range(len(self.test_data)))
        k = int(len(self.test_data)*(1-running_time_option['test_holdout']))
        server_test_idxs = test_idxs[:k]
        server_val_idxs = test_idxs[k:]
        server_test_data = self.TaskDataset(self.test_data, server_test_idxs) if len(server_test_idxs)>0 else None
        server_val_data = self.TaskDataset(self.test_data, server_val_idxs) if len(server_val_idxs)>0 else None
        task_data = {'server':{'test':server_test_data, 'val':server_val_data}}
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata_idxs = self.feddata[cname]['data']
            k1 = int(len(cdata_idxs)*(1-running_time_option['train_holdout']))
            k2 = int((1-0.5*running_time_option['train_holdout'])*len(cdata_idxs)) if running_time_option['local_test'] else int(len(cdata_idxs))
            cdata_train_idxs = cdata_idxs[:k1]
            cdata_val_idxs = cdata_idxs[k1:k2]
            cdata_test_idxs = cdata_idxs[k2:]
            cdata_train = self.TaskDataset(self.train_data, cdata_train_idxs) if len(cdata_train_idxs)>0 else None
            cdata_val = self.TaskDataset(self.train_data, cdata_val_idxs) if len(cdata_val_idxs)>0 else None
            cdata_test = self.TaskDataset(self.train_data, cdata_test_idxs) if len(cdata_test_idxs)>0 else None
            task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test':cdata_test}
        return task_data

class GeneralCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super().__init__(device, optimizer_name)
        self.criterion = torch.nn.L1Loss()
        self.DataLoader = DataLoader

    def compute_loss(self, model, data):
        batch_data = self.to_device(data)
        outputs = model(batch_data)
        loss = self.criterion(outputs,batch_data['rating'])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        mse_loss = 0.0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data)
            batch_mean_loss = self.criterion(outputs, batch_data['rating']).item()
            total_loss += batch_mean_loss * len(batch_data['rating'])
            mse_loss += ((outputs-batch_data['rating'])**2).sum().item()
        return {'mae':total_loss/len(dataset), 'rmse': np.sqrt(mse_loss/len(dataset))}

    def to_device(self, data):
        res = {}
        for k,v in data.items():
            try: res[k] = torch.Tensor(v).to(self.device)
            except: res[k] = v
        return res

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

class UserLevelCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super().__init__(device, optimizer_name)
        self.criterion = torch.nn.L1Loss()
        self.DataLoader = DataLoader

    def compute_loss(self, models:tuple|list, data:Any) -> dict:
        global_model, local_model = models
        if global_model is not None and hasattr(global_model, 'to'):global_model.to(self.device)
        if local_model is not None and hasattr(local_model, 'to'):local_model.to(self.device)
        batch_data = self.to_device(data)
        outputs = global_model(batch_data, local_model)
        loss = self.criterion(outputs, batch_data['rating'])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, models:tuple|list, dataset:Any, batch_size=64, num_workers=0, pin_memory=False):
        global_model, local_model = models
        if global_model is not None and hasattr(global_model, 'to'):global_model.to(self.device)
        if local_model is not None and hasattr(local_model, 'to'):local_model.to(self.device)
        global_model.eval()
        if batch_size == -1: batch_size = len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        mse_loss = 0.0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = global_model(batch_data, local_model)
            batch_mean_loss = self.criterion(outputs, batch_data['rating']).item()
            total_loss += batch_mean_loss * len(batch_data['rating'])
            mse_loss += ((batch_data['rating']-outputs)**2).sum().cpu().item()
        return {'mae': total_loss / len(dataset), 'rmse': np.sqrt(mse_loss/len(dataset))}

    def to_device(self, data):
        return {k: torch.Tensor(v).to(self.device) for k, v in data.items()}

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=collate_fn)