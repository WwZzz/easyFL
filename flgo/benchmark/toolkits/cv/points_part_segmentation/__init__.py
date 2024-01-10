import random
import os

import numpy as np

try:
    import ujson as json
except:
    import json
import torch.utils.data
import flgo.benchmark.base

FromDatasetGenerator = flgo.benchmark.base.FromDatasetGenerator

class FromDatasetPipe(flgo.benchmark.base.FromDatasetPipe):
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, perturbation=None, pin_memory=False):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices
            self.perturbation = {idx:p for idx, p in zip(indices, perturbation)} if perturbation is not None else None
            self.pin_memory = pin_memory
            if hasattr(dataset, 'num_classes'):
                self.num_classes = dataset.num_classes
            else:
                yset = set()
                if isinstance(dataset[0][-1], torch.Tensor):
                    for d in dataset:
                        yset = yset.union(set(d[-1].unique().tolist()))
                elif isinstance(dataset[0][-1], np.ndarray):
                    yset = set()
                    for d in dataset:
                        yset = yset.union(set(torch.from_numpy(d[-1]).unique().tolist()))
                else:
                    yset = set()
                    for d in dataset:
                        yset = yset.union(set(torch.tensor(d[-1]).unique().tolist()))
                dataset.num_classes = len(yset)
                self.num_classes = dataset.num_classes
            self.num_part = dataset.num_part
            if not self.pin_memory:
                self.X = None
                self.Y = None
            else:
                self.X = torch.stack([self.dataset[i][0] for i in self.indices])
                self.Y = torch.LongTensor([self.dataset[i][1] for i in self.indices])

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
    def __init__(self, task_path, train_data, val_data=None, test_data=None):
        super(FromDatasetPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        if hasattr(generator.partitioner, 'local_perturbation'): feddata['local_perturbation'] = generator.partitioner.local_perturbation
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data
        # rearrange data for server
        if val_data is None:
            server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        else:
            server_data_test = test_data
            server_data_val = val_data
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if  local_perturbation[cid] is None else [torch.tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], cpert, running_time_option['pin_memory'])
            num_classes = cdata.num_classes
            num_part = cdata.num_part
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            if cdata_train is not None:
                cdata_train.num_classes = num_classes
                cdata_train.num_part = num_part
            if cdata_test is not None:
                cdata_test.num_classes = num_classes
                cdata_test.num_part = num_part
            if cdata_val is not None:
                cdata_val.num_classes = num_classes
                cdata_val.num_part = num_part
            task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        if server_data_val is not None:
            server_data_val.num_classes = train_data.num_classes
            server_data_val.num_part = train_data.num_part
        if server_data_test is not None:
            server_data_test.num_classes = train_data.num_classes
            server_data_test.num_part = train_data.num_part
        return task_data

class DecentralizedFromDatasetPipe(flgo.benchmark.base.DecentralizedFromDatasetPipe):
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
                self.Y = torch.LongTensor([self.dataset[i][1] for i in self.indices])

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
    def __init__(self, task_path, train_data, val_data=None, test_data=None):
        super(DecentralizedFromDatasetPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names}
        for cid in range(len(client_names)):
            feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        if hasattr(generator.partitioner, 'local_perturbation'):
            feddata['local_perturbation'] = generator.partitioner.local_perturbation
        feddata = self.save_topology(feddata)
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data
        # rearrange data for server
        if val_data is None:
            server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        else:
            server_data_test = test_data
            server_data_val = val_data
        task_data = {'protocol': {'test': server_data_test, 'val': server_data_val}}
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

class HierFromDatasetPipe(flgo.benchmark.base.HierFromDatasetPipe):
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
                self.Y = torch.LongTensor([self.dataset[i][1] for i in self.indices])

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
    def save_task(self, generator):
        feddata = self.create_feddata(generator)
        if hasattr(generator.partitioner, 'local_perturbation'): feddata['local_perturbation'] = generator.partitioner.local_perturbation
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option):
        # rearrange data for server
        if self.val_data is None:
            server_data_test, server_data_val = self.split_dataset(self.test_data, running_time_option['test_holdout'])
        else:
            server_data_test = self.test_data
            server_data_val = self.val_data
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if local_perturbation[cid] is None else [torch.tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(self.train_data, self.feddata[cname]['data'], cpert, running_time_option['pin_memory'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        return task_data

class BuiltinClassGenerator(flgo.benchmark.base.BasicTaskGenerator):
    r"""
    Generator for the dataset in torchvision.datasets.

    Args:
        benchmark (str): the name of the benchmark
        rawdata_path (str): the path storing the raw data
        builtin_class (class): class in torchvision.datasets
        transform (torchvision.transforms.*): the transform
    """
    def __init__(self, benchmark, rawdata_path, builtin_class, transform=None):
        super(BuiltinClassGenerator, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.transform = transform
        self.additional_option = {}
        self.train_additional_option = {}
        self.test_additional_option = {}
        self.download = True

    def load_data(self):
        # load the datasets
        train_default_init_para = {'root': self.rawdata_path, 'download':self.download, 'train':True, 'transform':self.transform}
        test_default_init_para = {'root': self.rawdata_path, 'download':self.download, 'train':False, 'transform':self.transform}
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

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)

class BuiltinClassPipe(flgo.benchmark.base.BasicTaskPipe):
    r"""
    TaskPipe for the dataset in torchvision.datasets.

    Args:
        task_path (str): the path of the task
        builtin_class (class): class in torchvision.datasets
        transform (torchvision.transforms.*): the transform
    """
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, perturbation=None, pin_memory=False):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices
            self.perturbation = {idx:p for idx, p in zip(indices, perturbation)} if perturbation is not None else None
            self.pin_memory = pin_memory
            if hasattr(dataset, 'classes'): self.classes = dataset.classes
            if hasattr(dataset, 'num_classes'): self.num_classes = dataset.num_classes
            if not self.pin_memory:
                self.X = None
                self.Y = None
            else:
                self.X = torch.stack([self.dataset[i][0] for i in self.indices])
                self.Y = torch.LongTensor([self.dataset[i][1] for i in self.indices])

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
        if hasattr(generator.partitioner, 'local_perturbation'): feddata['local_perturbation'] = generator.partitioner.local_perturbation
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_default_init_para = {'root': self.feddata['rawdata_path'], 'download':True, 'train':True, 'transform':self.transform}
        test_default_init_para = {'root': self.feddata['rawdata_path'], 'download':True, 'train':False, 'transform':self.transform}
        if 'additional_option' in self.feddata.keys():
            train_default_init_para.update(self.feddata['additional_option'])
            test_default_init_para.update(self.feddata['additional_option'])
        if 'train_additional_option' in self.feddata.keys(): train_default_init_para.update(self.feddata['train_additional_option'])
        if 'test_additional_option' in self.feddata.keys(): test_default_init_para.update(self.feddata['test_additional_option'])
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            train_pop_key = [k for k in train_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            test_pop_key = [k for k in test_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in train_pop_key: train_default_init_para.pop(k)
            for k in test_pop_key: test_default_init_para.pop(k)
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
            num_classes = cdata.num_classes
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            if cdata_train is not None: cdata_train.num_classes = num_classes
            if cdata_test is not None: cdata_test.num_classes = num_classes
            if cdata_val is not None: cdata_val.num_classes = num_classes
            task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        return task_data

class GeneralCalculator(flgo.benchmark.base.BasicTaskCalculator):
    r"""
    Calculator for the dataset in torchvision.datasets.

    Args:
        device (torch.device): device
        optimizer_name (str): the name of the optimizer
    """
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.NLLLoss()
        self.DataLoader = torch.utils.data.DataLoader
        def collect_fn_(x):
            return torch.stack([xi[0] for xi in x]).transpose(2,1), torch.cat([xi[1] for xi in x]), torch.stack([xi[2] for xi in x])
        # self.collect_fn = lambda x: (x[0].transpose(2, 1), x[1][:,0])
        self.collect_fn = collect_fn_

    def compute_loss(self, model, data):
        tdata = self.to_device(data)
        outputs = model(*tdata[:-1])
        if hasattr(model, 'compute_loss'):
            loss = model.compute_loss(outputs, tdata[-1])
        else:
            loss = self.criterion(outputs, tdata[-1])
        return {'loss':loss}

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
        npoints = 0
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(dataset.num_part)]
        total_correct_class = [0 for _ in range(dataset.num_part)]
        ious = []
        for batch_id, batch_data in enumerate(data_loader):
            cur_batch_size,_,NUM_POINT = batch_data[0].size()
            batch_data = self.to_device(batch_data)
            outputs = model(*batch_data[:-1])
            if hasattr(model, 'compute_loss'):
                batch_mean_loss = model.compute_loss(outputs, batch_data[-1]).item()
            else:
                batch_mean_loss = self.criterion(outputs, batch_data[-1]).item()
            cur_pred_val = outputs.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = batch_data[-1].cpu().data.numpy()
            total_loss += batch_mean_loss * len(batch_data[-1])
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)
            for l in range(dataset.num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                part_ious = [0.0 for _ in range(dataset.num_part)]
                for l in range(dataset.num_part):
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l] = 1.0
                    else:
                        part_ious[l] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                ious.append(part_ious)
        part_ious = np.mean(ious, axis=0)
        mean_iou = np.mean(part_ious)
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['iou'] = mean_iou
        test_metrics['loss'] = total_loss/len(dataset)
        return test_metrics

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=self.collect_fn)
