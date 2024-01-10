import random
import torch.utils.data
from tqdm import tqdm
import flgo.benchmark.base
from flgo.benchmark.base import *
from flgo.benchmark.toolkits.cv.segmentation.utils import ConfusionMatrix, collate_fn,cat_list
import os
FromDatasetGenerator = flgo.benchmark.base.FromDatasetGenerator

class FromDatasetPipe(flgo.benchmark.base.FromDatasetPipe):
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, perturbation=None, pin_memory=False):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices
            self.perturbation = {idx:p for idx, p in zip(indices, perturbation)} if perturbation is not None else None
            self.pin_memory = pin_memory
            self.num_classes = dataset.num_classes
            if not self.pin_memory:
                self.X = None
                self.Y = None
            else:
                self.X = torch.stack([self.dataset[i][0] for i in self.indices])
                self.Y = [self.dataset[i][1] for i in self.indices]

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
        if hasattr(train_data, 'num_classes'):
            num_classes = train_data.num_classes
        else:
            num_classes = 1
            setattr(train_data, 'num_classes', 1)
            if test_data is not None:
                setattr(test_data, 'num_classes', 1)
        # rearrange data for server
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        if server_data_val is not None: server_data_val.num_classes = num_classes
        if server_data_test is not None: server_data_test.num_classes = num_classes
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
            c_all_data = [cdata_train, cdata_val, cdata_test]
            for d in c_all_data:
                if d is not None: d.num_classes = num_classes
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
    def __init__(self, benchmark, rawdata_path, builtin_class, train_transform=None, test_transform=None, num_classes=0):
        super(BuiltinClassGenerator, self).__init__(benchmark, rawdata_path)
        self.num_classes = num_classes
        self.builtin_class = builtin_class
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.additional_option = {}
        self.train_additional_option = {}
        self.test_additional_option = {}
        self.download = True

    def load_data(self):
        # load the datasets
        train_default_init_para = {'root': self.rawdata_path, 'download':self.download, 'train':True, 'transforms':self.train_transform}
        test_default_init_para = {'root': self.rawdata_path, 'download':self.download, 'train':False, 'transforms':self.test_transform}
        train_default_init_para.update(self.additional_option)
        train_default_init_para.update(self.train_additional_option)
        test_default_init_para.update(self.additional_option)
        test_default_init_para.update(self.test_additional_option)
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
            self.perturbation = {idx: p for idx, p in zip(indices, perturbation)} if perturbation is not None else None
            self.pin_memory = pin_memory
            if not self.pin_memory:
                self.X = None
                self.Y = None
            else:
                self.X = [self.dataset[i][0] for i in self.indices]
                self.Y = [self.dataset[i][1] for i in self.indices]

        def __getitem__(self, idx):
            if self.X is not None:
                if self.perturbation is None:
                    return self.X[idx], self.Y[idx]
                else:
                    return self.X[idx] + self.perturbation[self.indices[idx]], self.Y[idx]
            else:
                if self.perturbation is None:
                    if isinstance(idx, list):
                        return self.dataset[[self.indices[i] for i in idx]]
                    return self.dataset[self.indices[idx]]
                else:
                    return self.dataset[self.indices[idx]][0] + self.perturbation[self.indices[idx]], \
                           self.dataset[self.indices[idx]][1]

    def __init__(self, task_path, buildin_class, train_transform=None, test_transform=None):
        super(BuiltinClassPipe, self).__init__(task_path)
        self.builtin_class = buildin_class
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.num_classes = 0

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))),  'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option, 'train_additional_option':generator.train_additional_option, 'test_additional_option':generator.test_additional_option, 'num_classes':generator.num_classes}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_default_init_para = {'root': self.feddata['rawdata_path'], 'download':True, 'train':True, 'transforms':self.train_transform}
        test_default_init_para = {'root': self.feddata['rawdata_path'], 'download':True, 'train':False, 'transforms':self.test_transform}
        if 'additional_option' in self.feddata.keys():
            train_default_init_para.update(self.feddata['additional_option'])
            test_default_init_para.update(self.feddata['additional_option'])
        if 'train_additional_option' in self.feddata.keys(): train_default_init_para.update(self.feddata['train_additional_option'])
        if 'test_additional_option' in self.feddata.keys(): test_default_init_para.update(self.feddata['test_additional_option'])
        train_pop_key = [k for k in train_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
        test_pop_key = [k for k in test_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
        for k in train_pop_key: train_default_init_para.pop(k)
        for k in test_pop_key: test_default_init_para.pop(k)
        train_data = self.builtin_class(**train_default_init_para)
        test_data = self.builtin_class(**test_default_init_para)
        test_data = self.TaskDataset(test_data, list(range(len(test_data))), None, running_time_option['pin_memory'])
        # rearrange data for server
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        num_classes = self.feddata['num_classes']
        if server_data_val is not None: server_data_val.num_classes = num_classes
        if server_data_test is not None: server_data_test.num_classes = num_classes
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
            if cdata_train is not None: cdata_train.num_classes = num_classes
            if cdata_val is not None: cdata_val.num_classes = num_classes
            if cdata_test is not None: cdata_test.num_classes = num_classes
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
        self.DataLoader = torch.utils.data.DataLoader
        self.criterion = self.compute_criterion

    def compute_criterion(self, inputs, target):
        if isinstance(inputs, torch.Tensor):
            return torch.nn.functional.cross_entropy(inputs, target, ignore_index=255)
        losses = {}
        for name, x in inputs.items():
            # if len(x.shape)==len(target.shape): target = torch.squeeze(target, 1)
            losses[name] = torch.nn.functional.cross_entropy(x, target, ignore_index=255)
        if len(losses) == 1:
            return losses["out"]
        return losses["out"] + 0.5 * losses["aux"]

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data (Any): the training dataset
        Returns:
            result (dict): dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.to_device(data)
        outputs = model(tdata[0])
        loss = self.criterion(outputs, tdata[-1])
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
        confmat = ConfusionMatrix(num_classes=dataset.num_classes)
        if batch_size==-1:batch_size=len(dataset)
        if len(dataset)==0: return {}
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        for batch_id, batch_data in tqdm(enumerate(data_loader), desc='Predicting...'):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0])
            loss = self.criterion(outputs, batch_data[-1])
            # if type(outputs) is not dict: outputs = {'out': outputs}
            if isinstance(outputs, dict):outputs = outputs['out']
            confmat.update(batch_data[-1].flatten(), outputs.argmax(1).flatten())
            total_loss += loss.item()*len(batch_data[0])
        acc_global, acc, iu = confmat.compute()
        mAcc = acc_global.item()*100
        classAcc = (acc*100).tolist()
        classIoU = (iu * 100).tolist()
        mIoU = (iu.mean()*100).item()
        total_loss = total_loss/len(dataset)
        return {'loss': total_loss,'mAcc':mAcc, 'classAcc':classAcc, 'mIoU':mIoU, 'classIoU':classIoU}

    def to_device(self, data:tuple):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=collate_fn)
