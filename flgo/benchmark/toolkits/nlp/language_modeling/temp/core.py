import os
import torch
import torch.utils.data
from typing import *
from torch import Tensor
from flgo.benchmark.base import FromDatasetPipe, FromDatasetGenerator, BasicTaskCalculator
try:
    import ujson as json
except:
    import json
from .config import train_data
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None

class TaskGenerator(FromDatasetGenerator):
    def __init__(self, seq_len=5, stride=1):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)
        self.seq_len = seq_len
        self.stride = stride

    def prepare_data_for_partition(self):
        return list(self.train_data)

def preprocess(data:List[Tensor], seq_len:int=5, stride:int=1):
    if data is None: return None
    all_data = torch.cat(data)
    seqs = [all_data[i:i+seq_len+1] for i in range(0, len(all_data)-seq_len-1, stride)]
    return torch.stack(seqs)

def collate_fn(batch):
    batch = torch.stack(batch)
    seq_len = batch.shape[-1]-1
    return batch[:,:seq_len], batch[:,1:]

class TaskPipe(FromDatasetPipe):
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            if data is None: return None
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, item):
            return self.data[item]

    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'seq_len': generator.seq_len, 'stride':generator.stride}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        seq_len = self.feddata['seq_len']
        stride = self.feddata['stride']
        server_test_data = self.TaskDataset(preprocess(list(self.test_data) if self.test_data is not None else None, seq_len=seq_len, stride=stride))
        server_val_data = self.TaskDataset(preprocess(list(self.val_data) if self.val_data is not None else None, seq_len=seq_len, stride=stride))
        if server_test_data is not None and server_val_data is None:
            server_test_data, server_val_data = self.split_dataset(server_test_data, running_time_option['test_holdout'])
        # rearrange data for server
        task_data = {'server': {'test': server_test_data, 'val': server_val_data}}
        self.train_data = list(self.train_data)
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = [self.train_data[did] for did in self.feddata[cname]['data']]
            cdata = self.TaskDataset(preprocess(cdata, seq_len=seq_len, stride=stride))
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

    def split_dataset(self, dataset, p=0.0):
        if dataset is None: return None, None
        if p==0.0: return dataset, None
        elif p==1.0: return None,dataset
        k = int(len(dataset)*p)
        return dataset[:k], dataset[k:]

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = torch.utils.data.DataLoader
        self.collect_fn = collate_fn
        self.criterion = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        sources, targets = self.to_device(data)
        outputs = model(sources)
        loss = self.criterion(outputs.view(-1, outputs.shape[-1]), targets.reshape(-1))
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
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0])
            batch_mean_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), batch_data[1].reshape(-1)).item()
            total_loss += batch_mean_loss*len(batch_data[0])
        total_loss = total_loss/len(dataset)
        return {'loss':total_loss}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=self.collect_fn)
