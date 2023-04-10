from torch import Tensor
from torch_geometric.data import Data, DataLoader

import collections
import numpy as np

from flgo.benchmark.base import *

from flgo.benchmark.toolkits import BasicTaskPipe, BasicTaskCalculator

class BuiltinClassGenerator(BasicTaskGenerator):
    def __init__(self, benchmark, rawdata_path, builtin_class, transform=None):
        super(BuiltinClassGenerator, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.transforms = transform

    def load_data(self):
        default_init_para = {'root': self.rawdata_path, 'download': self.download, 'train': True,'transform': self.transform}
        self.all_data, self.perm = self.builtin_class(root=self.rawdata_path, name=self.dataset_name, transform=self.transforms).shuffle(return_perm=True)
        self.num_samples = len(self.all_data)
        k = int(0.9 * self.num_samples)
        self.train_data = self.all_data[:k]
        self.test_data = list(range(self.num_samples))[k:]

    def load_data(self):

        default_init_para.update(self.additional_option)
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            pop_key = [k for k in default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in pop_key: default_init_para.pop(k)
        self.dataset = T.RandomNodeSplit(split='train_rest', num_val=0.0, num_test=self.test_rate)(
            self.builtin_class(**default_init_para).data)
        self.G = torch_geometric.utils.to_networkx(self.dataset, to_undirected=self.dataset.is_undirected(),
                                                   node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'])
        self.test_nodes = mask_to_index(self.dataset.test_mask).tolist() if self.test_rate > 0 else []
        self.train_nodes = mask_to_index(self.dataset.train_mask).tolist()
        self.train_data = nx.subgraph(self.G, self.train_nodes)

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)


class BuiltinClassPipe(BasicTaskPipe):
    def __init__(self, task_name, buildin_class, transform=None):
        super(BuiltinClassPipe, self).__init__(task_name)
        self.builtin_class = buildin_class
        self.transform = transform

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_nodes))
        feddata = {'client_names': client_names,
                   'server_data': list(range(len(generator.test_data))),
                   'rawdata_path': generator.rawdata_path,
                   'para': generator.para,
                   'dataset_name': generator.dataset_name,
                   'perm': generator.perm}
        for cid in range(len(client_names)):
            feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        dataset = self.builtin_class(root=self.feddata['rawdata_path'], name=self.feddata['dataset_name'],
                                     transform=self.transform)
        dataset = dataset[self.feddata['perm']]
        test_data = dataset[self.feddata['server_data']]
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.feddata[cname]['data']
            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train': cdata_train, 'valid': cdata_valid}
        return task_data


class GeneralCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = DataLoader

    def compute_loss(self, model, data):
        """
        Args: model: the model to train
                 data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.to_device(data)
        outputs = model(tdata[0])
        loss = self.criterion(outputs, tdata[-1])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        Args:
            dataset:
                 batch_size:
        Returns: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size == -1: batch_size = len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0])
            batch_mean_loss = self.criterion(outputs, batch_data[-1]).item()
            y_pred = outputs.data.max(1, keepdim=True)[1]
            correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
            num_correct += correct.item()
            total_loss += batch_mean_loss * len(batch_data[-1])
        return {'accuracy': 1.0 * num_correct / len(dataset), 'loss': total_loss / len(dataset)}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
