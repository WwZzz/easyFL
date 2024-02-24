import os
import torch
try:
    import ujson as json
except:
    import json
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator, FromDatasetPipe, FromDatasetGenerator
from .config import train_data
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset

class TaskGenerator(FromDatasetGenerator):
    def __init__(self, seq_len=None):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)
        self.seq_len = seq_len
        if self.seq_len is not None:
            if self.train_data is not None and hasattr(self.train_data, 'set_len'):
                self.train_data.set_len(*seq_len)
            if self.test_data is not None and hasattr(self.test_data, 'set_len'):
                self.test_data.set_len(*seq_len)
            if self.val_data is not None and hasattr(self.val_data, 'set_len'):
                self.val_data.set_len(*seq_len)

class TaskPipe(FromDatasetPipe):
    TaskDataset = Subset
    def __init__(self, task_path):
        super(FromDatasetPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)
        if os.path.exists(os.path.join(self.task_path, 'info')):
            with open(os.path.join(self.task_path, 'info'), 'r') as inf:
                self.info = json.load(inf)

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
        if val_data is None and test_data is not None:
            server_data_val, server_data_test = self.split_dataset(test_data, 1.0-running_time_option['test_holdout'])
        else:
            server_data_test = test_data
            server_data_val = val_data
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        partitioner_name = self.info.get('partitioner', None)
        if partitioner_name is not None and 'Concat' in partitioner_name:
            assert hasattr(train_data, 'datasets')
            all_datasets = train_data.datasets
            cdata_trains = []
            cdata_vals = []
            cdata_tests = []
            for cid, cname in enumerate(self.feddata['client_names']):
                dataset_ids = self.feddata[cname]['data']
                cdataset = [all_datasets[did] for did in dataset_ids]
                for cdata_i in cdataset:
                    cdata_train_i, cdata_val_i = self.split_dataset(cdata_i, running_time_option['train_holdout'])
                    if cdata_val_i is not None:
                        cdata_val_i, cdata_test_i = self.split_dataset(cdata_val_i, 0.5)
                    else:
                        cdata_test_i = None
                    if cdata_train_i is not None: cdata_trains.append(cdata_train_i)
                    if cdata_val_i is not None: cdata_vals.append(cdata_val_i)
                    if cdata_test_i is not None: cdata_tests.append(cdata_test_i)
                cdata_train = ConcatDataset(cdata_trains) if len(cdata_trains)>0 else None
                cdata_val = ConcatDataset(cdata_vals) if len(cdata_vals)>0 else None
                cdata_test = ConcatDataset(cdata_tests) if len(cdata_tests)>0 else None
                task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        else:
            for cid, cname in enumerate(self.feddata['client_names']):
                cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
                cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
                if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                    cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
                else:
                    cdata_test = None
                task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        return task_data

class TaskCalculator(GeneralCalculator):
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

