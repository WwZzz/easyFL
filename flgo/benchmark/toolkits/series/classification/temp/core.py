import os
from flgo.benchmark.toolkits.cv.classification import GeneralCalculator, FromDatasetPipe, FromDatasetGenerator
from .config import train_data
from torch.utils.data import Subset, ConcatDataset
import torch
try:
    import ujson as json
except:
    import json
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None

class TaskGenerator(FromDatasetGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)

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
            return torch.utils.data.random_split(dataset, [s2, s1])

    def load_data(self, running_time_option) -> dict:
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data
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

TaskCalculator = GeneralCalculator