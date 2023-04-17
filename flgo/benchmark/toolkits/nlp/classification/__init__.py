from typing import Any
from collections.abc import Callable
from ...cv.classification import GeneralCalculator
import torch.utils.data
# import torchtext
from flgo.benchmark.base import BasicTaskCalculator, BasicTaskGenerator, BasicTaskPipe
import os
try:
    import ujson as json
except:
    import json

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
    TaskDataset = torch.utils.data.Subset
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
        # load train datapipe and convert it to train dataset
        train_options = self.feddata['additional_option'].copy()
        train_options.update(self.feddata['train_additional_option'])
        train_dp = self.build_datapipes(**train_options)
        train_dp = train_dp.map(lambda x: {'feature': x[0], 'label': x[1]})
        train_dp = train_dp.add_index('index')
        train_dp = train_dp.map(lambda x: (x['index'], (x['feature'], x['label'])))
        train_data = train_dp.to_map_datapipe()
        # load test datapipe and convert it to test dataset
        test_options = self.feddata['additional_option'].copy()
        test_options.update(self.feddata['train_additional_option'])
        test_dp = self.build_datapipes(**test_options)
        test_dp = test_dp.map(lambda x: {'feature': x[0], 'label': x[1]})
        test_dp = test_dp.add_index('index')
        test_dp = test_dp.map(lambda x: (x['index'], (x['feature'], x['label'])))
        test_data = test_dp.to_map_datapipe()
        test_data = self.TaskDataset(test_data, list(range(len(test_data))))
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