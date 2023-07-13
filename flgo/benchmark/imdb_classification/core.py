import os
import torch
import torch.utils.data
from flgo.benchmark.toolkits.nlp.classification import GeneralCalculator
from flgo.benchmark.base import FromDatasetPipe, FromDatasetGenerator
from torchtext.data.functional import to_map_style_dataset

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

def collate_batch(batch):
    label_list, text_list = [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list, label_list

class TaskGenerator(FromDatasetGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)

    def prepare_data_for_partition(self):
        return to_map_style_dataset(self.train_data)

class TaskPipe(FromDatasetPipe):
    TaskDataset = torch.utils.data.Subset
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        if self.test_data is not None:
            test_data = to_map_style_dataset(self.test_data)
            if self.val_data is not None:
                server_test_data = test_data
                server_val_data = to_map_style_dataset(self.val_data)
            elif running_time_option['test_holdout'] > 0:
                server_test_data, server_val_data = self.split_dataset(test_data, running_time_option['test_holdout'])
            else:
                server_test_data = test_data
                server_val_data = None
        elif self.val_data is not None:
            server_test_data = None
            server_val_data = to_map_style_dataset(self.val_data)
        else:
            server_test_data = server_val_data = None
        # rearrange data for server
        task_data = {'server': {'test': server_test_data, 'val': server_val_data}}
        train_data = to_map_style_dataset(self.train_data)
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

class TaskCalculator(GeneralCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = torch.utils.data.DataLoader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.collect_fn = collate_batch