import os
import math
import torch
import torch.utils.data
from flgo.benchmark.toolkits.nlp.classification import GeneralCalculator
from flgo.benchmark.base import FromDatasetPipe, FromDatasetGenerator, BasicTaskCalculator
from torchtext.data.functional import to_map_style_dataset
from torch.nn.utils.rnn import pad_sequence
try:
    import ujson as json
except:
    import json
from .config import train_data
try:
    from .config import padding_value
except:
    padding_value = 0
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        if isinstance(src_sample, list):
            src_batch.append(torch.tensor(src_sample, dtype=torch.int64))
            tgt_batch.append(torch.tensor(tgt_sample, dtype=torch.int64))

        else:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
    src_batch = pad_sequence(src_batch, padding_value=padding_value)
    tgt_batch = pad_sequence(tgt_batch, padding_value=padding_value)
    return src_batch, tgt_batch

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
        feddata = {'client_names': client_names,}
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

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = torch.utils.data.DataLoader
        self.collect_fn = collate_fn
        self.criterion = self.loss_func

    def loss_func(self, outputs, targets, ignore_index=-100):
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(outputs[1:].view(-1, outputs.shape[-1]), targets[1:].view(-1))

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        sources, targets = self.to_device(data)
        outputs = model(sources, targets)
        loss = self.criterion(outputs, targets, padding_value)
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
            outputs = model(batch_data[0], batch_data[1])
            batch_mean_loss = self.criterion(outputs, batch_data[1], model.ignore_index if hasattr(model, 'ignore_index') else -100).item()
            total_loss += batch_mean_loss * batch_data[-1].shape[-1]
        total_loss = total_loss/len(dataset)
        ppl = math.exp(total_loss)
        return {'loss':total_loss, 'ppl':ppl}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=self.collect_fn)
