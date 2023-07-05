import os
import torch
import torch.utils.data
from flgo.benchmark.base import FromDatasetPipe, FromDatasetGenerator, BasicTaskCalculator
from torchtext.data.functional import to_map_style_dataset
import math
try:
    import ujson as json
except:
    import json
from .config import train_data
from .config import language_pair, tokenizers, vocabs
from torch.nn.utils.rnn import pad_sequence
try:
    from .config import UNK_IDX
except:
    UNK_IDX = 0
try:
    from .config import PAD_IDX
except:
    PAD_IDX = 1
try:
    from .config import BOS_IDX
except:
    BOS_IDX = 2
try:
    from .config import EOS_IDX
except:
    EOS_IDX = 3
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

text_transform = {}
for ln in language_pair:
    text_transform[ln] = sequential_transforms(tokenizers[ln],  #Tokenization
                                               vocabs[ln],  #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[language_pair[0]](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[language_pair[1]](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
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
        loss = self.criterion(outputs, targets, PAD_IDX)
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

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=self.collect_fn)
