import collections
import os
import shutil
import urllib
import zipfile
import flgo.benchmark
import os.path
import numpy as np
import torch
from torchvision.datasets import utils

import json
from torch.utils.data import Dataset

from flgo.benchmark.toolkits import BasicTaskGenerator
from flgo.benchmark.base import BasicTaskPipe, BasicTaskCalculator


def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url:
        urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]

class REDDIT(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.raw_folder = os.path.join(self.root, 'raw_data')
        self.processed_folder = os.path.join(self.root, 'processed_data')

        file = 'train_data.json' if self.train else 'test_data.json'
        if not os.path.exists(os.path.join(self.processed_folder, file)):
            self.download()
        with open(os.path.join(self.processed_folder, file), 'r') as f:
            data = json.load(f)
        if self.train:
            self.id = data['id']
        self.x = data['x']
        self.y = data['y']

    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.long), self.y[index][-1]

    def __len__(self):
        return len(self.x)

    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        src_path = os.path.join(self.raw_folder, 'reddit_subsampled.zip')
        if not os.path.exists(src_path):
            download_from_url("https://drive.google.com/uc?id=1ISzp69JmaIJqBpQCX-JJ8-kVyUns8M7o&export=download",
                              src_path)
        extract_from_zip(src_path, self.raw_folder)
        # process and save as torch files0
        print('Processing...')
        with open(os.path.join(self.raw_folder, 'new_small_data', 'train_data.json'), 'r') as f:
            train_data = json.load(f)
        with open(os.path.join(self.raw_folder, 'new_small_data', 'val_data.json'), 'r') as f:
            val_data = json.load(f)
        with open(os.path.join(self.raw_folder, 'new_small_data', 'test_data.json'), 'r') as f:
            test_data = json.load(f)
        users = train_data['users']
        for user in users:
            train_data['user_data'][user]['x'].extend(val_data['user_data'][user]['x'])
            train_data['user_data'][user]['y'].extend(val_data['user_data'][user]['y'])

        trainXs = []
        trainYs = []
        train_seq_masks = []
        train_idx = []
        self.vocab = self.load_vocab(train_data['user_data'])['vocab']
        cid = 0
        for user in users:
            examples = train_data['user_data'][user]
            trainXs.extend(self._tokens_to_ids(examples['x']))
            y = [seq['target_tokens'] for seq in examples['y']]
            trainYs.extend(self._tokens_to_ids(y))
            train_idx.extend([cid] * len(y))
            cid += 1
        train_data = {
            'x': trainXs,
            'y': trainYs,
            'seq_mask': train_seq_masks,
            'id': train_idx
        }
        with open(os.path.join(self.processed_folder, 'train_data.json'), 'w') as f:
            json.dump(train_data, f)

        testXs = []
        testYs = []
        test_seq_masks = []
        for user in users:
            examples = test_data['user_data'][user]
            testXs.extend(self._tokens_to_ids(examples['x']))
            y = [seq['target_tokens'] for seq in examples['y']]
            testYs.extend(self._tokens_to_ids(y))
        test_data = {
            'x': testXs,
            'y': testYs,
            'seq_mask': test_seq_masks,
        }
        with open(os.path.join(self.processed_folder, 'test_data.json'), 'w') as f:
            json.dump(test_data, f)

    def load_vocab(self, data):
        counter = self.build_counter(data)
        return self.build_vocab(counter)

    def build_counter(self, train_data):
        train_tokens = []
        for u in train_data:
            for c in train_data[u]['x']:
                [train_tokens.extend(s) for s in c]

        counter = collections.Counter()
        counter.update(train_tokens)
        return counter

    def build_vocab(self, counter, vocab_size=10000):
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        count_pairs = count_pairs[:(vocab_size - 2)]  # -2 to account for the unknown and pad symbols

        words, _ = list(zip(*count_pairs))

        vocab = {}
        for i, w in enumerate(words):
            vocab[w] = i
        vocab['<UNK>'] = vocab_size - 1
        return {'vocab': vocab, 'size': vocab_size, 'unk_symbol': vocab['<UNK>'], 'pad_symbol': vocab['<PAD>']}

    def _tokens_to_ids(self, data):
        to_ret = [self.tokens_to_word_ids(seq) for seq in data]
        return to_ret

    def tokens_to_word_ids(self, token):
        return [self.vocab[word] if word in self.vocab else self.vocab['<UNK>'] for word in token[0]]

class TaskGenerator(BasicTaskGenerator):
    def __init__(self,  rawdata_path=os.path.join(flgo.benchmark.path, 'RAW_DATA', 'REDDIT')):
        super(TaskGenerator, self).__init__(benchmark='leaf_reddit',
                                            rawdata_path=rawdata_path)
    def load_data(self):
        self.train_data = REDDIT(self.rawdata_path, train=True)
        self.test_data = REDDIT(self.rawdata_path, train=False)
        return

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)

class TaskPipe(BasicTaskPipe):
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx):
            if isinstance(idx, list):
                return self.dataset[[self.indices[i] for i in idx]]
            return self.dataset[self.indices[idx]]

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))),
                   'rawdata_path': generator.root}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_data = REDDIT(root=self.feddata['rawdata_path'], train=True)
        test_data = REDDIT(root=self.feddata['rawdata_path'], train=False)
        # rearrange data for server
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train': cdata_train, 'valid': cdata_valid}
        return task_data

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = torch.utils.data.DataLoader

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
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
            model:
            dataset:
            batch_size:
        Returns: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
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
        return {'accuracy': 1.0*num_correct/len(dataset), 'loss':total_loss/len(dataset)}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
