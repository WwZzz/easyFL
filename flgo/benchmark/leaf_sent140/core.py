import json
import urllib
import zipfile
import torch
from torch.utils.data import Dataset
import flgo.benchmark
import os.path
from flgo.benchmark.toolkits import BasicTaskGenerator, BasicTaskCalculator
from flgo.benchmark.base import XYHorizontalTaskPipe as TaskPipe, BasicTaskPipe
import collections
import re
import os
import os.path
import json
import csv
import numpy as np

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

class SENTIMENT140(Dataset):
    def __init__(self, root, train=True, max_len=140):
        self.train = train
        self.root = root
        self.max_len = max_len

        if not os.path.exists(os.path.join(self.root, 'raw_data', 'embs.json')):
            self.load_emb()
        with open(os.path.join(self.root, 'raw_data', 'embs.json'), 'r') as inf:
            embs = json.load(inf)
        self.id2word = embs['vocab']
        self.word2id = {v: k for k, v in enumerate(self.id2word)}

        file = 'train_data.json' if self.train else 'test_data.json'
        if not os.path.exists(os.path.join(self.root, 'raw_data', file)):
            self.download()
        with open(os.path.join(self.root, 'raw_data', file), 'r') as f:
            data = json.load(f)
            if self.train:
                self.id = data['id']
            self.x = data['x']
            self.y = data['y']

    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.long), self.y[index]

    def __len__(self):
        return len(self.x)

    def load_emb(self):
        raw_path = os.path.join(self.root, 'raw_data')
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)
        embs_path = download_from_url("https://nlp.stanford.edu/data/glove.6B.zip",
                                      os.path.join(raw_path, 'tmp'))
        tar_paths = extract_from_zip(embs_path, raw_path)
        os.remove(tar_paths[0])
        os.remove(tar_paths[1])
        os.remove(tar_paths[2])
        os.remove(embs_path)
        with open(tar_paths[3], 'r') as inf:
            lines = inf.readlines()
        lines = [i.split() for i in lines]
        vocab = [i[0] for i in lines]
        emb_floats = [[float(n) for n in i[1:]] for i in lines]
        emb_floats.append([0.0 for _ in range(300)])  # for unknown word
        js = {'vocab': vocab, 'emba': emb_floats}
        with open(os.path.join(raw_path, 'embs.json'), 'w') as ouf:
            json.dump(js, ouf)

    def download(self):
        raw_path = os.path.join(self.root, 'raw_data')
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)
        src_path = download_from_url("https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
                                     os.path.join(raw_path, 'tmp'))
        tar_paths = extract_from_zip(src_path, raw_path)
        os.remove(src_path)
        with open(tar_paths[0], 'rt', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            test = list(reader)
        with open(tar_paths[1], 'rt', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            train = list(reader)

        test_user_data = {}
        for i in range(len(test)):
            y = 1 if test[i][0] == "4" else 0
            if test[i][4] not in test_user_data:
                test_user_data[test[i][4]] = {'x': [], 'y': []}
                test_user_data[test[i][4]]['x'].append(test[i][1:])
                test_user_data[test[i][4]]['y'].append(y)
            else:
                test_user_data[test[i][4]]['x'].append(test[i][1:])
                test_user_data[test[i][4]]['y'].append(y)
        test_users = list(test_user_data.keys())
        testXs = []
        testYs = []
        for user in test_users:
            examples = test_user_data[user]
            x = self.tokenizer(examples['x'])
            testXs.extend(x)
            testYs.extend(examples['y'])
        test_data = {
            'x': testXs,
            'y': testYs,
        }
        with open(os.path.join(self.root, 'raw_data', 'test_data.json'), 'w') as f:
            json.dump(test_data, f)

        train_user_data = {}
        for i in range(len(train)):
            y = 1 if train[i][0] == "4" else 0
            if train[i][4] not in train_user_data:
                train_user_data[train[i][4]] = {'x': [], 'y': []}
                train_user_data[train[i][4]]['x'].append(train[i][1:])
                train_user_data[train[i][4]]['y'].append(y)
            else:
                train_user_data[train[i][4]]['x'].append(train[i][1:])
                train_user_data[train[i][4]]['y'].append(y)
        train_users = list(train_user_data.keys())
        train_sample_ids = []
        trainXs = []
        trainYs = []
        cid = 0
        for user in train_users:
            examples = train_user_data[user]
            x = self.tokenizer(examples['x'])
            trainXs.extend(x)
            trainYs.extend(examples['y'])
            train_sample_ids.extend([cid] * len(x))
            cid += 1
        train_data = {
            'x': trainXs,
            'y': trainYs,
            'id': train_sample_ids
        }
        with open(os.path.join(self.root, 'raw_data', 'train_data.json'), 'w') as f:
            json.dump(train_data, f)
        os.remove(tar_paths[0])
        os.remove(tar_paths[1])

    def tokenizer(self, data):
        # [ID, Date, Query, User, Content]
        unk = len(self.word2id)
        processed_data = []
        for raw_text in data:
            line_list = self.split_line(raw_text[4])  # split phrase in words
            indl = [self.word2id[w] if w in self.word2id else unk for w in line_list[:self.max_len]]
            indl += [unk] * (self.max_len - len(indl))
            processed_data.append(indl)
        return processed_data

    def split_line(self, line):
        '''split given line/phrase into list of words
        Arguments:
            line: string representing phrase to be split
        Returnss:
            list of strings, with each string representing a word
        '''
        return re.findall(r"[\w']+|[.,!?;]", line)

class TaskGenerator(BasicTaskGenerator):
    def __init__(self,rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'SENTIMENT140')):
        super(TaskGenerator, self).__init__(benchmark='leaf_sent140',
                                            rawdata_path=rawdata_path)

    def load_data(self):
        self.train_data = SENTIMENT140(self.rawdata_path, train=True)
        self.test_data = SENTIMENT140(self.rawdata_path, train=False)
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
                   'rawdata_path': generator.rawdata_path}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_data = SENTIMENT140(root=self.feddata['rawdata_path'], train=True)
        test_data = SENTIMENT140(root=self.feddata['rawdata_path'], train=False)
        # rearrange data for server
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train': cdata_train, 'val': cdata_val}
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

        Returns:
            dict of train-one-step's result, which should at least contains the key 'loss'
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