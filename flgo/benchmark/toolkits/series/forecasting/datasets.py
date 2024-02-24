import gzip

import numpy
import ujson as json
import os
import urllib
import numpy as np

import torch
from torch.utils.data import Dataset


def download_from_url(url=None, filepath='.'):
    """Download dataset from url to filepath."""
    if url:
        urllib.request.urlretrieve(url, filepath)
    return filepath


def extract_from_gz(src_file, target_file):
    """Unzip the .zip file (src_path) to target_path"""
    with open(target_file, 'wb') as f:
        zf = gzip.open(src_file, mode='rb')
        f.write(zf.read())
        zf.close()
    return target_file


def normalized(rawdata, normalize):
    n, m = rawdata.shape
    scale = numpy.ones(m)
    if normalize == 0:
        data = rawdata
    elif normalize == 1:
        data = rawdata / np.max(rawdata)
    elif normalize == 2:
        data = np.zeros((n, m))
        for i in range(m):
            scale[i] = np.max(np.abs(rawdata[:, i]))
            data[:, i] = rawdata[:, i] / np.max(np.abs(rawdata[:, i]))
    else:
        raise RuntimeError("The parameter 'normalize' can only take values from 0, 1, 2")
    return data


class BuiltinClassDataset(Dataset):

    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        self.raw_folder = os.path.join(self.root, 'raw_data')
        self.processed_folder = os.path.join(self.root, 'processed_data')
        if not hasattr(self, 'file'):
            self.file = 'train_data.json' if self.train else 'test_data.json'
        else:
            self.file = self.file + 'train_data.json' if self.train else self.file + 'test_data.json'
        if not os.path.exists(os.path.join(self.processed_folder, self.file)):
            self.download()
        with open(os.path.join(self.processed_folder, self.file), 'r') as f:
            data = json.load(f)
        self.x = torch.tensor(data['x'], dtype=torch.float)
        self.y = torch.tensor(data['y'], dtype=torch.float)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def download(self):
        raise NotImplementedError("please override the download() method in class {}".format(self.__class__.__name__))


class Electricity(BuiltinClassDataset):
    """
    在download中将原始数据进行分片并划分为训练集和测试集，分别存储到train_data.json和test_data.json两个文件中
    文件路径默认设置为 self.processed_folder/XX_data.json
    """

    def __init__(self, root, train=True, window=24, horizon=12, normalize=2):
        self.normalize = normalize
        self.url = \
            "https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz"
        self.file = 'w_' + str(window) + '_h_' + str(horizon) + '_'
        self.window = window
        self.horizon = horizon
        super(Electricity, self).__init__(root, train)

    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        src_path = os.path.join(self.raw_folder, 'electricity.txt.gz')
        if not os.path.exists(src_path):
            download_from_url(self.url, src_path)
        src_path = extract_from_gz(src_path, os.path.join(self.raw_folder, 'electricity.txt'))
        raw_data = np.loadtxt(src_path, delimiter=',')
        data = normalized(raw_data, self.normalize)
        self.split(data)

    def split(self, data):
        n, m = data.shape
        s1 = int(n * 0.8)
        train_idx = range(self.window + self.horizon - 1, s1)
        test_idx = range(s1, n)
        train_data = self.batchify(train_idx, data)
        with open(os.path.join(self.processed_folder, 'w_' + str(self.window) + '_h_' + str(self.horizon) + '_train_data.json'), 'w') as f:
            json.dump(train_data, f)
        test_data = self.batchify(test_idx, data)
        with open(os.path.join(self.processed_folder, 'w_' + str(self.window) + '_h_' + str(self.horizon) + '_test_data.json'), 'w') as f:
            json.dump(test_data, f)

    def batchify(self, idx_set, data):
        n = len(idx_set)
        X = []
        Y = []
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X.append(data[start:end, :].tolist())
            Y.append(data[idx_set[i], :].tolist())
        return {'x': X, 'y': Y}


