import zipfile

import numpy
import ujson as json
import os
import urllib
import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import Dataset


def download_from_url(url=None, filepath='.'):
    """Download dataset from url to filepath."""
    if url:
        urllib.request.urlretrieve(url, filepath)
    return filepath


def extract_from_zip(src_file, target_directory):
    """Unzip the .zip file (src_path) to target_path"""
    file = zipfile.ZipFile(src_file)
    file.extractall(target_directory)
    file.close()


def preprocessing(data, names, column_type, truelabel, default_dic=None):
    res = [[] for i in range(len(data))]
    for index, type in enumerate(column_type):
        id = names[index]
        if type=='continuous':
            max_val = int(data[id].max())
            min_val = int(data[id].min())
            for i in range(len(data)):
                value = int(data[id][i])
                # data.loc[i, id] = (value - min_val) / (max_val - min_val)
                res[i].append((value - min_val) / (max_val - min_val))
        elif id == 'label':
            for i in range(len(data)):
                value = data[id][i]
                if value == truelabel:
                    res[i].append(1)
                else:
                    res[i].append(0)
        else:
            if default_dic is None:
                keys = data[id].unique()
                key_dic = {}
                for i, k in enumerate(keys):
                    key_dic[k]=i
            else:
                key_dic = default_dic[index]
            # onehot
            for i in range(len(data)):
                value = data[id][i]
                # data.loc[i, id] = '0' * key_dic[value] + '1' + '0' * (len(key_dic) - key_dic[value] - 1)
                temp = [0] * key_dic[value] + [1] + [0] * (len(key_dic) - key_dic[value] - 1)
                res[i] = res[i] + temp
    return res


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


class Adult(BuiltinClassDataset):
    """
    在download中将原始数据进行分片并划分为训练集和测试集，分别存储到train_data.json和test_data.json两个文件中
    文件路径默认设置为 self.processed_folder/XX_data.json
    """
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marial-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    column_type = ['continuous', 8, 'continuous', 16, 'continuous', 7, 14, 6, 5, 2, 'continuous', 'continuous',
                   'continuous', 42, 2]
    true_label = '<=50k'

    def __init__(self, root, train=True):
        self.url = {
            'train': "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            'test': "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        }
        self.train = train
        self.file = 'adult_'
        self.root = root
        self.default_dic = [{} for i in range(len(Adult.names))]
        super().__init__(root, train)

    def get_dic(self, paths):
        data1 = pd.read_csv(paths[0], header=None, names=Adult.names, skipinitialspace=True)
        data2 = pd.read_csv(paths[1], header=None, names=Adult.names, skipinitialspace=True)
        data2.drop(index=[0], inplace=True)
        data2 = data2.reset_index(drop=True)
        for index, val in enumerate(Adult.column_type):
            if val == 'continuous':
                continue
            name = Adult.names[index]
            temp1 = data1[name].unique().tolist()
            temp2 = data2[name].unique().tolist()
            keys = list(set(temp1+temp2))
            self.default_dic[index] = {k: i for i, k in enumerate(keys)}


    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        train_src_path = os.path.join(self.raw_folder, 'adult.data')
        test_src_path = os.path.join(self.raw_folder, 'adult.test')
        if not os.path.exists(train_src_path):
            download_from_url(self.url['train'], train_src_path)
        if not os.path.exists(test_src_path):
            download_from_url(self.url['test'], test_src_path)
        target_file = 'raw_data\\adult.data' if self.train is True else 'raw_data\\adult.test'
        paths = [os.path.join(self.root, 'raw_data\\adult.data'), os.path.join(self.root, 'raw_data\\adult.test')]
        self.get_dic(paths)
        self.to_json(os.path.join(self.root, target_file))

    def to_json(self, path):
        raw_data = pd.read_csv(path, header=None, names=Adult.names, skipinitialspace=True)
        if self.train is False:
            raw_data.drop(index=[0], inplace=True)
            raw_data = raw_data.reset_index(drop=True)
        processed_data = preprocessing(raw_data, Adult.names, Adult.column_type, Adult.true_label, self.default_dic)
        processed_data = np.array(processed_data)
        dits = {'x':processed_data[:, :-1].tolist(), 'y':processed_data[:, -1].tolist()}
        file_name = 'adult_train_data.json' if self.train is True else 'adult_test_data.json'
        with open(os.path.join(self.processed_folder, file_name), 'w') as f:
            json.dump(dits, f)





class BankMarketing(BuiltinClassDataset):
    """
    在download中将原始数据进行分片并划分为训练集和测试集，分别存储到train_data.json和test_data.json两个文件中
    文件路径默认设置为 self.processed_folder/XX_data.json
    """
    names = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
             "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "label"]
    column_type = ['continuous', 12, 3, 4, 2, 'continuous', 2, 2, 3, 'continuous', 12, 'continuous',
                   'continuous', 'continuous', 'continuous', 4, 2]
    true_label = 'yes'

    def __init__(self, root, train=True):
        self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
        self.train = train
        self.file = 'bankmarketing_'
        self.root = root
        super().__init__(root, train)


    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        src_path = os.path.join(self.raw_folder, 'bank.zip')
        if not os.path.exists(src_path):
            download_from_url(self.url, src_path)
        target_file = os.path.join(self.root, 'raw_data\\bank-full.csv')
        if not os.path.exists(target_file):
            extract_from_zip(src_path, os.path.join(self.root, 'raw_data'))
        self.to_json(target_file)

    def to_json(self, path):
        raw_data = pd.read_csv(path, sep=';', header=0, names=BankMarketing.names)
        raw_data.loc[raw_data['pdays'] == -1, ['pdays']] = 99999999
        processed_data = preprocessing(raw_data, BankMarketing.names, BankMarketing.column_type, BankMarketing.true_label)
        processed_data = np.array(processed_data)
        num = len(processed_data)
        index = [i for i in range(num)]
        train_num = int(num * 0.8)
        train = random.sample(index, train_num)
        test = [i for i in range(num) if i not in train]
        train_dic = {'x':processed_data[train, :-1].tolist(), 'y':processed_data[train, -1].tolist()}
        test_dic = {'x':processed_data[test, :-1].tolist(), 'y':processed_data[test, -1].tolist()}
        with open(os.path.join(self.processed_folder, 'bankmarketing_train_data.json'), 'w') as f:
            json.dump(train_dic, f)
        with open(os.path.join(self.processed_folder, 'bankmarketing_test_data.json'), 'w') as f:
            json.dump(test_dic, f)



# import flgo
# pat = os.path.join(flgo.benchmark.data_root, 'heart_disease_classification')

class HeartDisease(BuiltinClassDataset):
    """
    在download中将原始数据进行分片并划分为训练集和测试集，分别存储到train_data.json和test_data.json两个文件中
    文件路径默认设置为 self.processed_folder/XX_data.json
    """
    names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
             "oldpeak", "slope", "ca", "thal", "label"]
    column_type = ['continuous', 2, 4, 'continuous', 'continuous', 2, 3, 'continuous', 2, 'continuous', 3, 3,
                   3, 4]
    true_label = 1
    def __init__(self, root, train=True):
        self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        self.train = train
        self.file = 'heart_disease_'
        self.root = root
        super().__init__(root, train)


    def download(self):
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        src_path = os.path.join(self.raw_folder, 'processed.cleveland.data')
        if not os.path.exists(src_path):
            download_from_url(self.url, src_path)
        self.to_json(src_path)

    def to_json(self, path):
        raw_data = pd.read_csv(path, header=0, names=HeartDisease.names)
        raw_data.loc[raw_data['label'] > 0, ['label']] = 1
        processed_data = preprocessing(raw_data, HeartDisease.names, HeartDisease.column_type, HeartDisease.true_label)
        processed_data = np.array(processed_data)
        num = len(processed_data)
        index = [i for i in range(num)]
        train_num = int(num * 0.8)
        train = random.sample(index, train_num)
        test = [i for i in range(num) if i not in train]
        train_dic = {'x':processed_data[train, :-1].tolist(), 'y':processed_data[train, -1].tolist()}
        test_dic = {'x':processed_data[test, :-1].tolist(), 'y':processed_data[test, -1].tolist()}
        with open(os.path.join(self.processed_folder, 'heart_disease_train_data.json'), 'w') as f:
            json.dump(train_dic, f)
        with open(os.path.join(self.processed_folder, 'heart_disease_test_data.json'), 'w') as f:
            json.dump(test_dic, f)


# a = HeartDisease(pat)