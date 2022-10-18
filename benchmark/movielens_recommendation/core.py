from benchmark.toolkits import BasicTaskGen, BasicTaskCalculator, BasicTaskPipe
import numpy as np
import os.path
import urllib
import torch
from torch.utils.data import DataLoader
import os
import zipfile
from urllib.request import urlopen
import numpy as np
import pandas as pd
import ujson

def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    if url: urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]

class TaskGen(BasicTaskGen):
    _DATASET_INFO = {
        '100k':{'num_clients':1000, 'num_items':1700, 'num_ratings':1e5, 'url': "https://files.grouplens.org/datasets/movielens/ml-100k.zip"},
        '1m':{'num_clients':6000, 'num_items':4000, 'num_ratings':1e6, 'url': "https://files.grouplens.org/datasets/movielens/ml-1m.zip"},
        '10m': {'num_clients': 72000, 'num_items': 10000, 'num_ratings': 10e6, 'url': "https://files.grouplens.org/datasets/movielens/ml-10m.zip"},
        '20m': {'num_clients': 138000, 'num_items': 27000, 'num_ratings': 20e6, 'url': "https://files.grouplens.org/datasets/movielens/ml-20m.zip"},
        '25m': {'num_clients': 162000, 'num_items': 62000, 'num_ratings': 25e6, 'url': "https://files.grouplens.org/datasets/movielens/ml-25m.zip"},
        'latest-small':{'num_clients':600, 'num_items':9000, 'num_ratings':1e5, 'url': "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"},
        'latest-full': {'num_clients': 280000, 'num_items': 58000, 'num_ratings': 27e6, 'url':"https://files.grouplens.org/datasets/movielens/ml-latest.zip"},
    }
    def __init__(self, version='100k', dist_id = 0, num_clients = 0, skewness = 0.5, minvol=50, rawdata_path ='./benchmark/RAW_DATA/MOVIELENS/', seed=0):
        super(TaskGen, self).__init__(benchmark='movielens_recommendation',
                                      dist_id=5,
                                      skewness=1.0,
                                      rawdata_path=os.path.join(rawdata_path, version.upper()),
                                      seed=seed
                                      )
        # assert version in ['100k', '1m']
        assert version in ['100k','1m','10m','20m','25m','latest-small', 'latest-full']
        self.version = version
        self.zipname = self._DATASET_INFO[version]['url'].split('/')[-1]
        self.unzippath = os.path.join(self.rawdata_path, self.zipname.split('.')[0])
        self.url= self._DATASET_INFO[version]['url']
        self.num_clients = self._DATASET_INFO[version]['num_clients']
        self.num_items = self._DATASET_INFO[version]['num_items']
        self.num_ratings = self._DATASET_INFO[version]['num_ratings']
        self.cnames = self.get_client_names()
        self.taskname = self.get_taskname()
        self.save_task = TaskPipe.save_task
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)

    def run(self):
        if self._check_task_exist():
            print("Task Already Exists.")
            return
        print('-----------------------------------------------------')
        print('Loading...')
        self.train_datas, self.valid_datas, self.test_data = self.load_data()
        print('Done.')
        # save task infomation as .json file and the federated dataset
        print('-----------------------------------------------------')
        print('Saving data...')
        try:
            # create the directory of the task
            self.create_task_directories()
            self.save_task(self)
        except Exception as e:
            print(e)
            self._remove_task()
            print("Failed to saving splited dataset.")
        print('Done.')
        return

    def load_data(self, *args, **kwargs):
        # download zip file
        zippath = os.path.join(self.rawdata_path, self.zipname)
        if not os.path.exists(zippath):
            _ = download_from_url(self.url, zippath)
        # extract zip
        if not os.path.exists(self.unzippath):
            _ = extract_from_zip(zippath, self.rawdata_path)
        # read data into memory
        if self.version=='100k':
            rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
            self.ratings = pd.read_csv(os.path.join(self.unzippath, 'u.data'), sep='\t', names=rnames).drop(columns=['timestamp']).astype(int)
        if self.version=='1m':
            unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
            self.users = pd.read_table(os.path.join(self.unzippath, 'users.dat'), sep='::', header=None, names=unames, engine='python')
            mnames = ['movie_id', 'title', 'genres']
            self.movies = pd.read_table(os.path.join(self.unzippath, 'movies.dat'), sep='::', header=None, names=mnames, engine='python')
            rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
            self.ratings = pd.read_table(os.path.join(self.unzippath, 'ratings.dat'), sep='::', header=None, names=rnames, engine='python').drop(columns=['timestamp']).astype(int)

        local_datas = [np.array(self.ratings[self.ratings['user_id']==(uid+1)]).tolist() for uid in range(self.num_clients)]
        train_datas = []
        valid_datas = []
        test_data = []
        for uid in range(self.num_clients):
            udata = local_datas[uid]
            np.random.shuffle(udata)
            k1 = int(len(udata)*0.8)
            k2 = int(len(udata)*0.9)
            train_datas.append(udata[:k1])
            valid_datas.append(udata[k1:k2])
            test_data.extend(udata[k2:])
        return train_datas, valid_datas, test_data

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super().__init__(device, optimizer_name)
        self.criterion = torch.nn.L1Loss()
        self.DataLoader = DataLoader

    def train_one_step(self, model, data):
        users, movies, ratings = self.data_to_device(data)
        outputs = model(users, movies)
        loss = self.criterion(outputs, ratings)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            outputs = model(batch_data[0], batch_data[1])
            batch_mean_loss = self.criterion(outputs, batch_data[-1]).item()
            total_loss += batch_mean_loss * len(batch_data[-1])
        return {'loss':total_loss/len(dataset)}

    def data_to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class TaskPipe(BasicTaskPipe):
    class TaskDataset(torch.utils.data.Dataset):
        _NUM_ITEMS = 0
        _NUM_CLIENTS = 0
        def __init__(self, data):
            self.user_ids = torch.LongTensor([d[0] for d in data])
            self.movie_ids = torch.LongTensor([d[1] for d in data])
            self.ratings = torch.FloatTensor([d[2] for d in data])

        def __getitem__(self, item):
            return self.user_ids[item], self.movie_ids[item], self.ratings[item]

        def __len__(self):
            return len(self.user_ids)

    @classmethod
    def load_task(cls, task_path, cross_validation=False, **kwargs):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        cnames = feddata['client_names']
        cls.TaskDataset._NUM_ITEMS = feddata['num_items']
        cls.TaskDataset._NUM_CLIENTS = feddata['num_clients']
        test_data = cls.TaskDataset(feddata['dtest'])
        train_datas = [cls.TaskDataset(feddata[cnames[cid]]['dtrain']) for cid in range(len(cnames))]
        valid_datas = [cls.TaskDataset(feddata[cnames[cid]]['dvalid']) for cid in range(len(cnames))]
        return train_datas, valid_datas, test_data, cnames

    @classmethod
    def save_task(cls, generator):
        feddata = {
            'store': 'X',
            'client_names': generator.cnames,
            'dtest': generator.test_data,
            'num_items': generator.num_items,
            'num_clients': generator.num_clients,
            'num_ratings': generator.num_ratings
        }
        for cid in range(len(generator.cnames)):
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_datas[cid],
                'dvalid': generator.valid_datas[cid]
            }
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return