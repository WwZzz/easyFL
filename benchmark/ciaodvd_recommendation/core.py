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
        'url':"https://guoguibing.github.io/librec/datasets/CiaoDVD.zip",
        'num_clients':7375,
        'num_items':99746,
        'num_ratings':278483,
        'range_rating':list(range(1,6)),
        'num_clinks': 111781,
        'link_type':'trust',
        'zipfile': 'CiaoDVD.zip',
    }
    def __init__(self, dist_id = 0, num_clients = 0, skewness = 0.5, minvol=50, rawdata_path ='./benchmark/RAW_DATA/CIAODVD/', seed=0):
        super(TaskGen, self).__init__(benchmark='ciaodvd_recommendation',
                                      dist_id=5,
                                      skewness=1.0,
                                      rawdata_path=rawdata_path,
                                      seed=seed
                                      )
        for key,value in self._DATASET_INFO.items():
            setattr(self, key, value)
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
        self.train_datas, self.valid_datas, self.test_data, self.trustees, self.trustors, self.coldstart_clients = self.load_data()
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
        zippath = os.path.join(self.rawdata_path, self.zipfile)
        if not os.path.exists(zippath):
            if not os.path.exists(self.rawdata_path):
                os.makedirs(self.rawdata_path)
            _ = download_from_url(self.url, zippath)
        # extract zip
        if not os.path.exists(os.path.join(self.rawdata_path, 'readme.txt')):
            _ = extract_from_zip(zippath, self.rawdata_path)
        # read data into memory
        ratings = pd.read_table(os.path.join(self.rawdata_path, 'movie-ratings.txt'), sep=',',names=['userID', 'movieID', 'genreID', 'reviewID', 'movieRating', 'date']).drop(columns=['genreID','reviewID','date']).astype(int)
        links = pd.read_table(os.path.join(self.rawdata_path, 'trusts.txt'), sep=',',names=['trustorID', 'trusteeID', 'trustValue']).astype(int)
        local_ratings = [np.array(ratings[ratings['userID']==(uid+1)]).tolist() for uid in range(self.num_clients)]
        local_trustees = [np.array(links[links['trustorID']==(uid+1)]['trusteeID']).tolist() for uid in range(self.num_clients)]
        local_trustors = [np.array(links[links['trusteeID']==(uid+1)]['trustorID']).tolist() for uid in range(self.num_clients)]
        # local holdout
        train_datas = [[] for _ in range(self.num_clients)]
        valid_datas = [[] for _ in range(self.num_clients)]
        test_data = []
        coldstart_clients = []
        for uid in range(self.num_clients):
            urs = local_ratings[uid]
            np.random.shuffle(urs)
            ktest = int(len(urs) * 0.9)
            if len(urs[:ktest])<3:
                test_data.extend(urs)
                coldstart_clients.append(uid+1)
            else:
                test_data.extend(urs[ktest:])
                kvalid = int(len(urs)*0.75)
                if kvalid==ktest: kvalid-=1
                train_datas[uid].extend(urs[:kvalid])
                valid_datas[uid].extend(urs[kvalid:ktest])
        return train_datas, valid_datas, test_data, local_trustees, local_trustors, coldstart_clients

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
        _SOCIAL_MATRIX = None
        _COLDSTART_UIDS = []
        def __init__(self, data):
            self.user_ids = torch.LongTensor([d[0] for d in data])
            self.movie_ids = torch.LongTensor([d[1] for d in data])
            self.ratings = torch.FloatTensor([d[2] for d in data])

        def __getitem__(self, item):
            return self.user_ids[item], self.movie_ids[item], self.ratings[item]

        def __len__(self):
            return len(self.user_ids)

        def get_trustees(self):
            if self._SOCIAL_MATRIX is None: return []
            return torch.LongTensor(self._SOCIAL_MATRIX[int(self.user_ids[0]) - 1, :])

        def get_trustors(self):
            if self._SOCIAL_MATRIX is None: return []
            return torch.LongTensor(self._SOCIAL_MATRIX[:, int(self.user_ids[0]) - 1])

        def get_social_matrix(self):
            return torch.LongTensor(self._SOCIAL_MATRIX)

        def get_coldstart_clients(self):
            return torch.LongTensor(self._COLDSTART_UIDS)

    @classmethod
    def load_task(cls, task_path, cross_validation=False, **kwargs):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        cnames = feddata['client_names']
        trustors = feddata['trustors']
        trustees = feddata['trustees']
        coldstart_clients = feddata['coldstart_clients']
        social_matrix = np.zeros((len(cnames), len(cnames)))
        for uid in range(len(cnames)):
            for tor in trustors[uid]:
                if tor>7375: continue
                social_matrix[tor-1][uid]=1
            for tee in trustees[uid]:
                if tee>7375: continue
                social_matrix[uid][tee-1]=1
        cls.TaskDataset._SOCIAL_MATRIX = social_matrix
        cls.COLD_START_UIDS = coldstart_clients
        test_data = cls.TaskDataset(feddata['dtest'])
        train_cnames = [name for uid, name in zip(list(range(len(cnames))), cnames) if uid + 1 not in coldstart_clients]
        train_datas = [cls.TaskDataset(feddata[train_cnames[cid]]['dtrain']) for cid in range(len(train_cnames))]
        valid_datas = [cls.TaskDataset(feddata[train_cnames[cid]]['dvalid']) for cid in range(len(train_cnames))]
        return train_datas, valid_datas, test_data, train_cnames

    @classmethod
    def save_task(cls, generator):
        feddata = {
            'store': 'X',
            'client_names': generator.cnames,
            'dtest': generator.test_data,
            'trustors': generator.trustors,
            'trustees': generator.trustees,
            'coldstart_clients': generator.coldstart_clients,
        }
        for cid in range(len(generator.cnames)):
            if (cid+1) in generator.coldstart_clients:
                continue
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_datas[cid],
                'dvalid': generator.valid_datas[cid]
            }
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return