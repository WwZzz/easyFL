import torch_geometric.data
from flgo.benchmark.toolkits import BasicTaskGenerator, BasicTaskCalculator, BasicTaskPipe
import os.path
import urllib
import torch
from torch.utils.data import DataLoader
import os
import zipfile
from urllib.request import urlopen
import numpy as np
import flgo
import scipy.io as scio
import pandas as pd
try:
    import ujson as json
except:
    import json

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

class TaskGenerator(BasicTaskGenerator):
    _DATASET_INFO = {
        'url':"https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip",
        'num_users':7375,
        'num_items':99746,
        'num_ratings':278483,
        'range_rating':list(range(1,6)),
        'num_clinks': 111781,
        'link_type':'trust',
        'zipfile': 'ciao.zip',
    }

    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'CIAO'), scale = 1.0, order='max', min_items = 3):
        super(TaskGenerator, self).__init__(benchmark='social_splitted_ciao', rawdata_path=rawdata_path)
        self.scene='vertical'
        for key,value in self._DATASET_INFO.items(): setattr(self, key, value)
        self.scale = scale
        self.order = order
        self.min_items = min_items
        self.num_users = int(self.num_users * scale)

    def partition(self, *args, **kwargs):
        # sort and filter users
        all_users = self.ratings['userID'].unique()
        self.num_users = int(len(all_users)*self.scale)
        tmp = self.ratings['userID'].value_counts()
        tmp = tmp[tmp.values>=self.min_items]
        sorted_users = list(tmp.index)
        # user_size = tmp.values
        self.num_users = min(len(sorted_users), self.num_users)
        if self.order == 'max':
            users = sorted_users[:self.num_users]
        elif self.order == 'min':
            sorted_users.reverse()
            users = sorted_users[:self.num_users]
        else:
            users = list(np.random.choice(sorted_users, self.num_users, replace=False))
        # collect items according to the users
        ratings = self.ratings[self.ratings['userID'].isin(users)]
        items = ratings['movieID'].unique()
        links = self.links[(self.links['trustorID'].isin(users)) & (self.links['trusteeID'].isin(users))]
        user_map = {uid:uidx for uidx,uid in enumerate(users)}
        item_map = {vid:vidx for vidx,vid in enumerate(items)}
        ratings['userID'] = ratings['userID'].map(user_map)
        ratings['movieID'] = ratings['movieID'].map(item_map)
        links['trustorID'] = links['trustorID'].map(user_map)
        links['trusteeID'] = links['trusteeID'].map(user_map)
        self.users = list(range(len(users)))
        self.items = list(range(len(items)))
        self.ratings = ratings
        self.links = links

    def get_task_name(self):
        return 'social_splitted_ciao_u{}_o{}_m{}'.format(self.num_users, self.order,self.min_items)

    def load_data(self, *args, **kwargs):
        # download zip file
        zippath = os.path.join(self.rawdata_path, self.zipfile)
        if not os.path.exists(zippath):
            if not os.path.exists(self.rawdata_path):
                os.makedirs(self.rawdata_path)
            _ = download_from_url(self.url, zippath)
        # extract zip
        if not os.path.exists(os.path.join(self.rawdata_path, 'ciao')): _ = extract_from_zip(zippath, self.rawdata_path)
        # read data into memory
        ratings = scio.loadmat(os.path.join(self.rawdata_path, 'ciao', 'rating.mat'))['rating'][:,[0,1,3]]
        ratings = ratings.T
        rating_dict = {'userID':ratings[0], 'movieID':ratings[1], 'Rating':ratings[2]}
        self.ratings = pd.DataFrame(rating_dict)
        links = scio.loadmat(os.path.join(self.rawdata_path, 'ciao', 'trustnetwork.mat'))['trustnetwork']
        links = links.T
        link_dict = {'trustorID':links[0],'trusteeID':links[1]}
        self.links = pd.DataFrame(link_dict)
        return

class TaskPipe(BasicTaskPipe):
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.user_ids = torch.LongTensor([d[0] for d in data])
            self.movie_ids = torch.LongTensor([d[1] for d in data])
            self.ratings = torch.FloatTensor([d[2] for d in data])

        def __getitem__(self, item):
            return self.user_ids[item], self.movie_ids[item], self.ratings[item]

        def __len__(self):
            return len(self.user_ids)

    def save_task(self, generator):
        party_names = ['Advertiser', 'MeidaCompany']
        feddata = {'party_names': party_names, 'rawdata_path': generator.rawdata_path}
        feddata['Advertiser'] = {
            'data':{
                'userID':generator.users,
                'itemID':generator.items,
                'ratings': generator.ratings.values.tolist(),
                'with_label':True
            }
        }
        feddata['MeidaCompany'] = {
            'data': {
                'userID': generator.users,
                'links': generator.links.values.tolist(),
                'with_label':False
            }
        }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)

    def load_data(self, running_time_option, *args, **kwargs):
        task_data = {}
        for pid, party_name in enumerate(self.feddata['party_names']):
            pdata = self.feddata[party_name]['data']
            if party_name=='Advertiser':
                ratings = pdata['ratings']
                users = pdata['userID']
                tmp = pd.DataFrame(ratings)
                train_data = []
                test_data = []
                val_data = []
                val_ratio = running_time_option['train_holdout']
                for uid in users:
                    udata = tmp[tmp[0]==uid].values.tolist()
                    local_data_size = max(int(len(udata)*0.9), 1)
                    ulocal = udata[:local_data_size]
                    utest = udata[local_data_size:]
                    train_size = int(len(ulocal)*(1-val_ratio))
                    utrain = ulocal[:train_size]
                    uvalid = ulocal[train_size:]
                    train_data.extend(utrain)
                    val_data.extend(uvalid)
                    test_data.extend(utest)
                train_data = self.TaskDataset(train_data)
                val_data = self.TaskDataset(val_data) if len(val_data)>0 else None
                test_data = self.TaskDataset(test_data)
                task_data[party_name] = {'userID': pdata['userID'], 'itemID':pdata['itemID'], 'train':train_data, 'val':val_data, 'test':test_data}
            else:
                social_links = pdata['links']
                edge_index = torch.tensor([[l[0] for l in social_links], [l[1] for l in social_links]], dtype=torch.int64)
                social_data = torch_geometric.data.Data(edge_index=edge_index)
                task_data[party_name] = {'userID':pdata['userID'],'social':social_data}
        return task_data

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super().__init__(device, optimizer_name)
        self.criterion = torch.nn.MSELoss()
        self.DataLoader = DataLoader

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
#
    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def compute_loss(self, model, data, *args, **kwargs):
        tdata = self.to_device(data)
        model = model.to(self.device)
        output = model(tdata)
        loss =  self.criterion(output, torch.tensor(tdata[-1], dtype=torch.float))
        return {'loss':loss}

    def test(self, model, dataset, batch_size=64, num_workers = 0, *args, **kwargs):
        model.to(self.device)
        with torch.no_grad():
            model.eval()
            if batch_size == -1: batch_size = len(dataset)
            data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
            mse_loss = 0
            mae_loss = 0
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.to_device(batch_data)
                outputs = model(batch_data)
                batch_mae = torch.abs(outputs-batch_data[-1]).sum().item()
                batch_mse = torch.nn.MSELoss()(outputs, batch_data[-1]).item()
                mse_loss += batch_mse * len(batch_data[-1])
                mae_loss += batch_mae
            mae = mae_loss/len(dataset)
            rmse = torch.sqrt(torch.tensor(mse_loss/len(dataset)))
        return {'mae': mae, 'rmse': rmse.item()}