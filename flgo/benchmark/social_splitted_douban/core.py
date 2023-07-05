import h5py
import torch_geometric.data
from flgo.benchmark.toolkits import BasicTaskGenerator, BasicTaskCalculator, BasicTaskPipe
import os.path
import torch
from torch.utils.data import DataLoader
import os
import zipfile
from urllib.request import urlopen
import urllib.request
import numpy as np
import flgo
import pandas as pd
import scipy.sparse as sp
try:
    import ujson as json
except:
    import json

def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url: urllib.request.urlretrieve(url, filepath)
    return filepath

class TaskGenerator(BasicTaskGenerator):
    _DATASET_INFO = {
        'url':"https://raw.githubusercontent.com/fmonti/mgcnn/master/Data/douban/training_test_dataset.mat",
        'link_type':'trust',
        'file': 'training_test_dataset.mat',
    }
    #
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'DOUBAN')):
        super(TaskGenerator, self).__init__(benchmark='social_splitted_douban', rawdata_path=rawdata_path)
        self.scene='vertical'
        for key,value in self._DATASET_INFO.items(): setattr(self, key, value)
        self.num_users = 0

    def partition(self, *args, **kwargs):
        pass

    def get_task_name(self):
        return 'social_splitted_douban'

    def load_data(self, *args, **kwargs):
        # download zip file
        filepath = os.path.join(self.rawdata_path, self.file)
        if not os.path.exists(filepath):
            if not os.path.exists(self.rawdata_path):
                os.makedirs(self.rawdata_path)
            _ = download_from_url(self.url, filepath)
        # read data into memory
        db = h5py.File(filepath, 'r')
        fields = ['M', 'Otraining', 'Otest', 'W_users']
        outputs = []
        for name_field in fields:
            ds = db[name_field]
            try:
                if 'ir' in ds.keys():
                    data = np.asarray(ds['data'])
                    ir = np.asarray(ds['ir'])
                    jc = np.asarray(ds['jc'])
                    out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
            except AttributeError:
                # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
                out = np.asarray(ds).astype(np.float32).T
            outputs.append(out)
        M, Otraining, Otest, Wrow = outputs
        self.num_users=3000
        self.num_items = 3000
        train_idxs = np.where(Otraining>0)
        train_ratings = M[train_idxs]
        test_idxs = np.where(Otest>0)
        test_ratings = M[test_idxs]
        train_dict = {'userID':train_idxs[0].tolist(), 'movieID':train_idxs[1].tolist(), 'Rating':train_ratings.tolist()}
        test_dict = {'userID':test_idxs[0].tolist(), 'movieID':test_idxs[1].tolist(), 'Rating':test_ratings.tolist()}
        self.ratings_train = pd.DataFrame(train_dict)
        self.ratings_test = pd.DataFrame(test_dict)
        link_idxs = np.where(Wrow>0)
        link_values = Wrow[link_idxs]
        link_dict = {'trustorID':link_idxs[0].tolist(), 'trusteeID':link_idxs[1].tolist(), 'Value':link_values}
        self.links = pd.DataFrame(link_dict)
        self.users = list(range(self.num_users))
        self.items = list(range(self.num_items))
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
                'ratings_train': generator.ratings_train.values.tolist(),
                'ratings_test': generator.ratings_test.values.tolist(),
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
                users = pdata['userID']
                test_data = pdata['ratings_test']
                ratings_train = pdata['ratings_train']
                tmp = pd.DataFrame(ratings_train)
                train_data = []
                val_data = []
                val_ratio = running_time_option['train_holdout']
                for uid in users:
                    udata = tmp[tmp[0]==uid].values.tolist()
                    train_size = max(int(len(udata)*(1-val_ratio)), 1)
                    utrain = udata[:train_size]
                    uvalid = udata[train_size:]
                    train_data.extend(utrain)
                    val_data.extend(uvalid)
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