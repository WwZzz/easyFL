from typing import Any
from torch.utils.data import Dataset
import os
import torchvision.datasets.utils
import pandas as pd
from .utils import map_column
import scipy.io as scio

class RatingPredictDataset(Dataset):
    urls = {}
    filenames = {}
    def __init__(self, root:str, download:bool=True, transform:Any=None, split:str=''):
        if split not in self.urls.keys(): raise ValueError("{} is not in list {}".format(split, list(self.urls.keys())))
        self.root = root
        self.enum_data = []
        self.unenum_data = []
        self.transform = transform
        self.split = split
        self.num_users = 0
        self.num_items = 0
        if download:
            self.download()
        self.load_data()

    def get_unenum_data(self, key=None):
        if key is not None:
            if key not in self.unenum_data: return {}
            return {key: getattr(self, key)}
        return {k:getattr(self, k) for k in self.unenum_data}

    def download(self):
        filename = os.path.join(self.root, self.filenames[self.split])
        urls = self.urls[self.split]
        if not os.path.exists(filename):
            for url in urls:
                if url.endswith('.zip' or '.tar.gz'):
                    torchvision.datasets.utils.download_and_extract_archive(url, download_root=self.root, remove_finished=True)
                else:
                    torchvision.datasets.utils.download_url(url, root=self.root)

    def __len__(self):
        if len(self.enum_data)==0:return 0
        return len(getattr(self, self.enum_data[0]))

    def __getitem__(self, item):
        data = {attr: getattr(self, attr)[item] for attr in self.enum_data}
        data.update({attr: getattr(self, attr) for attr in self.unenum_data})
        data.update({'__enum__':self.enum_data})
        return data

    def load_data(self):
        """load original data and set self.enum_data and self.unenum_data"""
        return NotImplementedError

class MovieLens(RatingPredictDataset):
    urls = {
        '100k': ["https://files.grouplens.org/datasets/movielens/ml-100k.zip"],
        '1m': ["https://files.grouplens.org/datasets/movielens/ml-1m.zip"],
        '10m': ["https://files.grouplens.org/datasets/movielens/ml-10m.zip"],
        '20m': ["https://files.grouplens.org/datasets/movielens/ml-20m.zip"],
        '25m': ["https://files.grouplens.org/datasets/movielens/ml-25m.zip"],
        'latest-small': ["https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"],
        'latest-full': ["https://files.grouplens.org/datasets/movielens/ml-latest.zip"],
    }
    filenames = {
        '100k': "ml-100k",
        '1m': "ml-1m",
        '10m': "ml-10m",
        '20m': "ml-20m",
        '25m': "ml-25m",
        'latest-small': "ml-latest-small",
        'latest-full': "ml-latest",
    }
    def __init__(self, root:str, split:str='100k', download:bool=True, train:bool=True, min_val:int=10, max_val:int=10e8):
        self.train = train
        self.min_val = min_val
        self.max_val = max_val
        super(MovieLens, self).__init__(root=root, split=split, download=download)


    def load_data(self):
        if self.split == '100k':
            rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
            file = 'u.data'
            data = pd.read_csv(os.path.join(self.root, self.filenames[self.split], file), sep='\t', names=rnames).drop(columns=['timestamp']).astype(int)
        else:
            rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
            data = pd.read_table(os.path.join(self.root, self.filenames[self.split],'ratings.dat'), sep='::', header=None, names=rnames, engine='python').drop(columns=['timestamp']).astype(int)
        # filter the data according to the numbers of samples of users
        data = data.groupby('user_id').filter(lambda x: len(x)>self.min_val and len(x)<self.max_val)
        data['user_id'] = map_column(data, 'user_id')
        data['movie_id'] = map_column(data, 'movie_id')
        self.num_users = len(data['user_id'].unique())
        self.num_items = len(data['movie_id'].unique())
        # split the training part and the testing part
        if self.train:
            data = data.groupby('user_id').apply(lambda x: x.head(int(len(x)*0.8)))
        else:
            data = data.groupby('user_id').apply(lambda x: x.tail(len(x)-int(len(x)*0.8)))
        data = data.rename(columns = {'movie_id':'item_id'})
        enum_data = ['user_id', 'item_id', 'rating']
        self.enum_data = enum_data
        for k in self.enum_data:
            setattr(self, k, data[k].to_numpy())
        return

class Ciao(RatingPredictDataset):
    urls = {
        'train':["https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip"],
        'val':["https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip"]
    }
    filenames = {
        'train':"ciao",
        'val':"ciao"
    }
    def __init__(self, root:str, split:str='train', download:bool=True, min_val:int=0, max_val:int=10e8):
        self.min_val = min_val
        self.max_val = max_val
        super(Ciao, self).__init__(root=root, split=split, download=download)

    def load_data(self):
        # read data into memory
        records = scio.loadmat(os.path.join(self.root, 'ciao', 'rating.mat'))['rating']
        data = pd.DataFrame(records, columns=['user_id', 'item_id', 'item_category', 'rating', 'rating_conf'])
        data = data.groupby('user_id').filter(lambda x: len(x) > self.min_val and len(x) < self.max_val)
        original_users = data['user_id'].unique()
        links = pd.DataFrame(scio.loadmat(os.path.join(self.root, 'ciao', 'trustnetwork.mat'))['trustnetwork'], columns=['trustor', 'trustee'])
        links = links[links['trustor'].isin(original_users) & links['trustee'].isin(original_users)]
        map_dict ={k:v for k,v in zip(original_users, list(range(len(original_users))))}
        links['trustor'] = map_column(links, 'trustor', [map_dict[k] for k in links['trustor'].unique()])
        links['trustee'] = map_column(links, 'trustee', [map_dict[k] for k in links['trustee'].unique()])
        data['user_id'] = map_column(data, 'user_id')
        data['item_id'] = map_column(data, 'item_id')
        self.num_users = len(data['user_id'].unique())
        self.num_items = len(data['item_id'].unique())
        self.enum_data = ['user_id', 'item_id', 'rating', 'item_category',  'rating_conf']
        for k in self.enum_data: setattr(self, k, data[k].to_numpy())
        self.unenum_data = ['social_link']
        setattr(self, 'social_link', links.to_numpy())
        return

class Epinions(RatingPredictDataset):
    urls = {
        'train': ["https://www.cse.msu.edu/~tangjili/datasetcode/epinions.zip"],
        'val': ["https://www.cse.msu.edu/~tangjili/datasetcode/epinions.zip"]
    }
    filenames = {
        'train': "epinions",
        'val': "epinions"
    }

    def __init__(self, root: str, split: str = 'train', download: bool = True, min_val: int = 0, max_val: int = 10e8):
        self.min_val = min_val
        self.max_val = max_val
        super(Epinions, self).__init__(root=root, split=split, download=download)

    def load_data(self):
        # read data into memory
        records = scio.loadmat(os.path.join(self.root, 'epinions', 'rating.mat'))['rating']
        data = pd.DataFrame(records, columns=['user_id', 'item_id', 'item_category', 'rating'])
        data = data.groupby('user_id').filter(lambda x: len(x) > self.min_val and len(x) < self.max_val)
        original_users = data['user_id'].unique()
        links = pd.DataFrame(scio.loadmat(os.path.join(self.root, 'epinions', 'trustnetwork.mat'))['trustnetwork'],
                             columns=['trustor', 'trustee'])
        links = links[links['trustor'].isin(original_users) & links['trustee'].isin(original_users)]
        map_dict = {k: v for k, v in zip(original_users, list(range(len(original_users))))}
        links['trustor'] = map_column(links, 'trustor', [map_dict[k] for k in links['trustor'].unique()])
        links['trustee'] = map_column(links, 'trustee', [map_dict[k] for k in links['trustee'].unique()])
        data['user_id'] = map_column(data, 'user_id')
        data['item_id'] = map_column(data, 'item_id')
        self.num_users = len(data['user_id'].unique())
        self.num_items = len(data['item_id'].unique())
        self.enum_data = ['user_id', 'item_id', 'rating', 'item_category']
        for k in self.enum_data: setattr(self, k, data[k].to_numpy())
        self.unenum_data = ['social_link']
        setattr(self, 'social_link', links.to_numpy())
        return