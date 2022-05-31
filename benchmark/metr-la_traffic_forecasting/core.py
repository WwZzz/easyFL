import os
import pickle
import tarfile
import urllib
import numpy as np
import torch
import json
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import dense_to_sparse

from benchmark.toolkits import BasicTaskGen, BasicTaskPipe, BasicTaskCalculator

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

class TaskGen(BasicTaskGen):
    def __init__(self, dist_id = 5, num_clients = 207, rawdata_path ='./benchmark/RAW_DATA/METR-LA&PEMS-BAY', skewness = 0, seed = 0, local_hld_rate=0.0):
        super(TaskGen, self).__init__(benchmark='metr-la_traffic_forecasting',
                                      dist_id=5,
                                      skewness=skewness,
                                      rawdata_path=rawdata_path,
                                      seed=seed,
                                      local_hld_rate=local_hld_rate
                                      )
        self.train_ratio = 1.0 - self.skewness
        self.num_nodes = 207
        self.num_clients = int(self.train_ratio*self.num_nodes)
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        self.save_task = TaskPipe.save_task

    def run(self):
        if self._check_task_exist():
            print("Task Already Exists.")
            return
        self.load_data()
        try:
            print('Saving data...',end='')
            self.create_task_directories()
            self.save_task(self)
        except Exception as e:
            print(e)
            self._remove_task()
            print("Failed to saving splited dataset.")
        print('done')

    def get_taskname(self):
        """Create task name and return it."""
        taskname = '_'.join([self.benchmark, 'cnum' +  str(self.num_clients), 'dist' + str(self.dist_id), 'ratio' + str(self.train_ratio).replace(" ", ""), 'seed'+str(self.seed)])
        return taskname

    def load_data(self):
        # download'
        if not os.path.exists(os.path.join(self.rawdata_path, 'data.tar.bz2')):
            if not os.path.exists(self.rawdata_path):
                os.mkdir(self.rawdata_path)
            print("Downloading Dataset...")
            url = 'https://zenodo.org/record/4521262/files/data.tar.bz2'
            urllib.request.urlretrieve(url, self.rawdata_path)
        if not os.path.exists(os.path.join(self.rawdata_path,'data','traffic')):
            print('Unzip Downloaded File...',end='')
            tarf = tarfile.open(os.path.join(self.rawdata_path, 'data.tar.bz2'))
            tarf.extractall(self.rawdata_path)
            print('done')
        self.data_path = os.path.join(self.rawdata_path,'data','traffic','data','METR-LA')
        self.adj_path = os.path.join(self.rawdata_path,'data','traffic','data','sensor_graph')
        self.adj_mx_name = 'adj_mx.pkl'
        _, _, adj_mx = load_pickle(os.path.join(self.adj_path, self.adj_mx_name))
        all_nodes = [i for i in range(self.num_nodes)]
        self.test_nodes = all_nodes
        self.train_nodes = np.random.choice(all_nodes, int(self.train_ratio*self.num_nodes), replace=False).tolist()
        self.train_nodes.sort()
        self.adj_mx = adj_mx.tolist()

class TaskPipe(BasicTaskPipe):
    class STNodeDataset(TensorDataset):
        __num_nodes = 0
        __adj = None
        __edge = {
            'edge_index':[],
            'edge_attr':[],
            'num_edges':0,
        }
        __mask = None
        __masked_adj = None
        __masked_edge = {
            'edge_index': [],
            'edge_attr': [],
            'num_edges': 0,
        }

        def __init__(self, *tensors, node_list=[]):
            super(TaskPipe.STNodeDataset, self).__init__(*tensors)
            self.node_id = node_list

        @classmethod
        def set_adj(cls, adj_mx):
            cls.__adj = adj_mx
            cls.__num_nodes = len(adj_mx)

        @classmethod
        def get_adj(cls):
            return cls.__adj

        @classmethod
        def get_masked_adj(cls):
            return cls.__masked_adj

        @classmethod
        def get_sparse_edge(cls):
            return cls.__edge

        @classmethod
        def get_masked_edge(cls):
            return cls.__masked_edge

        @classmethod
        def convert_adj_to_sparse(cls):
            if cls.__adj is not None:
                edge_index, edge_attr = dense_to_sparse(cls.__adj)
                cls.__edge['edge_index'] = edge_index
                cls.__edge['edge_attr'] = edge_attr
                cls.__edge['num_edges'] = len(edge_index[0])

        @classmethod
        def set_mask(cls, mask):
            cls.__mask = mask
            selected_nodes = [nid for nid in range(cls.__num_nodes) if mask[nid][0]]
            cls.__masked_adj = cls.__adj[selected_nodes,:][:,selected_nodes]
            edge_index, edge_attr = dense_to_sparse(cls.__masked_adj)
            cls.__masked_edge['edge_index'] = edge_index
            cls.__masked_edge['edge_attr'] = edge_attr
            cls.__masked_edge['num_edges'] = len(edge_index[0])

        @classmethod
        def get_mask(cls):
            return cls.__mask

    TaskDataset = STNodeDataset

    @classmethod
    def save_task(cls, generator):
        feddata = {
            'data_path': generator.data_path,
            'train_nodes': generator.train_nodes,
            'num_nodes': generator.num_nodes,
            'num_clients': generator.num_clients,
            'adj_mx': generator.adj_mx,
        }
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)

    @classmethod
    def load_task(cls, task_path):
        # read fedtask
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = json.load(inf)
        data_path = feddata['data_path']
        adj_mx = torch.Tensor(feddata['adj_mx'])
        num_nodes = feddata['num_nodes']
        train_nodes = feddata['train_nodes']
        num_clients = feddata['num_clients']
        eval_adj_mx = adj_mx
        raw_data = {}
        for name in ['train', 'val', 'test']:
            raw_data[name] = np.load(os.path.join(data_path, '{}.npz'.format(name)))
        FEATURE_START, FEATURE_END = 0, 1
        ATTR_START, ATTR_END = 1, 2
        train_features = raw_data['train']['x'][..., FEATURE_START:FEATURE_END]
        train_node_mask = torch.BoolTensor([False] * eval_adj_mx.shape[0])
        train_node_mask[train_nodes] = True
        train_node_mask = train_node_mask.unsqueeze(-1)
        train_features = train_features[:, :, train_nodes, :]
        # calculate mean and std of training features to normalize the features
        train_features = train_features.reshape(-1, train_features.shape[-1])
        feature_scaler = StandardScaler(mean=train_features.mean(axis=0), std=train_features.std(axis=0))
        attr_scaler = StandardScaler(mean=0, std=1)
        loaded_data = {'feature_scaler': feature_scaler, 'attr_scaler': attr_scaler}
        train_datas, valid_datas = [], []
        for name in ['train', 'val','test']:
            x = feature_scaler.transform(raw_data[name]['x'][..., FEATURE_START:FEATURE_END])
            y = feature_scaler.transform(raw_data[name]['y'][..., FEATURE_START:FEATURE_END])
            x_attr = attr_scaler.transform(raw_data[name]['x'][..., ATTR_START:ATTR_END])
            y_attr = attr_scaler.transform(raw_data[name]['y'][..., ATTR_START:ATTR_END])
            data = {}
            # if name is 'train':
            #     edge_index, edge_attr = train_edge_index, train_edge_attr
            # else:
            #     edge_index, edge_attr = eval_edge_index, eval_edge_attr
            data.update(
                x=torch.from_numpy(x).float(), y=torch.from_numpy(y).float(),
                x_attr=torch.from_numpy(x_attr).float(),
                y_attr=torch.from_numpy(y_attr).float(),
                # edge_index=edge_index, edge_attr=edge_attr
            )
            if name is 'train' and num_clients<len(train_nodes):
                data.update(selected=train_node_mask)
            loaded_data[name] = data

        client_names = []
        for cid in range(num_clients):
            node_id = train_nodes[cid]
            client_names.append('Client{}'.format(node_id))
            train_data = cls.TaskDataset(
                loaded_data['train']['x'][:,:,node_id:node_id+1,:],
                loaded_data['train']['y'][:, :, node_id:node_id + 1, :],
                loaded_data['train']['x_attr'][:, :, node_id:node_id + 1, :],
                loaded_data['train']['y_attr'][:, :, node_id:node_id + 1, :],
                node_list = [node_id],
            )
            valid_data = cls.TaskDataset(
                loaded_data['val']['x'][:,:,node_id:node_id+1,:],
                loaded_data['val']['y'][:, :, node_id:node_id + 1, :],
                loaded_data['val']['x_attr'][:, :, node_id:node_id + 1, :],
                loaded_data['val']['y_attr'][:, :, node_id:node_id + 1, :],
                node_list = [node_id],
            )
            train_datas.append(train_data)
            valid_datas.append(valid_data)
        test_data = cls.TaskDataset(
                loaded_data['test']['x'],
                loaded_data['test']['y'],
                loaded_data['test']['x_attr'],
                loaded_data['test']['y_attr'],
                node_list = [nid for nid in range(num_nodes)],
            )
        cls.TaskDataset.set_adj(adj_mx)
        cls.TaskDataset.convert_adj_to_sparse()
        cls.TaskDataset.set_mask(train_node_mask)
        return train_datas, valid_datas, test_data, client_names

class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
        self.criterion = torch.nn.MSELoss()
        self.DataLoader = DataLoader
        self.batches_seen = 0
        self.label_index = 2

    def train_one_step(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata, self.batches_seen, training=True)
        loss = self.criterion(outputs, tdata[self.label_index])
        self.batches_seen += 1
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        model.eval()
        data_loader = self.DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        ave_loss = 0
        for batch_idx, batch_data in enumerate(data_loader):
            tdata = self.data_to_device(batch_data)
            outputs = model(tdata, self.batches_seen, training=False)
            loss = self.criterion(outputs, tdata[self.label_index]).cpu().detach().numpy()
            ave_loss += loss*len(tdata[self.label_index])
        ave_loss/=len(dataset)
        return {'loss': ave_loss}

    def data_to_device(self, data):
        res = []
        for di in data:
            res.append(di.to(self.device))
        return tuple(res)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)







