import copy
import community.community_louvain
import networkx as nx
import torch_geometric
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling, from_networkx
import torch_geometric.utils
import collections

from flgo.benchmark.base import *

class BuiltinClassGenerator(BasicTaskGenerator):
    def __init__(self, benchmark, rawdata_path, builtin_class, transform=None, pre_transform=None, test_rate=0.2, test_node_split=True, disjoint_train_ratio=0.3, neg_sampling_ratio=1.0):
        super(BuiltinClassGenerator, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.transform = transform
        self.pre_transform = pre_transform
        self.test_rate = test_rate
        self.test_node_split = test_node_split
        self.disjoint_train_ratio = disjoint_train_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.download = False

    def load_data(self):
        default_init_para = {'root': self.rawdata_path, 'download':self.download, 'train':True, 'transform':self.transform, 'pre_transform':self.pre_transform}
        default_init_para.update(self.additional_option)
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            pop_key = [k for k in default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in pop_key: default_init_para.pop(k)
        self.dataset = self.builtin_class(**default_init_para).data
        transform_for_create_test = T.RandomLinkSplit(neg_sampling_ratio=0.0, is_undirected=self.dataset.is_undirected(), num_test=self.test_rate, num_val=0, add_negative_train_samples=False)
        _,_,test_data = transform_for_create_test(self.dataset)
        self.test_nodes = list(set(test_data.edge_label_index.view(-1).tolist()))
        all_nodes = list(set(self.dataset.edge_index.view(-1).tolist()))
        self.train_nodes = list(set(all_nodes).difference(self.test_nodes)) if self.test_node_split else all_nodes
        self.G = torch_geometric.utils.to_networkx(self.dataset, to_undirected=self.dataset.is_undirected(), node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'])
        self.train_data = nx.subgraph(self.G, self.train_nodes)

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)

    def get_task_name(self):
        return '_'.join(['B-' + self.benchmark, 'P-None', 'N-' + str(self.num_clients)])

class BuiltinClassPipe(BasicTaskPipe):
    def __init__(self, task_name, builtin_class, transform=None, pre_transform=None):
        super(BuiltinClassPipe, self).__init__(task_name)
        self.builtin_class = builtin_class
        self.pre_transform = pre_transform
        self.transform = transform

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': generator.test_nodes,
                   'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option,
                   'train_additional_option': generator.train_additional_option,
                   'test_additional_option': generator.test_additional_option,
                   'test_rate': generator.test_rate,
                   'disjoint_train_ratio': generator.disjoint_train_ratio,
                   'neg_sampling_ratio': generator.neg_sampling_ratio
                   }
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        default_init_para = {'root': self.feddata['rawdata_path'], 'download': True, 'train': True, 'transform': self.transform, 'pre_transform':self.pre_transform}
        default_init_para.update(self.feddata['additional_option'])
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            pop_key = [k for k in default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in pop_key: default_init_para.pop(k)
        self.dataset = self.builtin_class(**default_init_para).data
        self.G = torch_geometric.utils.to_networkx(self.dataset, to_undirected=self.dataset.is_undirected(), node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'])
        if self.feddata['test_rate']>0:
            test_dataset = from_networkx(nx.subgraph(self.G, self.feddata['server_data']))
            server_valid_data,_,server_test_data = T.RandomLinkSplit(neg_sampling_ratio=self.feddata['neg_sampling_ratio'],is_undirected=self.dataset.is_undirected(),num_test=1-running_time_option['test_holdout'], num_val=0.0, add_negative_train_samples=True)(test_dataset)
            if len(server_valid_data.edge_label)==0:server_valid_data = None
            if len(server_test_data.edge_label)==0:server_test_data = None
        else:
            server_test_data,server_valid_data = None,None
        task_data = {'server': {'test': server_test_data, 'valid': server_valid_data}}
        # rearrange data for clients
        if running_time_option['local_test']:
            num_val = running_time_option['train_holdout']*0.5
            num_test = running_time_option['train_holdout']*0.5
        else:
            num_val = running_time_option['train_holdout']
            num_test = 0
        for cid, cname in enumerate(self.feddata['client_names']):
            c_dataset = from_networkx(nx.subgraph(self.G, self.feddata[cname]['data']))
            ctrans = RandomLinkSplit(neg_sampling_ratio=self.feddata['neg_sampling_ratio'],is_undirected=self.dataset.is_undirected(), num_test=num_test, num_val=num_val,
                                        add_negative_train_samples=False, disjoint_train_ratio=self.feddata['disjoint_train_ratio'])
            cdata_train, cdata_valid, cdata_test = ctrans(c_dataset)
            task_data[cname] = {'train': cdata_train, 'valid': cdata_valid if len(cdata_valid.edge_label)>0 else None, 'test':cdata_test if len(cdata_test.edge_label)>0 else None}
        return task_data

class GeneralCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.DataLoader = torch_geometric.loader.DataLoader

    def compute_loss(self, model, data):
        tdata = self.data_to_device(data)
        z = model.encode(tdata.x, tdata.edge_label_index)
        neg_edge_index = negative_sampling(
            edge_index=tdata.edge_index, num_nodes=tdata.num_nodes,
            num_neg_samples=tdata.edge_label_index.size(1)
        )
        edge_label_index = torch.cat(
            [tdata.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            tdata.edge_label,
            tdata.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        outputs = model.decode(z, edge_label_index).view(-1)
        loss = self.criterion(outputs, edge_label)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        total_loss = 0
        total_num_samples = 0
        tdata = self.data_to_device(dataset)
        z = model.encode(tdata.x, tdata.edge_index)
        outputs = model.decode(z, tdata.edge_label_index).view(-1)
        loss = self.criterion(outputs, tdata.edge_label)
        num_samples = len(tdata.x)
        total_loss += num_samples * loss
        total_num_samples += num_samples
        total_loss = total_loss.item()
        return {'loss': total_loss / total_num_samples}

    def data_to_device(self, data):
        return data.to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False):
        return self.DataLoader([dataset], batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
