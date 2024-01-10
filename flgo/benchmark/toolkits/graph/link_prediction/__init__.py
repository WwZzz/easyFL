import copy
import networkx as nx
import torch_geometric
from torch_geometric.transforms import RandomLinkSplit
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling, from_networkx
import torch_geometric.utils
import torch
import flgo.benchmark.base
import os
try:
    import ujson as json
except:
    import json
from sklearn.metrics import roc_auc_score, average_precision_score

class FromDatasetGenerator(flgo.benchmark.base.FromDatasetGenerator):
    def prepare_data_for_partition(self):
        edge_attrs=['edge_attr'] if hasattr(self.train_data, 'edge_attr') and self.train_data.edge_attr is not None else None
        return torch_geometric.utils.to_networkx(self.train_data, to_undirected=self.train_data.is_undirected(), node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'], edge_attrs=edge_attrs)

class FromDatasetPipe(flgo.benchmark.base.FromDatasetPipe):
    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names,}
        partitioner_name = generator.partitioner.__class__.__name__.lower()
        feddata['partition_by_edge'] = 'link' in partitioner_name or 'edge' in partitioner_name
        for cid in range(len(client_names)):
            feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        train_data = self.train_data
        neg_sampling_ratio = train_data.neg_sampling_ratio if hasattr(train_data, 'neg_sampling_ratio') else 1.0
        disjoint_train_ratio = train_data.disjoint_train_ratio if hasattr(train_data, 'disjoint_train_ratio') else 0.0
        test_data = self.test_data
        val_data = self.val_data
        server_test_data = test_data
        server_val_data = val_data
        if server_test_data is not None:
            if val_data is None:
                if running_time_option['test_holdout']>0:
                    num_test,num_val = 1 - running_time_option['test_holdout'], running_time_option['test_holdout'] * 0.2
                else:
                    num_test, num_val = 0.2,0.0
                # split test_data
                server_val_data, _, server_test_data = T.RandomLinkSplit(
                    neg_sampling_ratio=neg_sampling_ratio, is_undirected=server_test_data.is_undirected(),
                    num_test=num_test, num_val=num_val)(server_test_data)
                if num_val==0.0: server_val_data = None
            else:
                _tmp1, _tmp2, server_test_data = T.RandomLinkSplit(
                        neg_sampling_ratio=neg_sampling_ratio, is_undirected=server_test_data.is_undirected(),
                        num_test=0.2, num_val=0.0)(server_test_data)
                _tmp1, _tmp2, server_val_data = T.RandomLinkSplit(
                    neg_sampling_ratio=neg_sampling_ratio, is_undirected=server_val_data.is_undirected(),
                    num_test=0.2, num_val=0.0)(server_val_data)
        elif server_val_data is not None:
            _tmp1, _tmp2, server_val_data = T.RandomLinkSplit(
                neg_sampling_ratio=neg_sampling_ratio, is_undirected=server_val_data.is_undirected(),
                num_test=0.2, num_val=0.0)(server_val_data)
        task_data = {'server': {'test': server_test_data, 'val': server_val_data}}
        edge_attrs=['edge_attr'] if hasattr(train_data, 'edge_attr') and train_data.edge_attr is not None else None
        G = torch_geometric.utils.to_networkx(train_data, to_undirected=train_data.is_undirected(), node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'], edge_attrs=edge_attrs)
        num_val = running_time_option['train_holdout']
        if num_val>0 and running_time_option['local_test']:
            num_test, num_val = 0.5*num_val, 0.5*num_val
        else:
            num_test = 0.0
        all_edges = list(G.edges)
        for cid, cname in enumerate(self.feddata['client_names']):
            if self.feddata['partition_by_edge']:
                c_dataset = from_networkx(nx.edge_subgraph(G, [all_edges[eid] for eid in self.feddata[cname]['data']]))
            else:
                c_dataset = from_networkx(nx.subgraph(G, self.feddata[cname]['data']))
            ctrans = T.RandomLinkSplit(neg_sampling_ratio=neg_sampling_ratio, is_undirected=c_dataset.is_undirected(), num_test=num_test, num_val=num_val, add_negative_train_samples=False, disjoint_train_ratio=disjoint_train_ratio)
            cdata_train, cdata_val, cdata_test = ctrans(c_dataset)
            task_data[cname] = {'train': cdata_train, 'val': cdata_val if len(cdata_val.edge_label)>0 else None, 'test':cdata_test if len(cdata_test.edge_label)>0 else None}
        return task_data

class BuiltinClassGenerator(flgo.benchmark.base.BasicTaskGenerator):
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

class BuiltinClassPipe(flgo.benchmark.base.BasicTaskPipe):
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
            server_val_data,_,server_test_data = T.RandomLinkSplit(neg_sampling_ratio=self.feddata['neg_sampling_ratio'],is_undirected=self.dataset.is_undirected(),num_test=1-running_time_option['test_holdout'], num_val=0.0, add_negative_train_samples=True)(test_dataset)
            if len(server_val_data.edge_label)==0:server_val_data = None
            if len(server_test_data.edge_label)==0:server_test_data = None
        else:
            server_test_data,server_val_data = None,None
        task_data = {'server': {'test': server_test_data, 'val': server_val_data}}
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
            cdata_train, cdata_val, cdata_test = ctrans(c_dataset)
            task_data[cname] = {'train': cdata_train, 'val': cdata_val if len(cdata_val.edge_label)>0 else None, 'test':cdata_test if len(cdata_test.edge_label)>0 else None}
        return task_data

class GeneralCalculator(flgo.benchmark.base.BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.DataLoader = torch_geometric.loader.DataLoader

    def compute_loss(self, model, data):
        model.train()
        tdata = self.to_device(data)
        outputs, neg_edge_index = model(tdata)
        edge_label = torch.cat([
            tdata.edge_label,
            tdata.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
        loss = self.criterion(outputs, edge_label)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        res = {}
        model.eval()
        tdata = self.to_device(dataset)
        outputs = model(tdata)
        loss = self.criterion(outputs, tdata.edge_label)
        res['loss'] = loss.item()
        sigmoid_out = outputs.sigmoid()
        res['ap'] = average_precision_score(tdata.edge_label.cpu().numpy(), sigmoid_out.cpu().numpy())
        if len(set(tdata.edge_label.cpu().tolist()))>1:
            res['roc_auc'] = roc_auc_score(tdata.edge_label.cpu().numpy(), sigmoid_out.cpu().numpy())
        return res

    def to_device(self, data):
        return data.to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        return self.DataLoader([dataset], batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
