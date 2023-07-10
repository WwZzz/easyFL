import os
import torch_geometric
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx
from flgo.benchmark.toolkits.graph.link_prediction import GeneralCalculator, FromDatasetPipe, FromDatasetGenerator
from .config import train_data
try:
    import ujson as json
except:
    import json
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None

class TaskGenerator(FromDatasetGenerator):
    def __init__(self, disjoint_train_ratio=0.0, neg_sampling_ratio=1.0, add_negative_train_samples=False):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)
        self.disjoin_train_ratio = disjoint_train_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.add_negative_train_samples = add_negative_train_samples

class TaskPipe(FromDatasetPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'disjoint_train_ratio': generator.disjoin_train_ratio, 'neg_sampling_ratio': generator.neg_sampling_ratio, 'add_negative_train_samples':generator.add_negative_train_samples}
        partitioner_name = generator.partitioner.__class__.__name__.lower()
        feddata['partition_by_edge'] = 'link' in partitioner_name or 'edge' in partitioner_name
        for cid in range(len(client_names)):
            feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        train_data = self.train_data
        neg_sampling_ratio = self.feddata['neg_sampling_ratio']
        disjoint_train_ratio = self.feddata['disjoint_train_ratio']
        add_negative_train_samples = self.feddata['add_negative_train_samples']
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
            ctrans = T.RandomLinkSplit(neg_sampling_ratio=neg_sampling_ratio, is_undirected=c_dataset.is_undirected(), num_test=num_test, num_val=num_val, add_negative_train_samples=add_negative_train_samples, disjoint_train_ratio=disjoint_train_ratio)
            cdata_train, cdata_val, cdata_test = ctrans(c_dataset)
            task_data[cname] = {'train': cdata_train, 'val': cdata_val if len(cdata_val.edge_label)>0 else None, 'test':cdata_test if len(cdata_test.edge_label)>0 else None}
        return task_data

TaskCalculator = GeneralCalculator