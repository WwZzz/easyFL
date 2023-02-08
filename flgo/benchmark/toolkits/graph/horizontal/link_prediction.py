import copy

import community.community_louvain
import networkx as nx
import torch_geometric
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling, from_networkx
import torch_geometric.utils
import collections

from benchmark.toolkits.base import *


class LinkPredicitonTaskGenenerator(BasicTaskGenerator):
    def __init__(self, benchmark, rawdata_path, builtin_class, dataset_name, transforms=None, num_clients=10):
        super(LinkPredicitonTaskGenenerator, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.num_clients = num_clients

    def load_data(self):
        all_data = self.builtin_class(root=self.rawdata_path, name=self.dataset_name, transform=self.transforms).data
        transform = RandomLinkSplit(is_undirected=all_data.is_undirected(), num_test=0.2, num_val=0,
                                    add_negative_train_samples=False)
        local_data, _, test_data = transform(all_data)
        self.local_data_edge_label_index = local_data.edge_label_index
        self.test_data_edge_label_index = test_data.edge_label_index
        self.test_data_edge_label = test_data.edge_label
        self.train_G = torch_geometric.utils.to_networkx(all_data, to_undirected=all_data.is_undirected(),
                                                         node_attrs=['x'])

    def partition(self):
        self.local_nodes = [[] for _ in range(self.num_clients)]
        node_groups = community.community_louvain.best_partition(self.train_G)
        groups = collections.defaultdict(list)
        for ni, gi in node_groups.items():
            groups[gi].append(ni)
        groups = {k: groups[k] for k in list(range(len(groups)))}
        # ensure the number of groups is larger than the number of clients
        while len(groups) < self.num_clients:
            # find the group with the largest size
            groups_lens = [groups[k] for k in range(len(groups))]
            max_gi = np.argmax(groups_lens)
            # set the size of the new group
            min_glen = min(groups_lens)
            max_glen = max(groups_lens)
            if max_glen < 2 * min_glen: min_glen = max_glen // 2
            # split the group with the largest size into two groups
            nodes_in_gi = groups[max_gi]
            new_group_id = len(groups)
            groups[new_group_id] = nodes_in_gi[:min_glen]
            groups[max_gi] = nodes_in_gi[min_glen:]
        # allocate different groups to clients
        groups_lens = [groups[k] for k in range(len(groups))]
        group_ids = np.argsort(groups_lens)
        for gi in group_ids:
            cid = np.argmin([len(li) for li in self.local_nodes])
            self.local_nodes[cid].extend(groups[gi])

    def get_task_name(self):
        return '_'.join(['B-' + self.benchmark, 'P-None', 'N-' + str(self.num_clients)])


class LinkPredicitonTaskPipe(HorizontalTaskPipe):
    def __init__(self, task_name, buildin_class, transform=None):
        super(LinkPredicitonTaskPipe, self).__init__(task_name)
        self.builtin_class = buildin_class
        self.transform = transform

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_nodes))
        feddata = {'client_names': client_names,
                   'server_data':
                       {'edge_label_index': generator.test_data_edge_label_index.tolist(),
                        'edge_label': generator.test_data_edge_label.tolist()},
                   'rawdata_path': generator.rawdata_path,
                   'dataset_name': generator.dataset_name,
                   'local_edge_label_index': generator.local_data_edge_label_index.tolist()}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_nodes[cid]}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        all_data = self.builtin_class(root=self.feddata['rawdata_path'], name=self.feddata['dataset_name'],
                                           transform=self.transform).data
        server_data = copy.deepcopy(all_data)
        server_data.edge_label_index = torch.Tensor(self.feddata['server_data']['edge_label_index'])
        server_data.edge_label = torch.Tensor(self.feddata['server_data']['edge_label'])

        local_data = copy.deepcopy(all_data)
        local_data_edge_label_index = torch.Tensor(self.feddata['local_edge_label_index'])
        edge_index = torch.cat([local_data_edge_label_index,
                                           torch.flip(local_data_edge_label_index, dims=[0])], dim=1)
        local_data.edge_index = edge_index
        server_data.edge_index = edge_index
        task_data = {'server': {'test': server_data}}
        G = torch_geometric.utils.to_networkx(local_data, to_undirected=local_data.is_undirected(),
                                              node_attrs=['x'])
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = from_networkx(nx.subgraph(G, self.feddata[cname]['data']))
            transform = RandomLinkSplit(is_undirected=cdata.is_undirected(), num_test=running_time_option['train_holdout'], num_val=0,
                                        add_negative_train_samples=False)
            train_data, _, val_data = transform(cdata)
            task_data[cname] = {'train': train_data, 'valid': val_data}
        return task_data


class LinkPredicitonTaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(LinkPredicitonTaskCalculator, self).__init__(device, optimizer_name)
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
    def test(self, model, dataset, batch_size=64, num_workers=0):
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

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        return self.DataLoader([dataset], batch_size=batch_size, shuffle=shuffle)
