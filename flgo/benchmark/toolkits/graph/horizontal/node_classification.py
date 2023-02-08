import community.community_louvain
import torch_geometric.transforms as T
from torch_geometric.utils import mask_to_index, index_to_mask, from_networkx
import torch_geometric.utils
import collections
from benchmark.toolkits.base import *
import networkx as nx


class NodeClassificationTaskGen(BasicTaskGenerator):
    def __init__(self, benchmark, rawdata_path, builtin_class, dataset_name, transforms=None, num_clients=10):
        super(NodeClassificationTaskGen, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.num_clients = num_clients


    def load_data(self):
        self.all_data = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.2)(
            self.builtin_class(root=self.rawdata_path, name=self.dataset_name, transform=self.transforms).data)
        self.test_nodes = mask_to_index(self.all_data.test_mask)
        self.all_train_nodes = mask_to_index(self.all_data.train_mask)
        self.all_val_nodes = mask_to_index(self.all_data.val_mask)
        self.G = torch_geometric.utils.to_networkx(self.all_data, to_undirected=self.all_data.is_undirected(),
                                                   node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'])


    def partition(self):
        self.local_nodes = [[] for _ in range(self.num_clients)]
        node_groups = community.community_louvain.best_partition(self.G)
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
        return '_'.join(['B-'+self.benchmark,  'P-None', 'N-'+str(self.num_clients)])


class NodeClassificationTaskPipe(HorizontalTaskPipe):
    def __init__(self, task_name, buildin_class, transform=None):
        super(NodeClassificationTaskPipe, self).__init__(task_name)
        self.builtin_class = buildin_class
        self.transform = transform

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_nodes))
        feddata = {'client_names': client_names,
                   'server_data': generator.test_nodes.tolist(),
                   'rawdata_path': generator.rawdata_path,
                   'dataset_name': generator.dataset_name,
                   'all_train_nodes': generator.all_train_nodes.tolist(),
                   'all_val_nodes': generator.all_val_nodes.tolist()}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_nodes[cid]}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        self.all_data = self.builtin_class(root=self.feddata['rawdata_path'], name=self.feddata['dataset_name'],
                                           transform=self.transform).data
        self.all_data.test_mask = index_to_mask(torch.LongTensor(self.feddata['server_data']),
                                                size=self.all_data.num_nodes)
        self.all_data.train_mask = index_to_mask(torch.LongTensor(self.feddata['all_train_nodes']),
                                                 size=self.all_data.num_nodes)
        self.all_data.val_mask = index_to_mask(torch.LongTensor(self.feddata['all_val_nodes']),
                                               size=self.all_data.num_nodes)
        task_data = {'server': {'test': self.all_data}}
        G = torch_geometric.utils.to_networkx(self.all_data, to_undirected=self.all_data.is_undirected(),
                                              node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'])
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = from_networkx(nx.subgraph(G, self.feddata[cname]['data']))
            cdata.test_mask = cdata.val_mask
            task_data[cname] = {'train': cdata, 'valid': cdata}
        return task_data

"""
load_data -> return task_data = {
    'server': {'test': anything, 'xxx': anything},
    'Client01': {'train': anything, ...}
    ...
}

generate_objects -> [object1, object2, ...]
object1.name = task_data[0] = 'server'
object2.name = task_data[1] = 'Client01'
....

distribute:
    specify the object according to the name of the object
        object_x
        x_data = task_data[x_name] (i.e. {'xxx':anything, ...})
        for key in x_data:
            object_x.set_data(key, x_data[key])
            
set_data(data_name, data):
    setattr(self, data_name+'_data', data)
    
"""
class NodeClassificationTaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(NodeClassificationTaskCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.NLLLoss()
        self.DataLoader = torch_geometric.loader.DataLoader

    def compute_loss(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        loss = self.criterion(outputs[tdata.train_mask], tdata.y[tdata.train_mask])
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        loader = self.DataLoader([dataset], batch_size=batch_size)
        total_loss = 0
        total_num_samples = 0
        for batch in loader:
            tdata = self.data_to_device(batch)
            outputs = model(tdata)
            loss = self.criterion(outputs[tdata.test_mask], tdata.y[tdata.test_mask])
            num_samples = len(tdata.x)
            total_loss += num_samples * loss
            total_num_samples += num_samples
        total_loss = total_loss.item()
        return {'loss': total_loss / total_num_samples}

    def data_to_device(self, data):
        return data.to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        return self.DataLoader([dataset], batch_size=batch_size, shuffle=shuffle)
